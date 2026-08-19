#!/usr/bin/env python3
"""
常駐監控迴圈 —— 每根 5min K 線收盤後 +2 秒觸發一次即時推論，印出買漲/買跌、
信心度、進場價、訊號 K 線時間、結算時間。純觀察 + 手動下單，不接自動下單，
不碰任何 API key。

用法：
  .venv/bin/python predict_live.py --data_dir ./dataset/event30m
  .venv/bin/python predict_live.py --data_dir ./dataset/event30m --once   # 只跑一次就結束，測試用
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'toolbywi'))

from event_common import load_meta, build_and_load_model, new_cluster_state, cluster_step
from feature_scale import features, microstructure_features, alt_features, extra_features, load_scaler, apply_scaler
from binance_live import (get_latest_closed_klines, get_current_price, StaleDataError,
                           get_live_metrics, get_live_funding, get_live_premium_index, get_live_mark_price)
from utils.timefeatures import time_features

WARMUP_BARS = 500  # 指標暖機需求上限（ADX_156 最長），含餘裕
PRICE_DRIFT_STALE_PCT = 0.1  # 訊號價與即時價偏離超過此百分比就視為過期


def build_live_window(meta, sc, feature_cols):
    symbol = meta['symbol']
    alt_symbols = meta['alt_symbols']
    seq_len = meta['seq_len']
    total_bars = seq_len + WARMUP_BARS + 1  # +1 給丟棄未收盤那根留餘裕

    df_btc = get_latest_closed_klines(symbol, total_bars=total_bars)
    alt_dfs = {}
    for sym in alt_symbols:
        df_alt = get_latest_closed_klines(sym, total_bars=total_bars)
        alt_dfs[sym.replace('USDT', '')] = df_alt

    signal_close = float(df_btc['Close'].iloc[-1])
    signal_time = df_btc['date'].iloc[-1]

    df_feat = features(df_btc)

    # Round 4：模型若是用微結構特徵（Open/High/Low/Quote_asset_volume 衍生）
    # 訓練的，meta.json 會有 microstructure_features 欄位——依此決定要不要
    # 算這 6 個特徵，Round 1-3 的舊 meta.json 沒有這個欄位，預設不算，行為
    # 跟原本完全一樣。呼叫順序跟 prepare_event_data.py 對齊：features() 之後、
    # alt_features() 之前。
    if meta.get('microstructure_features'):
        df_feat = microstructure_features(df_feat)

    df_feat = alt_features(df_feat, alt_dfs)

    # Round 3：模型若是用新特徵來源（metrics/funding/premium/mark）訓練的，
    # meta.json 會有 with_extra_sources=True——依此決定要不要抓這三組即時
    # 資料，Round 1/2 的舊 meta.json 沒有這個欄位，預設當作 False，行為
    # 跟原本完全一樣。
    if meta.get('with_extra_sources'):
        metrics_df = get_live_metrics(symbol, total_bars=total_bars)
        funding_df = get_live_funding(symbol)
        premium_df = get_live_premium_index(symbol, total_bars=total_bars)
        mark_df = get_live_mark_price(symbol, total_bars=total_bars)
        df_feat = extra_features(df_feat, metrics_df, funding_df, premium_df, mark_df)

    # extra_features() 結尾會 dropna()（inner join 缺一根就整根丟），可能把
    # 最新那根連同其 metrics/premium/mark 一起刪掉——若真的發生，df_feat 最
    # 後一根就不是 signal_time 那根了，印出來的訊號時間會跟模型實際看到的
    # 資料對不上。這裡直接斷言，寧可拋錯重試，也不要靜默印出錯位的訊號。
    latest_feat_date = pd.Timestamp(df_feat['date'].iloc[-1])
    if latest_feat_date != pd.Timestamp(signal_time):
        raise StaleDataError(
            f"特徵表最新一根（{latest_feat_date}）與訊號 K 線時間（{signal_time}）不符，"
            f"可能是 extra_features() 的 dropna() 把最新一根連同缺漏的新來源資料一起丟掉了。"
            f"寧可這一輪跳過，也不要印出時間對不上的訊號。"
        )

    missing = [c for c in feature_cols if c not in df_feat.columns]
    if missing:
        raise RuntimeError(
            f"即時特徵表缺少訓練時的欄位: {missing}。"
            f"代表特徵工程程式碼跟訓練時已經不一致，直接停止，不做任何猜測補值。"
        )

    df_feat = apply_scaler(df_feat, sc, feature_cols)

    if len(df_feat) < seq_len:
        raise RuntimeError(f"暖機後只剩 {len(df_feat)} 根，不足 seq_len={seq_len}，資料抓太少。")

    window = df_feat[feature_cols].tail(seq_len).values.astype(np.float32)
    dates = pd.to_datetime(df_feat['date']).tail(seq_len)

    # 補 target=0 欄，效果對應 Dataset_Custom.__getitem__ 的 seq_x[:, -1] = 0 遮罩
    target_col = np.zeros((seq_len, 1), dtype=np.float32)
    x_enc = np.concatenate([window, target_col], axis=1)  # [seq_len, n_features+1]

    stamp = time_features(pd.to_datetime(dates.values), freq='t')  # [n_time_feat, seq_len]
    x_mark_enc = stamp.transpose(1, 0).astype(np.float32)  # [seq_len, n_time_feat]

    return x_enc, x_mark_enc, signal_close, signal_time


def run_once(meta, exp, model_args, sc, feature_cols, threshold, gap, entry_at,
             cluster_state, bar_idx):
    """
    跑一輪即時推論，並把 p_up 餵進叢集狀態機（cluster_step，語意與
    evaluate_event.py 的批次版 cluster_entries() 完全一致，已用自我測試驗證）。
    只認買漲方向：p_up < threshold 的根完全不影響叢集狀態。

    回傳 (result_dict, new_cluster_state)。
    """
    x_enc, x_mark_enc, signal_close, signal_time = build_live_window(meta, sc, feature_cols)

    x_enc_t = torch.from_numpy(x_enc).unsqueeze(0).to(exp.device)
    x_mark_t = torch.from_numpy(x_mark_enc).unsqueeze(0).to(exp.device)

    label_len = model_args.label_len
    pred_len = model_args.pred_len
    # dec_inp/dec_mark 對 iTransformer 完全不影響輸出（模型只用 x_enc/x_mark_enc，
    # 已由 models/iTransformer.py 的 forecast() 原始碼核對過），這裡只是維持介面一致。
    dec_inp = torch.zeros((1, label_len + pred_len, x_enc.shape[-1]),
                           dtype=torch.float32, device=exp.device)
    dec_mark = torch.zeros((1, label_len + pred_len, x_mark_enc.shape[-1]),
                            dtype=torch.float32, device=exp.device)

    with torch.no_grad():
        outputs = exp.model(x_enc_t, x_mark_t, dec_inp, dec_mark)
    logit = outputs[0, -1, -1].item()
    p_up = float(expit(logit))

    new_state, should_enter = cluster_step(cluster_state, bar_idx, p_up, threshold, gap, entry_at)

    settlement_time = signal_time + pd.Timedelta(minutes=5 * meta['lookahead'])

    live_price, drift_pct, stale = None, None, True
    try:
        live_price = get_current_price(meta['symbol'])
        drift_pct = abs(live_price - signal_close) / signal_close * 100
        stale = drift_pct > PRICE_DRIFT_STALE_PCT
    except Exception:
        pass

    now = pd.Timestamp.now('UTC').tz_localize(None)

    if should_enter:
        icon, label = '\U0001F7E2', '買漲進場'
    elif p_up >= threshold:
        icon, label = '⚪', '買漲訊號但叢集判斷不進場（同叢集已進場，或未輪到 entry_at）'
    else:
        icon, label = '⚪', '觀望'

    print(f"[{now:%Y-%m-%d %H:%M:%S}] {icon} {label}  p_up={p_up:.3f}  門檻={threshold:.2f}  "
          f"叢集計數={new_state['cluster_count']}  entry_at={entry_at}")
    print(f"  訊號 K 線收盤: {signal_time} UTC, close={signal_close:.1f}")
    if should_enter:
        print(f"  結算時間: {settlement_time} UTC")
    if live_price is not None:
        flag = ""
        if should_enter and stale:
            flag = f" -- 訊號可能已過期（價格偏離 > {PRICE_DRIFT_STALE_PCT}%），下單前請重新確認即時報價"
        print(f"  即時參考價: {live_price:.1f}（與訊號價偏離 {drift_pct:.3f}%）{flag}")
    else:
        print("  無法取得即時參考價")

    result = {
        'time': str(now), 'should_enter': should_enter, 'p_up': p_up,
        'signal_close': signal_close, 'signal_time': str(signal_time),
        'settlement_time': str(settlement_time) if should_enter else None,
        'live_price': live_price, 'drift_pct': drift_pct, 'stale': stale,
    }
    return result, new_state


def sleep_until_next_bar(offset_sec=2):
    now = datetime.now(timezone.utc)
    next_minute = (now.minute // 5 + 1) * 5
    next_close = now.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=next_minute)
    target = next_close + timedelta(seconds=offset_sec)
    sleep_s = (target - now).total_seconds()
    if sleep_s > 0:
        time.sleep(sleep_s)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default='./dataset/event30m')
    p.add_argument('--threshold', type=float, default=None,
                    help='不給則用 evaluate_report.json 鎖定的門檻')
    p.add_argument('--entry_at', type=int, default=None,
                    help='不給則用 evaluate_report.json 鎖定的 entry_at')
    p.add_argument('--once', action='store_true', help='只跑一次就結束（測試/驗證用）')
    p.add_argument('--offset_sec', type=int, default=2, help='K 線收盤後等幾秒再抓資料')
    p.add_argument('--setting', default=None,
                    help='不給則用 meta.json 頂層 setting；同一份 meta.json 底下訓練過多個'
                         '超參數組合時，用這個明確指定要上線的是哪一個 checkpoint')
    return p.parse_args()


def main():
    args = parse_args()
    meta = load_meta(args.data_dir)
    gap = meta['lookahead']

    threshold, entry_at = args.threshold, args.entry_at
    if threshold is None or entry_at is None:
        eval_report_path = os.path.join(args.data_dir, 'evaluate_report.json')
        if not os.path.exists(eval_report_path):
            raise RuntimeError(
                f"沒有指定 --threshold/--entry_at，也找不到 {eval_report_path}。"
                f"請先跑 evaluate_event.py 鎖定門檻，或用 --threshold/--entry_at 手動指定。"
            )
        eval_report = json.load(open(eval_report_path))
        if not eval_report.get('passed'):
            print("警告：evaluate_event.py 的關卡沒有通過，這個模型統計上無法證明能穩定打贏抽水。")
        if threshold is None:
            threshold = eval_report.get('chosen_threshold')
        if entry_at is None:
            # 'entry_at'：Round 3 起固定值（不再網格選擇，見 evaluate_event.py）。
            # 'chosen_entry_at'：向下相容 Round 2 的 evaluate_report.json。
            entry_at = eval_report.get('entry_at', eval_report.get('chosen_entry_at'))
        if threshold is None or entry_at is None:
            raise RuntimeError("evaluate_report.json 裡沒有鎖定的 threshold/entry_at，無法繼續。")

    print("=== 常駐監控啟動 ===")
    print(f"symbol={meta['symbol']}  seq_len={meta['seq_len']}  gap={gap}  "
          f"門檻={threshold:.2f}  entry_at={entry_at}")
    print("只認買漲方向：p_up >= 門檻才算訊號，且需符合叢集/entry_at 條件才真的進場。")

    exp, model_args = build_and_load_model(meta, args.data_dir, setting=args.setting)
    sc, sc_feature_cols = load_scaler(os.path.join(os.path.abspath(args.data_dir), meta['scaler_path']))
    feature_cols = meta['feature_cols']
    assert sc_feature_cols == feature_cols, "scaler 與 meta 的 feature_cols 不一致，不應發生"

    # 叢集狀態放迴圈外的記憶體變數：單一持續執行的 process，不需要外部持久化。
    cluster_state = new_cluster_state()
    bar_idx = 0

    while True:
        try:
            _, cluster_state = run_once(meta, exp, model_args, sc, feature_cols,
                                         threshold, gap, entry_at, cluster_state, bar_idx)
        except StaleDataError as e:
            print(f"資料陳舊，跳過這一輪: {e}")
        except Exception as e:
            print(f"這一輪推論失敗: {e}")

        bar_idx += 1  # 不論這輪成功與否，時間都往前走了一根，叢集間隔判斷要跟真實時間對齊
        if args.once:
            break
        sleep_until_next_bar(args.offset_sec)


if __name__ == '__main__':
    main()
