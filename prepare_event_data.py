#!/usr/bin/env python3
"""
資料準備：下載主幣 + 輔幣 5min K 線 → 建立滾動方向標籤（事件合約用）→
算技術指標 → Z-Score 縮放 → 輸出訓練用 CSV + scaler + meta.json。

標籤定義：target[i] = 1 若 Close[i+lookahead] > Close[i]，否則 0。
lookahead=6 根 5min K 線 = 30 分鐘，對應幣安事件合約 30min 結算週期。

輸出到 --out_dir（預設 ./dataset/event30m/）：
  raw_ohlcv.csv       未縮放真實價格 + target（評估/回測用）
  stock_features.csv  訓練用最終特徵表（已縮放，target 在最後一欄）
  scaler.joblib        fitted StandardScaler + feature_cols
  meta.json             欄序、seq_len、lookahead、類別比例、切分索引等

用法：
  .venv/bin/python prepare_event_data.py --start_year 2025 --start_month 1 \
      --end_year 2026 --end_month 8
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toolbywi'))
from download_binance_monthly_batch import binance_load_5min_fast
from binance_extra import load_metrics, load_funding_rate, load_premium_index, load_mark_price
from target_calulate import target_direction, target_pct_threshold, target_run_center
from feature_scale import (features, microstructure_features, alt_features, extra_features,
                            scaler as zscore_scaler, save_scaler)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--symbol', default='BTCUSDT')
    p.add_argument('--alt_symbols', default='ETHUSDT,BNBUSDT,SOLUSDT')
    p.add_argument('--start_year', type=int, default=2025)
    p.add_argument('--start_month', type=int, default=1)
    p.add_argument('--end_year', type=int, default=2026)
    p.add_argument('--end_month', type=int, default=8)
    p.add_argument('--lookahead', type=int, default=6,
                    help='結算視野（根）。Round 1-4 用 6（30 分鐘）；Round 5 起 1 小時用 12')
    p.add_argument('--seq_len', type=int, default=1024)
    p.add_argument('--train_ratio', type=float, default=0.7)
    p.add_argument('--out_dir', default='./dataset/event30m')
    p.add_argument('--with_extra_sources', action=argparse.BooleanOptionalAction, default=True,
                    help='是否加入 Round 3 新特徵來源（metrics/funding/premium/mark），預設開')
    # --- Round 5 標籤選項（預設 direction，與 Round 1-4 行為完全相同）---
    p.add_argument('--label_mode', default='direction', choices=['direction', 'run_center'],
                    help="direction=Round 1-4 的漲跌標籤；run_center=幅度門檻 + 連續段取中間")
    p.add_argument('--min_pct', type=float, default=0.003,
                    help='run_center 模式：漲幅需 >= 此比例才算正樣本（0.003 = 0.3%%）')
    p.add_argument('--min_run', type=int, default=5,
                    help='run_center 模式：連續段長度下限，只有 >= 此值的段落才取中間')
    p.add_argument('--min_pct_floor', type=float, default=0.001,
                    help='尾端設限用的最寬鬆門檻（= 參數網格的最小 min_pct）。'
                         '因為 y_pct 對 min_pct 單調，用最小門檻設限一次即可支配整個網格，'
                         '讓所有組合共用同一組有效列與切分，比較才公平')
    p.add_argument('--label_grid', default='0.001,0.002,0.003,0.004,0.005|3,4,5,6,8',
                    help="run_center 模式額外輸出的參數網格標籤欄（'pct1,pct2|run1,run2'），"
                         "寫進 labels.csv 供 label_grid_search.py 使用；設為空字串則不輸出")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    alt_symbols = [s.strip() for s in args.alt_symbols.split(',') if s.strip()]

    print(f"=== 下載 {args.symbol}（主幣） ===")
    df_btc = binance_load_5min_fast(args.symbol, args.start_year, args.start_month,
                                     args.end_year, args.end_month)
    if df_btc is None or len(df_btc) == 0:
        raise RuntimeError(f"{args.symbol} 沒有抓到任何資料，中止。")
    print(f"{args.symbol}: {len(df_btc)} 根")

    alt_dfs = {}
    for sym in alt_symbols:
        print(f"=== 下載 {sym}（輔幣） ===")
        df_alt = binance_load_5min_fast(sym, args.start_year, args.start_month,
                                         args.end_year, args.end_month)
        if df_alt is None or len(df_alt) == 0:
            raise RuntimeError(f"{sym} 沒有抓到任何資料，中止。")
        print(f"{sym}: {len(df_alt)} 根")
        alt_dfs[sym.replace('USDT', '')] = df_alt

    # --- 標籤：在原始資料上算，Close 沒有指標 warmup 的 NaN、也沒有 join 造成的
    #     內部缺口，是唯一能正確判定「連續」的網格（run_center 模式必須如此）---
    print(f"=== 建立標籤（lookahead={args.lookahead} 根 = {args.lookahead * 5} 分鐘，"
          f"mode={args.label_mode}）===")
    df_btc = df_btc.copy()

    # contract_target 永遠是「漲就贏」的真實合約結算，與訓練標籤無關；
    # fwd_ret 供報告診斷「實際漲幅有沒有達到 min_pct」。兩者都不可進入
    # stock_features.csv（會被當成額外特徵而洩漏未來），稍後在
    # LABEL_AUX_COLS 取出。
    contract, valid_dir, fwd_ret = target_pct_threshold(
        df_btc, lookahead=args.lookahead, min_pct=0.0)
    contract = np.zeros(len(df_btc), dtype=int)
    contract[valid_dir] = (fwd_ret[valid_dir] > 0).astype(int)  # 與 target_direction 同定義（嚴格 >）

    if args.label_mode == 'direction':
        target, valid = target_direction(df_btc, lookahead=args.lookahead)
    else:
        target, valid, fwd_ret = target_run_center(
            df_btc, lookahead=args.lookahead, min_pct=args.min_pct,
            min_run=args.min_run, min_pct_floor=args.min_pct_floor)
        print(f"  幅度門檻 {args.min_pct*100:.2f}%、連續段下限 {args.min_run} 根"
              f"（奇數取中間 1 根、偶數取中間 2 根）")

    df_btc['target'] = target
    df_btc['contract_target'] = contract
    df_btc['fwd_ret'] = fwd_ret

    # run_center 模式額外算整個參數網格的標籤欄，全部共用同一個 min_pct_floor
    # 設限（見 target_run_center 的 R3），所以 valid 遮罩完全一致
    grid_cols = []
    if args.label_mode == 'run_center' and args.label_grid.strip():
        pct_s, run_s = args.label_grid.split('|')
        pcts = [float(x) for x in pct_s.split(',')]
        runs = [int(x) for x in run_s.split(',')]
        print(f"  另外產生 {len(pcts)}x{len(runs)}={len(pcts)*len(runs)} 組網格標籤欄")
        for p in pcts:
            for r in runs:
                col = f'y_bp{int(round(p*10000)):02d}_r{r}'
                g_tgt, g_valid, _ = target_run_center(
                    df_btc, lookahead=args.lookahead, min_pct=p, min_run=r,
                    min_pct_floor=args.min_pct_floor)
                assert np.array_equal(g_valid, valid), (
                    f'{col} 的 valid 遮罩與主標籤不一致——min_pct_floor 設限應該讓整個'
                    f'網格共用同一組有效列，不一致代表 R3 規則沒生效')
                df_btc[col] = g_tgt
                grid_cols.append(col)

    n_before = len(df_btc)
    df_btc = df_btc[valid].reset_index(drop=True)
    n_censored = n_before - len(df_btc)
    print(f"丟棄尾端 {n_censored} 根無效標籤"
          f"（看不到未來結算價／連續段被右設限，對應洩漏路徑 L1/L2）")
    pos_rate_raw = df_btc['target'].mean()
    print(f"標籤分佈：訓練標籤正樣本比例 = {pos_rate_raw:.4f}，"
          f"合約正樣本比例 = {df_btc['contract_target'].mean():.4f}")
    if args.label_mode == 'run_center':
        n_pos_raw = int(df_btc['target'].sum())
        hit = df_btc.loc[df_btc['target'] == 1, 'fwd_ret']
        assert (df_btc.loc[df_btc['target'] == 1, 'contract_target'] == 1).all(), \
            'target=1 必須 100% 蘊含 contract_target=1（漲幅達門檻必然是上漲）'
        print(f"  訓練正樣本 {n_pos_raw} 個，其 1 小時報酬 "
              f"mean={hit.mean()*100:+.3f}% median={hit.median()*100:+.3f}%")

    # --- 存原始（未縮放）OHLCV + target，供評估/回測/對齊使用 ---
    raw_path = os.path.join(args.out_dir, 'raw_ohlcv.csv')
    df_btc.to_csv(raw_path, index=False)
    print(f"已存 {raw_path}（{len(df_btc)} 列）")

    # --- 特徵工程（因果性已由 toolbywi/test_causality.py 驗證，無未來洩漏） ---
    print("=== 計算技術指標 ===")
    df_feat = features(df_btc)
    print(f"features() 後: {len(df_feat)} 列（warmup 頭部已丟棄）")

    print("=== 計算微結構特徵（Round 4，6 個，由 Open/High/Low/Quote_asset_volume 衍生）===")
    df_feat = microstructure_features(df_feat)
    print(f"microstructure_features() 後: {len(df_feat)} 列（逐根轉換，無新 warmup）")

    print("=== 合併輔幣特徵 ===")
    df_feat = alt_features(df_feat, alt_dfs)
    print(f"alt_features() 後: {len(df_feat)} 列")

    if args.with_extra_sources:
        print("=== 下載 Round 3 新特徵來源（metrics / funding / premium / mark）===")
        # load_metrics() 跟 binance_load_5min 一樣是「日包逐日下載」慣例：
        # end_month 只涵蓋到當月第 1 天，要 +1 個月才能抓到目標月份的完整資料。
        metrics_end_y, metrics_end_m = args.end_year, args.end_month + 1
        if metrics_end_m > 12:
            metrics_end_y, metrics_end_m = metrics_end_y + 1, 1

        metrics_df = load_metrics(args.symbol, args.start_year, args.start_month,
                                   metrics_end_y, metrics_end_m)
        funding_df = load_funding_rate(args.symbol, args.start_year, args.start_month,
                                        args.end_year, args.end_month)
        premium_df = load_premium_index(args.symbol, args.start_year, args.start_month,
                                         args.end_year, args.end_month)
        mark_df = load_mark_price(args.symbol, args.start_year, args.start_month,
                                   args.end_year, args.end_month)
        for name, d in [('metrics', metrics_df), ('funding', funding_df),
                         ('premium', premium_df), ('mark', mark_df)]:
            if d is None or len(d) == 0:
                raise RuntimeError(f"{name} 沒有抓到任何資料，中止。")

        print("=== 合併新特徵來源 ===")
        df_feat = extra_features(df_feat, metrics_df, funding_df, premium_df, mark_df)
        print(f"extra_features() 後: {len(df_feat)} 列")

    # --- 欄位整理：先丟掉已被指標吸收的原始欄，再 fit scaler，
    #     確保 scaler.feature_cols 與 meta.json 的 feature_cols 是同一份清單
    #     （否則 apply_scaler() 在即時推論時會要求 df 含有已經不存在的欄位）---
    drop_cols = ['Open', 'High', 'Low', 'Quote_asset_volume', 'Taker_buy_quote_asset_volume']
    df_feat = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns])

    # --- 把標籤輔助欄取出來，絕對不能留到 zscore_scaler()。
    #     scaler() 與 Dataset_Custom 都是「非 date/非 target 的每一欄都算特徵」，
    #     contract_target/fwd_ret/網格欄留在裡面會被當成模型輸入、不遮蔽、
    #     且每一根都含 Close[t+lookahead]——完全洩漏而且不會報錯。---
    LABEL_AUX_COLS = ['contract_target', 'fwd_ret'] + grid_cols
    aux_present = [c for c in LABEL_AUX_COLS if c in df_feat.columns]
    df_aux = df_feat[['date'] + aux_present].copy().reset_index(drop=True)
    df_feat = df_feat.drop(columns=aux_present)

    # --- Z-Score：只 fit 訓練段。train_ratio 必須與 Dataset_Custom 的切分公式一致 ---
    print("=== Z-Score 縮放 ===")
    df_scaled, sc, feature_cols = zscore_scaler(
        df_feat, train_ratio=args.train_ratio, return_scaler=True)

    other_cols = [c for c in df_scaled.columns if c not in ('date', 'target')]
    assert other_cols == feature_cols, "欄序與 scaler.feature_cols 不一致，不應發生"
    df_final = df_scaled[['date'] + other_cols + ['target']]

    # --- 結構性斷言：擋住「多一欄靜默變成第 N+1 個特徵」的洩漏路徑 ---
    for c in LABEL_AUX_COLS:
        assert c not in feature_cols, f"{c} 混進了 feature_cols，這會造成未來資料洩漏！"
        assert c not in df_final.columns, f"{c} 混進了 stock_features.csv，這會造成未來資料洩漏！"
    assert set(df_final.columns) == {'date', 'target'} | set(feature_cols), \
        "stock_features.csv 的欄位集合必須嚴格等於 {date, target} ∪ feature_cols"
    assert len(df_final.columns) == len(feature_cols) + 2, "欄數必須是 n_features + 2"
    assert len(df_aux) == len(df_final), "labels.csv 必須與 stock_features.csv 等長"
    assert (df_aux['date'].values == df_final['date'].values).all(), \
        "labels.csv 與 stock_features.csv 的 date 必須逐值相等（同序）"

    print(f"最終特徵表：{len(df_final)} 列 x {len(feature_cols)} 個特徵 + date + target")

    features_path = os.path.join(args.out_dir, 'stock_features.csv')
    df_final.to_csv(features_path, index=False)
    print(f"已存 {features_path}")

    scaler_path = os.path.join(args.out_dir, 'scaler.joblib')
    save_scaler(scaler_path, sc, feature_cols)

    # --- labels.csv：與 stock_features.csv 等長同序，含合約結果與參數網格標籤。
    #     刻意獨立成一個檔案而不是塞進特徵表，也不是事後用 raw_ohlcv.csv merge：
    #     這兩欄是隨著 features()/alt_features()/extra_features() 的每一次
    #     join 與 dropna 一起被篩掉的，對齊由「建構方式」保證，不靠事後對齊。---
    df_aux = df_aux.copy()
    df_aux.insert(1, 'target', df_final['target'].values)
    labels_path = os.path.join(args.out_dir, 'labels.csv')
    df_aux.to_csv(labels_path, index=False)
    print(f"已存 {labels_path}（{len(df_aux)} 列 x {len(df_aux.columns)} 欄，"
          f"含 contract_target/fwd_ret{' + ' + str(len(grid_cols)) + ' 個網格欄' if grid_cols else ''}）")

    # --- 類別比例 -> 建議 focal_alpha（沿用 experiment_runner.py 既有慣例）---
    n_pos = int(df_final['target'].sum())
    n_neg = len(df_final) - n_pos
    focal_alpha = n_neg / (n_pos + n_neg)

    # 與 data_provider/data_loader.py Dataset_Custom 完全相同的公式，
    # 供驗證步驟核對切分是否一致。
    num_train = int(len(df_final) * args.train_ratio)
    num_test = int(len(df_final) * 0.2)
    num_vali = len(df_final) - num_train - num_test

    meta = {
        'symbol': args.symbol,
        'alt_symbols': alt_symbols,
        'with_extra_sources': args.with_extra_sources,
        'lookahead': args.lookahead,
        'seq_len': args.seq_len,
        'label_len': 48,
        'pred_len': 1,
        'train_ratio': args.train_ratio,
        'n_rows': len(df_final),
        'num_train': num_train,
        'num_vali': num_vali,
        'num_test': num_test,
        'n_pos': n_pos,
        'n_neg': n_neg,
        'pos_rate': n_pos / len(df_final),
        'suggested_focal_alpha': round(focal_alpha, 4),
        'feature_cols': feature_cols,
        'n_features': len(feature_cols),
        'data_path': 'stock_features.csv',
        'raw_ohlcv_path': 'raw_ohlcv.csv',
        'scaler_path': 'scaler.joblib',
        'date_range': {
            'start': str(df_final['date'].min()),
            'end': str(df_final['date'].max()),
        },
        'microstructure_features': ['HL_Range', 'OC_Ret', 'Up_Wick', 'Low_Wick',
                                     'BarVWAP_Dev', 'ATR_Pct_14'],
        # --- Round 5 標籤設定 ---
        'label_mode': args.label_mode,
        'min_pct': args.min_pct if args.label_mode == 'run_center' else None,
        'min_run': args.min_run if args.label_mode == 'run_center' else None,
        'min_pct_floor': args.min_pct_floor if args.label_mode == 'run_center' else None,
        'labels_path': 'labels.csv',
        'label_grid_cols': grid_cols,
        'contract_pos_rate': float(df_aux['contract_target'].mean()),
        'censored_tail_bars': int(n_censored),
        'label_note': (
            'target 是訓練標籤，contract_target 才是真實合約結算（漲就贏）。'
            '勝率/權益/通關一律用 contract_target，用 target 算勝率會得到與'
            '損益兩平完全不可比的數字。'),
    }
    meta_path = os.path.join(args.out_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"已存 {meta_path}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
