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

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toolbywi'))
from download_binance_monthly_batch import binance_load_5min_fast
from binance_extra import load_metrics, load_funding_rate, load_premium_index, load_mark_price
from target_calulate import target_direction
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
    p.add_argument('--lookahead', type=int, default=6, help='30min / 5min = 6 根')
    p.add_argument('--seq_len', type=int, default=1024)
    p.add_argument('--train_ratio', type=float, default=0.7)
    p.add_argument('--out_dir', default='./dataset/event30m')
    p.add_argument('--with_extra_sources', action=argparse.BooleanOptionalAction, default=True,
                    help='是否加入 Round 3 新特徵來源（metrics/funding/premium/mark），預設開')
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

    # --- 標籤：在原始資料上算，Close 沒有指標 warmup 的 NaN，對齊最乾淨 ---
    print(f"=== 建立 {args.lookahead} 根（{args.lookahead * 5} 分鐘）滾動方向標籤 ===")
    target, valid = target_direction(df_btc, lookahead=args.lookahead)
    df_btc = df_btc.copy()
    df_btc['target'] = target
    n_before = len(df_btc)
    df_btc = df_btc[valid].reset_index(drop=True)
    print(f"丟棄尾端 {n_before - len(df_btc)} 根無效標籤（看不到未來結算價，對應洩漏路徑 L1）")
    pos_rate_raw = df_btc['target'].mean()
    print(f"標籤分佈：正樣本比例 = {pos_rate_raw:.4f}")

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

    # --- Z-Score：只 fit 訓練段。train_ratio 必須與 Dataset_Custom 的切分公式一致 ---
    print("=== Z-Score 縮放 ===")
    df_scaled, sc, feature_cols = zscore_scaler(
        df_feat, train_ratio=args.train_ratio, return_scaler=True)

    other_cols = [c for c in df_scaled.columns if c not in ('date', 'target')]
    assert other_cols == feature_cols, "欄序與 scaler.feature_cols 不一致，不應發生"
    df_final = df_scaled[['date'] + other_cols + ['target']]

    print(f"最終特徵表：{len(df_final)} 列 x {len(feature_cols)} 個特徵 + date + target")

    features_path = os.path.join(args.out_dir, 'stock_features.csv')
    df_final.to_csv(features_path, index=False)
    print(f"已存 {features_path}")

    scaler_path = os.path.join(args.out_dir, 'scaler.joblib')
    save_scaler(scaler_path, sc, feature_cols)

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
    }
    meta_path = os.path.join(args.out_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"已存 {meta_path}")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
