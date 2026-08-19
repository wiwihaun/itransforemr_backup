#!/usr/bin/env python3
"""
特徵篩選 — 依 gbdt_probe.py 算出的 1sigma 規則（importance_mean - importance_std > 0），
把既有已縮放的 stock_features.csv 做欄位子集，輸出到新資料目錄。

不重新下載、不重新算技術指標——因為每欄 Z-score 互相獨立，直接對已縮放的
CSV 做欄位子集 + 對既有 fitted StandardScaler 的 mean_/scale_/var_ 陣列取子集，
數學上等價於「只對這些欄位重新 fit 一次 scaler」，但快得多。

前置：必須先跑過 gbdt_probe.py（產生 {source_dir}/gbdt_probe_report.json，
裡面已經只用 train+val 算過 selected_features，測試期沒被碰過）。

用法：
  .venv/bin/python select_features.py --source_dir ./dataset/event30m --out_dir ./dataset/event30m_v2
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toolbywi'))
from feature_scale import load_scaler, save_scaler


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source_dir', default='./dataset/event30m')
    p.add_argument('--out_dir', default='./dataset/event30m_v2')
    return p.parse_args()


def subset_scaler(sc, feature_cols, selected_features):
    """
    對已 fit 的 StandardScaler 取欄位子集，不重新 fit。
    每欄 Z-score 是獨立的仿射變換，取子集在數學上等價於只對這些欄位重新
    fit 一次——但這裡直接用現成的 mean_/scale_/var_，不用重新掃資料。
    順便用目前安裝的 sklearn 版本重新建構物件，解決舊 scaler.joblib
    跨版本 InconsistentVersionWarning 的問題（新檔案不會再有這個警告）。
    """
    idx = [feature_cols.index(f) for f in selected_features]

    new_sc = StandardScaler(copy=sc.copy, with_mean=sc.with_mean, with_std=sc.with_std)
    new_sc.mean_ = np.asarray(sc.mean_)[idx].copy()
    new_sc.scale_ = np.asarray(sc.scale_)[idx].copy()
    new_sc.var_ = np.asarray(sc.var_)[idx].copy()
    new_sc.n_features_in_ = len(selected_features)
    new_sc.feature_names_in_ = np.array(selected_features, dtype=object)
    new_sc.n_samples_seen_ = sc.n_samples_seen_
    return new_sc


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    report_path = os.path.join(args.source_dir, 'gbdt_probe_report.json')
    if not os.path.exists(report_path):
        raise FileNotFoundError(
            f"{report_path} 不存在。請先跑 gbdt_probe.py --data_dir {args.source_dir}"
            f"（會用 train+val 算出 selected_features，測試期不會被碰）。"
        )
    gbdt_report = json.load(open(report_path))
    selected_features = gbdt_report['selected_features']
    if not selected_features:
        raise RuntimeError("gbdt_probe_report.json 裡 selected_features 是空的，無法繼續。")

    print(f"=== 特徵篩選：{args.source_dir} → {args.out_dir} ===")
    print(f"篩選規則：{gbdt_report['selection_rule']}")
    print(f"保留 {len(selected_features)} 個特徵：{selected_features}\n")

    meta = json.load(open(os.path.join(args.source_dir, 'meta.json')))
    df = pd.read_csv(os.path.join(args.source_dir, meta['data_path']))

    missing = [f for f in selected_features if f not in df.columns]
    if missing:
        raise RuntimeError(f"stock_features.csv 缺少被選中的欄位 {missing}，來源資料是否已變動？")

    df_new = df[['date'] + selected_features + ['target']].copy()
    features_path = os.path.join(args.out_dir, 'stock_features.csv')
    df_new.to_csv(features_path, index=False)
    print(f"已存 {features_path}（{len(df_new)} 列 x {len(selected_features)} 個特徵）")

    raw_src = os.path.join(args.source_dir, meta['raw_ohlcv_path'])
    raw_dst = os.path.join(args.out_dir, 'raw_ohlcv.csv')
    shutil.copyfile(raw_src, raw_dst)
    print(f"已複製 {raw_dst}")

    sc, feature_cols = load_scaler(os.path.join(args.source_dir, meta['scaler_path']))
    new_sc = subset_scaler(sc, feature_cols, selected_features)
    scaler_path = os.path.join(args.out_dir, 'scaler.joblib')
    save_scaler(scaler_path, new_sc, selected_features)

    new_meta = dict(meta)
    new_meta['feature_cols'] = selected_features
    new_meta['n_features'] = len(selected_features)
    new_meta['data_path'] = 'stock_features.csv'
    new_meta['raw_ohlcv_path'] = 'raw_ohlcv.csv'
    new_meta['scaler_path'] = 'scaler.joblib'
    new_meta['feature_selection'] = {
        'source_dir': args.source_dir,
        'rule': gbdt_report['selection_rule'],
        'n_features_before': gbdt_report['n_features'],
        'n_features_after': len(selected_features),
        'note': '重要度只用 train+val 計算（gbdt_probe.py），測試期未被讀取。',
    }
    # setting/train_args 是上一份資料的訓練紀錄，欄位數已變，不適用於新資料，清掉
    new_meta.pop('setting', None)
    new_meta.pop('train_args', None)

    meta_path = os.path.join(args.out_dir, 'meta.json')
    with open(meta_path, 'w') as f:
        json.dump(new_meta, f, indent=2, ensure_ascii=False)
    print(f"已存 {meta_path}")


if __name__ == '__main__':
    main()
