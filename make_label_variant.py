#!/usr/bin/env python3
"""
從 labels.csv 抽出某一個網格標籤欄，換掉 stock_features.csv 的 target，
產生一個可直接訓練的資料目錄（Round 5 階段 2 用）。

不重新下載、不重算特徵、不重 fit scaler——25 組網格共用同一份特徵表與
同一組切分（由 prepare_event_data.py 的 min_pct_floor 設限保證），所以
換標籤只是換一欄。這也確保階段 1（GBDT 篩選）與階段 2（transformer）
訓練在逐位元相同的標籤上。

用法：
  .venv/bin/python make_label_variant.py --source_dir ./dataset/event1h_v5_sel \
      --label_col y_bp30_r5 --out_dir ./dataset/event1h_v5_bp30_r5
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source_dir', required=True)
    p.add_argument('--label_col', required=True, help='labels.csv 裡的欄名，例如 y_bp30_r5')
    p.add_argument('--out_dir', required=True)
    return p.parse_args()


def parse_label_col(col):
    """y_bp30_r5 -> (0.003, 5)"""
    bp = int(col.split('_bp')[1].split('_')[0])
    run = int(col.split('_r')[-1])
    return bp / 10000.0, run


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    meta = json.load(open(os.path.join(args.source_dir, 'meta.json')))
    feat = pd.read_csv(os.path.join(args.source_dir, meta['data_path']))
    labels = pd.read_csv(os.path.join(args.source_dir, meta.get('labels_path', 'labels.csv')))

    assert len(feat) == len(labels), 'labels.csv 與 stock_features.csv 必須等長'
    assert (feat['date'].values == labels['date'].values).all(), 'date 必須逐值相等'
    assert args.label_col in labels.columns, (
        f"labels.csv 沒有欄位 {args.label_col}，可用的有：{[c for c in labels.columns if c.startswith('y_')]}")

    feature_cols = meta['feature_cols']
    new_target = labels[args.label_col].values
    feat_new = feat.copy()
    feat_new['target'] = new_target
    feat_new = feat_new[['date'] + feature_cols + ['target']]

    # 結構性斷言：換標籤不能改變欄位結構（擋洩漏）
    assert set(feat_new.columns) == {'date', 'target'} | set(feature_cols)
    assert len(feat_new.columns) == len(feature_cols) + 2

    feat_new.to_csv(os.path.join(args.out_dir, 'stock_features.csv'), index=False)
    for fn in ('labels.csv', 'raw_ohlcv.csv', 'scaler.joblib'):
        src = os.path.join(args.source_dir, fn)
        if os.path.exists(src):
            shutil.copyfile(src, os.path.join(args.out_dir, fn))

    min_pct, min_run = parse_label_col(args.label_col)
    n_pos = int(new_target.sum())
    n_rows = len(feat_new)

    new_meta = dict(meta)
    new_meta.update({
        'min_pct': min_pct, 'min_run': min_run,
        'label_variant_col': args.label_col,
        'label_variant_source': args.source_dir,
        'n_pos': n_pos, 'n_neg': n_rows - n_pos,
        'pos_rate': n_pos / n_rows,
        'suggested_focal_alpha': round((n_rows - n_pos) / n_rows, 4),
    })
    # 上一份資料的訓練紀錄不適用於新標籤，清掉（同 select_features.py 的做法）
    for k in ('setting', 'train_args', 'train_runs'):
        new_meta.pop(k, None)

    with open(os.path.join(args.out_dir, 'meta.json'), 'w') as f:
        json.dump(new_meta, f, indent=2, ensure_ascii=False)

    print(f"=== {args.label_col} → {args.out_dir} ===")
    print(f"  min_pct={min_pct*100:.2f}%  min_run={min_run}")
    print(f"  正樣本 {n_pos} / {n_rows} = {n_pos/n_rows*100:.3f}%")
    print(f"  suggested_focal_alpha = {new_meta['suggested_focal_alpha']}")
    print(f"  合約正樣本率 {labels['contract_target'].mean()*100:.3f}%（不隨標籤改變）")


if __name__ == '__main__':
    main()
