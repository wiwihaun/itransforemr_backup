#!/usr/bin/env python3
"""
Round 5 標籤參數網格搜尋（階段 1：GBDT 篩選）。

在 min_pct x min_run 的網格上，找出「哪一種訓練標籤定義，能產生對真實
合約結果排序最好的分數」。25 組共用同一份特徵表、同一組切分、同一份
scaler（由 prepare_event_data.py 的 min_pct_floor 設限保證），所以比較
是公平的。

三個關鍵設計：

1. **一律對照 contract_target 評分，不是對照訓練標籤。** 合約結果不隨
   訓練標籤改變，所以 25 個數字在同一把尺上；用各自的訓練標籤評分等於
   比較 25 個不同的問題，沒有意義。

2. **落後堆疊補上下文。** GBDT 只看單根，而 transformer 看 512 根窗口；
   取中間標籤本質是「這波漲勢會不會持續」，正是單根看不到的。加 lag
   {1,3,6,12,24,48} 與 rolling mean/std {12,48,144}（全部後看）讓 GBDT
   也有約 12 小時上下文，篩選結果才站得住腳。即使如此，這仍是 screen
   不是證明——階段 2 會用 transformer 重新決定。

3. **測試期全程不讀。** 只取前 num_train+num_vali 列（同 gbdt_probe.py）。

事前註冊的選擇規則（跑之前就寫死，見 --help）：
  - 平手判準：OOF 平均 pAUC 在最佳者 1 個標準誤內視為打平，取正樣本率
    最高者（較小 min_pct、較小 min_run）。偏向穩定，絕不偏向指標。
  - 空手門檻：最佳組的 block bootstrap CI 下界 <= 0.50 就不推任何組進
    階段 2，直接回報「沒有標籤變體能證明贏過無技巧」。
  - 功效不足：訓練段正樣本 < min_train_pos 標記 underpowered。

用法：
  .venv/bin/python label_grid_search.py --data_dir ./dataset/event1h_v5_sel
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toolbywi'))
from event_common import (mean_partial_auc, pauc_report, block_bootstrap_ci,
                           wilson_ci, cluster_entries, PAUC_MAX_FPRS)

LAGS = (1, 3, 6, 12, 24, 48)
ROLLS = (12, 48, 144)


def build_lag_stack(df, feature_cols):
    """
    落後堆疊：全部後看（shift 正值 / rolling 預設右對齊），無任何未來資訊。
    回傳新的 DataFrame（含原始欄 + lag + rolling mean/std）。
    """
    out = {c: df[c].values for c in feature_cols}
    for c in feature_cols:
        s = df[c]
        for L in LAGS:
            out[f'{c}_lag{L}'] = s.shift(L).values
        for W in ROLLS:
            out[f'{c}_rm{W}'] = s.rolling(W).mean().values
            out[f'{c}_rs{W}'] = s.rolling(W).std().values
    return pd.DataFrame(out, index=df.index)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default='./dataset/event1h_v5_sel')
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--top_k', type=int, default=3, help='推進階段 2 的組數')
    p.add_argument('--reference_label', default='y_bp30_r5',
                    help='事前註冊的對照組（使用者原始指定的 0.3%%/min_run=5）。'
                         '即使沒進前 top_k 也一定會被推進階段 2——沒有它就無法區分'
                         '「搜尋真的找到東西」與「搜尋只是擬合驗證集雜訊」')
    p.add_argument('--min_train_pos', type=int, default=500,
                    help='訓練段正樣本低於此值標記 underpowered')
    p.add_argument('--block_len', type=int, default=None,
                    help='block bootstrap 區塊長度，不給則用 meta[lookahead]')
    p.add_argument('--n_boot', type=int, default=1000)
    p.add_argument('--row_stride', type=int, default=1,
                    help='列取樣間隔，>1 可加速（標籤重疊使相鄰列資訊高度重複）')
    return p.parse_args()


def main():
    args = parse_args()
    meta = json.load(open(os.path.join(args.data_dir, 'meta.json')))
    block_len = args.block_len if args.block_len is not None else meta['lookahead']
    feature_cols = meta['feature_cols']

    feat = pd.read_csv(os.path.join(args.data_dir, meta['data_path']))
    labels = pd.read_csv(os.path.join(args.data_dir, meta.get('labels_path', 'labels.csv')))
    assert len(feat) == len(labels) == meta['n_rows'], '特徵表與 labels.csv 長度必須一致'
    assert (feat['date'].values == labels['date'].values).all(), 'date 必須逐值相等'

    grid_cols = [c for c in meta['label_grid_cols'] if c in labels.columns]
    assert grid_cols, 'labels.csv 裡找不到任何網格標籤欄'

    n_trainval = meta['num_train'] + meta['num_vali']
    print(f"=== 標籤網格搜尋：{len(grid_cols)} 組 x train+val {n_trainval} 列"
          f"（測試期 {meta['num_test']} 列全程不讀）===")
    print(f"主指標: OOF 平均 pAUC vs contract_target（PAUC_MAX_FPRS={PAUC_MAX_FPRS}）")
    print(f"block_len={block_len}  n_splits={args.n_splits}\n")

    print("建立落後堆疊（lag + rolling，全部後看）...")
    X_df = build_lag_stack(feat[feature_cols], feature_cols)
    warmup = max(max(LAGS), max(ROLLS))
    lo, hi = warmup, n_trainval
    X = X_df.iloc[lo:hi].values
    contract = labels['contract_target'].values[lo:hi]
    if args.row_stride > 1:
        X, contract = X[::args.row_stride], contract[::args.row_stride]
    print(f"堆疊後 {X.shape[1]} 個輸入欄，可用 {len(X)} 列"
          f"（前 {warmup} 列 warmup 丟棄）\n")

    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    splits = list(tscv.split(X))

    print(f"{'label':>14} {'pos_rate':>9} {'train_pos':>10} {'OOF_pAUC':>9} "
          f"{'CI_lo':>8} {'CI_hi':>8} {'OOF_AUC':>8} {'flag':>12}")
    results = []
    for col in grid_cols:
        y_full = labels[col].values[lo:hi]
        y = y_full[::args.row_stride] if args.row_stride > 1 else y_full
        n_train_pos = int(y[:int(len(y) * 0.78)].sum())  # 約略的訓練段正樣本數

        oof_true, oof_score = [], []
        for tr, va in splits:
            if y[tr].sum() == 0 or y[va].sum() == 0:
                continue
            clf = HistGradientBoostingClassifier(
                max_depth=6, random_state=42, class_weight='balanced',
                early_stopping=False)
            clf.fit(X[tr], y[tr])
            oof_score.append(clf.predict_proba(X[va])[:, 1])
            oof_true.append(contract[va])       # ← 對照真實合約，不是訓練標籤
        if not oof_true:
            print(f"{col:>14} {'--':>9} {n_train_pos:>10} {'跳過（某折無正樣本）':>30}")
            continue

        ot = np.concatenate(oof_true)
        os_ = np.concatenate(oof_score)
        pr = pauc_report(ot, os_)
        ci_lo, ci_hi = block_bootstrap_ci(ot, os_, mean_partial_auc,
                                           block_len=block_len, n_boot=args.n_boot)
        under = n_train_pos < args.min_train_pos
        results.append({
            'label': col, 'pos_rate': float(y.mean()), 'n_train_pos': n_train_pos,
            'oof_mean_pauc': pr['mean_pauc'], 'oof_auc': pr['auc'],
            'ci_lo': ci_lo, 'ci_hi': ci_hi, 'underpowered': bool(under),
            'pauc_by_max_fpr': {k: v for k, v in pr.items()
                                 if k not in ('auc', 'mean_pauc')},
            'n_unique_score': int(len(np.unique(np.round(os_, 6)))),
            'score_std': float(os_.std()),
        })
        print(f"{col:>14} {y.mean()*100:>8.3f}% {n_train_pos:>10} {pr['mean_pauc']:>9.4f} "
              f"{ci_lo:>8.4f} {ci_hi:>8.4f} {pr['auc']:>8.4f} "
              f"{'UNDERPOWERED' if under else '':>12}")

    assert results, '沒有任何組合成功評估'

    # ── 事前註冊的選擇規則 ──────────────────────────────────
    best = max(results, key=lambda r: r['oof_mean_pauc'])
    print(f"\n=== 事前註冊的選擇規則 ===")

    # 空手門檻
    if best['ci_lo'] <= 0.50:
        print(f"❌ 空手門檻觸發：最佳組 {best['label']} 的 95% CI 下界 "
              f"{best['ci_lo']:.4f} <= 0.50")
        print("   沒有任何標籤變體能證明贏過無技巧，依事前註冊規則不推進階段 2。")
        print("   這是合法的 Round 5 結果，應據實寫進報告。")
        promoted = []
    else:
        # 1-SE 平手規則：CI 半寬約當 1 個標準誤（bootstrap 分佈近似常態）
        se = (best['ci_hi'] - best['ci_lo']) / (2 * 1.959963984540054)
        thresh = best['oof_mean_pauc'] - se
        tied = [r for r in results
                if r['oof_mean_pauc'] >= thresh and not r['underpowered']]
        # 平手時取正樣本率最高者（偏向穩定：正樣本多 -> 梯度變異小、標籤單純）
        tied.sort(key=lambda r: -r['pos_rate'])
        print(f"✅ 最佳 {best['label']} pAUC={best['oof_mean_pauc']:.4f} "
              f"CI=[{best['ci_lo']:.4f}, {best['ci_hi']:.4f}]（1 SE ≈ {se:.4f}）")
        print(f"   1-SE 內打平的有 {len(tied)} 組，依「正樣本率最高」排序後取前 {args.top_k}")
        promoted = [r['label'] for r in tied[:args.top_k]]

    if args.reference_label and args.reference_label not in promoted:
        promoted.append(args.reference_label)
        print(f"   另加入事前註冊對照組 {args.reference_label}"
              f"（即使沒進前 {args.top_k}，用來區分『真的找到東西』與『擬合雜訊』）")
    print(f"\n推進階段 2 的組合：{promoted}")

    out = {
        'metric': 'oof_mean_pauc_vs_contract_target',
        'block_len': int(block_len), 'n_splits': args.n_splits,
        'row_stride': args.row_stride, 'lag_stack': {'lags': list(LAGS), 'rolls': list(ROLLS)},
        'n_rows_used': int(len(X)), 'n_input_cols': int(X.shape[1]),
        'results': sorted(results, key=lambda r: -r['oof_mean_pauc']),
        'best_label': best['label'],
        'null_gate_tripped': bool(best['ci_lo'] <= 0.50),
        'promoted': promoted,
        'reference_label': args.reference_label,
        'note': ('這 25 組在同一個驗證面上比較過，該比較本身用掉了驗證集資訊，'
                 '贏家的驗證數字是上界，不能當預期表現。測試期全程未讀。'
                 'GBDT 是 screen 不是證明——取中間標籤本質是「漲勢會不會持續」，'
                 '單根看不到，落後堆疊只補到約 12 小時上下文，不等於 512 根窗口。'),
    }
    out_path = os.path.join(args.data_dir, 'label_grid_report.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"已存 {out_path}")


if __name__ == '__main__':
    main()
