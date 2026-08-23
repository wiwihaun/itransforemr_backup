#!/usr/bin/env python3
"""
GBDT 特徵探測 — 在訓練 iTransformer 之前的第一道關卡。

GBDT 只看單根 K 線的特徵向量（transformer 看得到整個 seq_len 視窗的注意力），
所以這裡的 AUC 是「下界」不是上界：若 GBDT 都測不到訊號，代表這組特徵
對 30 分鐘方向幾乎沒有資訊，此時該回頭換特徵，而不是指望 transformer 變魔術。

一律用 TimeSeriesSplit（絕不 shuffle），避免未來資料混進訓練折（洩漏路徑 L4）。

Round 2 修正（見 no-lookahead-bias skill）：Round 1 版本對全部資料（含測試期）
做 TimeSeriesSplit，最後一折的驗證段 100% 落在測試期內，等於拿測試集資料算
「哪個特徵重要」——這本身就是 L4 決策洩漏。這一版一律先把資料截到
train+val（`meta.json` 的 num_train+num_vali），測試期從頭到尾不會被讀到。
特徵重要度也從「只取最後一折」改成「跨所有折 x n_repeats 合併計算」，樣本數
更多、估計更穩定。

Round 4 修正：這套系統只交易 p_up 排進前 5~20% 的樣本，全域 AUC 把後面
80~95% 完全不會下單的樣本也算進去，是選模指標與實際決策的錯位（實測：
R3 冠軍 checkpoint 測試集全域 AUC 0.5081，但同一組分數的平均 partial AUC
只有 0.5004——幾乎正好是 no-skill）。預設指標改成 mean_pauc（見
toolbywi/event_common.py 的 PAUC_MAX_FPRS 四個操作點平均），--metric auc
可切回舊行為以便回溯比較。同時標籤在 lookahead 根之內高度重疊
（R3 實測 lag-1 一致率 0.8118），舊的 iid bootstrap_auc_ci 會低估信賴區間
寬度（實測約 1.58 倍），改用 block bootstrap（toolbywi/event_common.py 的
block_bootstrap_ci，block_len 預設 meta['lookahead']）。

用法：
  .venv/bin/python gbdt_probe.py --data_dir ./dataset/event30m
  .venv/bin/python gbdt_probe.py --data_dir ./dataset/event30m_v4 --metric mean_pauc \
      --baseline_mean_pauc 0.5050
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, accuracy_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toolbywi'))
from test_causality import test_features_causal, test_alt_features_causal, test_microstructure_features_causal
from event_common import (mean_partial_auc, pauc_report, make_mean_pauc_scorer,
                           block_bootstrap_ci, PAUC_MAX_FPRS)


def load_trainval(data_dir):
    """
    讀資料並截到 train+val，測試期完全不碰。
    回傳 (X, y, feature_cols, meta, n_trainval)。
    """
    meta = json.load(open(os.path.join(data_dir, 'meta.json')))
    df = pd.read_csv(os.path.join(data_dir, meta['data_path']))
    feature_cols = meta['feature_cols']

    n_trainval = meta['num_train'] + meta['num_vali']
    df_trainval = df.iloc[:n_trainval].reset_index(drop=True)

    X = df_trainval[feature_cols].values
    y = df_trainval['target'].values
    return X, y, feature_cols, meta, n_trainval


def bootstrap_auc_ci(y_true, y_score, n_boot=1000, seed=0):
    """舊版 iid bootstrap（保留供回溯比較，見 module docstring 的 Round 4 說明；
    標籤重疊時會低估 CI 寬度，新流程預設走 block_bootstrap_ci）。"""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    aucs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            aucs[b] = np.nan
            continue
        aucs[b] = roc_auc_score(yt, ys)
    aucs = aucs[~np.isnan(aucs)]
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def compute_feature_importance(X, y, feature_cols, n_splits=5, n_repeats=5, random_state=42,
                                scoring='roc_auc'):
    """
    跨所有折合併計算 permutation importance（而非只取最後一折）。
    每折在自己的驗證段上跑一次 permutation_importance（n_repeats 次重複），
    把所有折 x 重複的原始樣本池在一起再算 mean/std——樣本數是 n_splits*n_repeats，
    比單折的 n_repeats 更穩定。呼叫端必須保證傳進來的 X/y 已經排除測試期
    （見 load_trainval）。

    scoring：'roc_auc'（舊行為）或 event_common.make_mean_pauc_scorer() 回傳的
    scorer（Round 4 預設，見 main() 的 --metric）——這是特徵篩選會不會偏向
    「交易區間有排序力」而非「全域排序力」的關鍵注入點。
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    pooled = {f: [] for f in feature_cols}

    for train_idx, val_idx in tscv.split(X):
        clf = HistGradientBoostingClassifier(max_depth=6, random_state=random_state)
        clf.fit(X[train_idx], y[train_idx])
        result = permutation_importance(
            clf, X[val_idx], y[val_idx],
            n_repeats=n_repeats, random_state=random_state, scoring=scoring, n_jobs=-1)
        for i, f in enumerate(feature_cols):
            pooled[f].extend(result.importances[i].tolist())

    importances = []
    for f in feature_cols:
        vals = np.array(pooled[f])
        importances.append({
            'feature': f,
            'importance_mean': float(vals.mean()),
            'importance_std': float(vals.std(ddof=1)),
            'n_samples': int(len(vals)),
        })
    importances.sort(key=lambda d: -d['importance_mean'])
    return importances


def select_features(importances, min_features=15):
    """
    Round 3 改為 top-k 規則：n_pass = 1sigma 規則（importance_mean -
    importance_std > 0）通過數，k = max(min_features, n_pass)，依
    importance_mean 降序取前 k。保證至少 min_features 個特徵——Round 2 的
    純 1sigma 規則在 34 個候選裡只留 4 個，太激進；這裡改成「規則事前宣告、
    保證下限，且不排除任何 1sigma 存活者」。
    """
    n_pass = sum(1 for d in importances if d['importance_mean'] - d['importance_std'] > 0)
    k = max(min_features, n_pass)
    ranked = sorted(importances, key=lambda d: -d['importance_mean'])
    return [d['feature'] for d in ranked[:k]]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default='./dataset/event30m')
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--min_features', type=int, default=15,
                    help='top-k 篩選的下限（k = max(min_features, 1sigma 通過數)）')
    p.add_argument('--metric', default='mean_pauc', choices=['mean_pauc', 'auc'],
                    help='Round 4 預設 mean_pauc（見 toolbywi/event_common.py 的 PAUC_MAX_FPRS）——'
                         '只在系統實際會下單的區間評估排序品質；auc 是 Round 1-3 的舊行為，'
                         '供回溯比較用。決定 permutation importance 的 scoring、OOF 主指標與'
                         '信賴區間/通關判定都用哪個量。')
    p.add_argument('--block_len', type=int, default=None,
                    help='block bootstrap 的區塊長度，不給則用 meta[\'lookahead\']（標籤重疊的根數）')
    p.add_argument('--baseline_auc', type=float, default=None,
                    help='對照的既有 OOF AUC（例如 Round 2 的 0.5081），用來比較新特徵來源有沒有推動訊號。'
                         '只在 --metric auc 時有意義')
    p.add_argument('--baseline_mean_pauc', type=float, default=None,
                    help='對照的既有 OOF 平均 pAUC（例如 Round 3 冠軍 checkpoint 的驗證集 0.5050）。'
                         '只在 --metric mean_pauc 時有意義。pAUC 與 AUC 不同尺度，不可混用兩個 baseline')
    p.add_argument('--skip_causality', action='store_true',
                    help='跳過因果性測試（已跑過一次時可加速）')
    return p.parse_args()


def main():
    args = parse_args()

    if not args.skip_causality:
        print("=== 因果性測試（確認特徵沒有未來洩漏）===")
        test_features_causal()
        test_alt_features_causal()
        test_microstructure_features_causal()
        print()

    X, y, feature_cols, meta, n_trainval = load_trainval(args.data_dir)
    block_len = args.block_len if args.block_len is not None else meta['lookahead']

    print(f"=== GBDT 特徵探測：train+val {n_trainval} 列（測試期 {meta['num_test']} 列已排除）"
          f" x {len(feature_cols)} 個特徵 ===")
    print(f"標籤正樣本比例: {y.mean():.4f}")
    print(f"主指標: {args.metric}  block_len: {block_len}（block bootstrap 區塊長度，"
          f"防標籤重疊讓 CI 虛假精確）\n")

    tscv = TimeSeriesSplit(n_splits=args.n_splits)
    splits = list(tscv.split(X))

    oof_true, oof_score = [], []
    fold_reports = []

    for fold, (train_idx, val_idx) in enumerate(splits, start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = HistGradientBoostingClassifier(max_depth=6, random_state=42)
        clf.fit(X_train, y_train)
        proba = clf.predict_proba(X_val)[:, 1]

        auc = roc_auc_score(y_val, proba)
        mean_pauc = mean_partial_auc(y_val, proba)
        acc = accuracy_score(y_val, (proba >= 0.5).astype(int))
        majority_baseline = max(y_val.mean(), 1 - y_val.mean())

        fold_reports.append({
            'fold': fold, 'train_size': int(len(train_idx)), 'val_size': int(len(val_idx)),
            'auc': float(auc), 'mean_pauc': float(mean_pauc), 'accuracy': float(acc),
            'majority_baseline': float(majority_baseline),
        })
        print(f"[fold {fold}] train={len(train_idx):>6} val={len(val_idx):>6} "
              f"AUC={auc:.4f}  mean_pAUC={mean_pauc:.4f}  acc={acc:.4f}  多數類基準={majority_baseline:.4f}")

        oof_true.append(y_val)
        oof_score.append(proba)

    oof_true = np.concatenate(oof_true)
    oof_score = np.concatenate(oof_score)
    oof_auc = roc_auc_score(oof_true, oof_score)
    oof_pauc = pauc_report(oof_true, oof_score)
    oof_mean_pauc = oof_pauc['mean_pauc']

    # 主指標決定 CI 用哪個 metric_fn、通關判定看哪個信賴區間——
    # --metric mean_pauc（預設）才是這套系統實際交易區間的排序品質；
    # --metric auc 保留給回溯比較舊行為。
    if args.metric == 'mean_pauc':
        primary_point = oof_mean_pauc
        primary_metric_fn = mean_partial_auc
        primary_baseline = args.baseline_mean_pauc
    else:
        primary_point = oof_auc
        primary_metric_fn = roc_auc_score
        primary_baseline = args.baseline_auc

    ci_lo, ci_hi = block_bootstrap_ci(oof_true, oof_score, primary_metric_fn, block_len=block_len)
    # 舊版 iid CI 一併記錄，供報告對照展示標籤重疊造成的低估幅度
    iid_ci_lo, iid_ci_hi = block_bootstrap_ci(oof_true, oof_score, primary_metric_fn, block_len=1)
    # 向下相容：不管 --metric 選哪個，legacy_auc_iid_ci95 永遠是「AUC + iid bootstrap」，
    # 逐位元等同 Round 1-3 的 bootstrap_auc_ci()（已用 R3 快取資料驗證 block_len=1
    # 與舊函式數值完全相同）——這樣重跑舊資料集時，report 裡引用的 CI 數字才對得上，
    # 不會因為這次改了預設 metric/CI 方法而悄悄變成別的數字。
    legacy_auc_ci_lo, legacy_auc_ci_hi = block_bootstrap_ci(oof_true, oof_score, roc_auc_score, block_len=1)

    print(f"\n=== OOF 整體結果（train+val 內 {len(oof_true)} 筆，1000 次 block bootstrap，block_len={block_len}）===")
    print(f"OOF AUC = {oof_auc:.4f}   OOF 平均 pAUC = {oof_mean_pauc:.4f}  "
          f"(pAUC@0.05/0.10/0.15/0.20 = {oof_pauc['pauc_0.05']:.4f}/{oof_pauc['pauc_0.10']:.4f}/"
          f"{oof_pauc['pauc_0.15']:.4f}/{oof_pauc['pauc_0.20']:.4f})")
    print(f"主指標（{args.metric}）= {primary_point:.4f}  95% CI（block） = [{ci_lo:.4f}, {ci_hi:.4f}]  "
          f"（iid CI 對照 = [{iid_ci_lo:.4f}, {iid_ci_hi:.4f}]，寬度比 block/iid = "
          f"{(ci_hi-ci_lo)/(iid_ci_hi-iid_ci_lo+1e-12):.2f}）")

    passes = bool(ci_lo > 0.5)
    if passes:
        print("✅ 信賴區間下界 > 0.50：特徵組確實帶有方向性訊號（這是下界，transformer 應更好）")
    else:
        print("⚠️ 信賴區間涵蓋 0.50：這組特徵在主指標衡量的區間幾乎沒有可偵測訊號，"
              "訓練 transformer 前應該先檢討特徵，而不是指望模型變魔術。")

    if primary_baseline is not None:
        delta = primary_point - primary_baseline
        print(f"\n=== 對照既有基準（{args.metric}）= {primary_baseline:.4f} ===")
        print(f"這一輪 {args.metric} = {primary_point:.4f}（差 {delta:+.4f}）")
        if ci_lo <= primary_baseline <= ci_hi:
            print("⚠️ 基準值落在這一輪的 95% CI 內，統計上無法區分這輪改動有沒有推動訊號——"
                  "這本身是重要結論，應該據實回報，不要因為跑得動就假裝有進展。")
        else:
            print("這一輪的 CI 不含基準值，訊號強度確實有可偵測的變化。")

    scorer = make_mean_pauc_scorer() if args.metric == 'mean_pauc' else 'roc_auc'
    print(f"\n=== 特徵重要度（跨 {args.n_splits} 折合併，每折 permutation importance n_repeats=5，"
          f"scoring={args.metric}，全部只在 train+val 內計算）===")
    importances = compute_feature_importance(X, y, feature_cols, n_splits=args.n_splits, scoring=scorer)

    selected = select_features(importances, min_features=args.min_features)
    n_pass_1sigma = sum(1 for d in importances if d['importance_mean'] - d['importance_std'] > 0)

    for d in importances:
        mark = '✓' if d['feature'] in selected else ' '
        bar = '#' * max(0, int(d['importance_mean'] * 500))
        print(f"  [{mark}] {d['feature']:<18} {d['importance_mean']:+.5f} +/- {d['importance_std']:.5f}  {bar}")

    print(f"\n篩選規則：k = max({args.min_features}, 1sigma 通過數={n_pass_1sigma}) "
          f"→ 保留 {len(selected)}/{len(feature_cols)} 個特徵（依重要度排序取前 k）")
    print(f"保留：{selected}")

    summary = {
        'n_rows_trainval': int(n_trainval), 'n_features': len(feature_cols), 'pos_rate': float(y.mean()),
        'fold_reports': fold_reports,
        'oof_auc': float(oof_auc),
        'oof_mean_pauc': float(oof_mean_pauc),
        'oof_pauc_by_max_fpr': {k: v for k, v in oof_pauc.items() if k not in ('auc', 'mean_pauc')},
        'metric': args.metric,
        'block_len': int(block_len),
        'primary_point': float(primary_point),
        'primary_ci95_block': [ci_lo, ci_hi],
        'primary_ci95_iid': [iid_ci_lo, iid_ci_hi],
        # 向下相容欄位：Round 1-3 的 compare_checkpoints.py / 報告腳本讀這兩個舊 key
        'oof_auc_ci95': [legacy_auc_ci_lo, legacy_auc_ci_hi],
        'baseline_auc': args.baseline_auc,
        'baseline_mean_pauc': args.baseline_mean_pauc,
        'passes_signal_check': passes,
        'feature_importance': importances,
        'selected_features': selected,
        'n_pass_1sigma': n_pass_1sigma,
        'min_features': args.min_features,
        'selection_rule': f'k = max({args.min_features}, 1sigma 通過數)，依 importance_mean 排序取前 k',
        'note': ('全部計算只用 train+val（前 num_train+num_vali 列），測試期未被讀取。'
                 f'Round 4 起主指標預設 mean_pauc（PAUC_MAX_FPRS={PAUC_MAX_FPRS} 的平均），'
                 '信賴區間預設用 block bootstrap（block_len=lookahead）取代 iid，'
                 '因為標籤在 lookahead 根之內高度重疊，iid 會低估 CI 寬度。'),
    }
    out_path = os.path.join(args.data_dir, 'gbdt_probe_report.json')
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n已存 {out_path}")


if __name__ == '__main__':
    main()
