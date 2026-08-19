#!/usr/bin/env python3
"""
比較多個 checkpoint 在「驗證集」上的表現，只用來在候選模型之間選贏家——
絕不碰測試集。Round 3 的損失函數比較（階段 A，6 個候選）與深度比較
（階段 B，2 個候選）都用這支腳本選，最終選定的單一組合才交給
evaluate_event.py 在測試集上評估一次。這是刻意的設計：如果每個候選都各自
去碰一次測試集再挑最好的，等於用測試集資訊做了模型選擇，就是
no-lookahead-bias skill 講的 L4 決策洩漏。

用法：
  .venv/bin/python compare_checkpoints.py --data_dir ./dataset/event30m_v3_sel \
      --settings "setting1,setting2,..." --labels "focal_g2,bce,..."

Round 4 修正：選模指標預設從 val_auc 改成 val_mean_pauc（見
toolbywi/event_common.py 的 PAUC_MAX_FPRS）——這套系統只交易 val_proba
排進前 5~20% 的樣本，全域 AUC 把後面完全不會下單的樣本也算進去，會挑到
「整體排序好但交易區間沒有邊際」的候選。--select_metric auc 可切回 Round 3
的舊行為以便回溯比較；兩個指標永遠都計算並輸出，不管選哪個當主指標。
"""
import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toolbywi'))
from event_common import load_meta, build_and_load_model, wilson_ci, cluster_entries, mean_partial_auc
from evaluate_event import collect_predictions


def load_specific_checkpoint(meta, data_dir, setting):
    """用指定的 setting 重建模型——依 meta['train_runs'][setting] 這份逐
    setting 快照（見 train_event.py），不依賴會被每次訓練覆寫的頂層
    train_args，這樣同一份 meta.json 才能正確輪流重建好幾個不同超參數
    組合訓練出來的 checkpoint。"""
    return build_and_load_model(meta, data_dir, setting=setting)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default='./dataset/event30m_v3_sel')
    p.add_argument('--settings', required=True, help='逗號分隔的 setting 字串清單')
    p.add_argument('--labels', required=True, help='逗號分隔的顯示用標籤，跟 settings 一一對應')
    p.add_argument('--top_pct', type=float, default=5.0,
                    help='參考用的驗證集門檻百分位（只供比較模型優劣，不是最終要用的門檻——'
                         '最終門檻由 evaluate_event.py 對「選定後的那個模型」重新掃描選出）')
    p.add_argument('--select_metric', default='mean_pauc', choices=['mean_pauc', 'auc'],
                    help='Round 4 預設 mean_pauc——只看系統實際會下單的區間排序品質；'
                         'auc 是 Round 1-3 的舊行為，供回溯比較。兩個指標永遠都算、都輸出，'
                         '只差在哪個決定 best')
    return p.parse_args()


def main():
    args = parse_args()
    meta = load_meta(args.data_dir)
    gap = meta['lookahead']
    settings = [s.strip() for s in args.settings.split(',')]
    labels = [l.strip() for l in args.labels.split(',')]
    assert len(settings) == len(labels), "settings 與 labels 數量必須相同"

    results = []
    print(f"{'label':14s} {'val_AUC':>9s} {'val_meanPAUC':>13s} {'top':>5s} {'n_trades':>9s} "
          f"{'win_rate':>9s} {'ci_lo':>9s}")
    for label, setting in zip(labels, settings):
        exp, model_args = load_specific_checkpoint(meta, args.data_dir, setting)
        val_proba, val_true, n_val = collect_predictions(exp, model_args, 'val')
        auc = float(roc_auc_score(val_true, val_proba))
        mean_pauc = float(mean_partial_auc(val_true, val_proba))

        th = float(np.percentile(val_proba, 100 - args.top_pct))
        entries = cluster_entries(val_proba, th, gap, 1)
        n_trades = len(entries)
        win_rate, ci_lo = None, None
        if n_trades > 0:
            n_correct = int(sum(val_true[k] == 1 for k in entries))
            win_rate = n_correct / n_trades
            ci_lo, _ = wilson_ci(n_correct, n_trades)

        results.append({
            'label': label, 'setting': setting, 'val_auc': auc, 'val_mean_pauc': mean_pauc,
            'val_n_trades_top_pct': n_trades, 'val_win_rate_top_pct': win_rate,
            'val_ci_lo_top_pct': ci_lo, 'top_pct': args.top_pct,
        })
        wr_s = f"{win_rate:.4f}" if win_rate is not None else "--"
        ci_s = f"{ci_lo:.4f}" if ci_lo is not None else "--"
        print(f"{label:14s} {auc:>9.4f} {mean_pauc:>13.4f} {args.top_pct:>4.0f}% {n_trades:>9d} "
              f"{wr_s:>9s} {ci_s:>9s}")

    select_key = 'val_mean_pauc' if args.select_metric == 'mean_pauc' else 'val_auc'
    best = max(results, key=lambda r: r[select_key])
    print(f"\n最佳（依{args.select_metric}）: {best['label']}（{best['setting']}）"
          f"{args.select_metric}={best[select_key]:.4f}"
          f"（同一候選的 auc={best['val_auc']:.4f}, mean_pauc={best['val_mean_pauc']:.4f}——"
          f"兩個指標不同尺度，只能各自比較同指標的候選，不可跨指標比大小）")

    out_path = os.path.join(args.data_dir, 'checkpoint_comparison.json')
    existing = []
    if os.path.exists(out_path):
        existing = json.load(open(out_path))
    with open(out_path, 'w') as f:
        json.dump(existing + [{'best_label': best['label'], 'select_metric': args.select_metric,
                                'results': results}], f,
                  indent=2, ensure_ascii=False)
    print(f"已存 {out_path}")


if __name__ == '__main__':
    main()
