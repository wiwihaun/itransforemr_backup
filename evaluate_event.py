#!/usr/bin/env python3
"""
評估關卡 Round 3 —— 門檻在「驗證集」上用百分位掃描、鎖定，entry_at 固定為 1
（不再網格選擇——Round 2 已證明驗證集量級撐不起可靠的 entry_at 選擇，且
entry_at=1 本身樣本最大、CI 最穩）。測試集只用鎖定的門檻跑一次叢集進場、
評估一次，並輸出逐筆交易明細供報告畫圖。

叢集進場（cluster_entries，見 toolbywi/event_common.py）保證任兩筆交易的
進場 index 至少差 gap 根，交易彼此不重疊，Wilson CI 才站得住腳。

門檻改用「驗證集百分位」而非絕對值：Round 2 發現模型輸出機率極度壓縮
（std 僅 0.0145），絕對門檻網格常常掃不到有效區間（例如 >=0.56 完全 0 筆）。
用百分位選門檻對機率尺度免疫，也讓不同損失函數之間可以公平比較——不同
損失會產生完全不同的機率尺度，用絕對門檻比較會失真。

只做「買漲」方向：p_up >= threshold 才算訊號，p_up <= 1-threshold 完全忽略、
不下單（對應使用者的規格）。

用法：
  .venv/bin/python evaluate_event.py --data_dir ./dataset/event30m_v3_sel --odds 0.85
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'toolbywi'))
from event_common import load_meta, build_and_load_model, wilson_ci, cluster_entries, pauc_report


def collect_predictions(exp, args, flag):
    """
    收集某個切分（train/val/test）的預測機率與真實標籤，保證回傳順序是
    時間序（chronological）。

    重要修正：data_provider() 對 flag != 'test' 回傳的 DataLoader 預設
    shuffle=True（見 data_provider/data_factory.py:26），而 cluster_entries()
    的 gap 判斷邏輯（k - last_sig > gap）要求陣列 position 就是時間順序——
    直接用那個 DataLoader 收集出來的驗證集機率序列，跑 cluster_entries() 會
    在亂序資料上算出沒有意義的「叢集」。這裡一律自己重建 shuffle=False 的
    DataLoader，不管 flag 是什麼、也不管 data_provider() 預設行為為何，
    確保收集到的序列永遠是時間序。
    """
    from data_provider.data_factory import data_provider
    data_set, _ = data_provider(args, flag)
    data_loader = DataLoader(data_set, batch_size=args.batch_size, shuffle=False,
                              num_workers=0, drop_last=False)

    all_logits, all_true = [], []
    with torch.no_grad():
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            batch_x = batch_x.float().to(exp.device)
            batch_y = batch_y.float()
            batch_x_mark = batch_x_mark.float().to(exp.device)
            batch_y_mark = batch_y_mark.float().to(exp.device)

            dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(exp.device)

            outputs = exp.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            f_dim = -1 if args.features == 'MS' else 0
            outputs = outputs[:, -args.pred_len:, f_dim:]
            batch_y = batch_y[:, -args.pred_len:, f_dim:]

            all_logits.append(outputs.detach().cpu().numpy().reshape(-1))
            all_true.append(batch_y.detach().cpu().numpy().reshape(-1))

    logits = np.concatenate(all_logits)
    true = np.concatenate(all_true)
    true_binary = (true > 0).astype(int)  # 與訓練用的損失函數定義一致
    proba = expit(logits)
    return proba, true_binary, len(data_set)


def evaluate_cluster(proba, true_binary, threshold, gap, entry_at):
    """對單一 threshold 跑叢集進場（只認買漲方向，entry_at 固定傳入值），回傳勝率統計。"""
    entries = cluster_entries(proba, threshold, gap, entry_at)
    n_trades = len(entries)
    if n_trades == 0:
        return {'threshold': threshold, 'gap': gap, 'entry_at': entry_at,
                'n_trades': 0, 'n_correct': 0, 'win_rate': None, 'ci_lo': None, 'ci_hi': None,
                'entries': []}

    n_correct = int(sum(true_binary[k] == 1 for k in entries))
    win_rate = n_correct / n_trades
    ci_lo, ci_hi = wilson_ci(n_correct, n_trades)
    return {'threshold': threshold, 'gap': gap, 'entry_at': entry_at,
            'n_trades': n_trades, 'n_correct': n_correct,
            'win_rate': win_rate, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
            'entries': entries}


def busy_only_entries(prob, threshold, gap):
    """
    次要參考統計：只用 busy 鎖倉（不要求叢集斷開才重置 entered），只認買漲
    方向。跟 cluster_entries() 一樣保證任兩筆交易間隔 > gap、Wilson CI 一樣
    有效，但不要求「同一叢集只進場一次」，樣本數通常大很多。用來量化叢集
    規則本身付出的統計功效代價，不是主結論——主結論仍以 cluster_entries()
    （使用者指定的叢集進場限制）為準。
    """
    out = []
    busy = -1
    for k in range(len(prob)):
        if prob[k] < threshold:
            continue
        if k <= busy:
            continue
        out.append(k)
        busy = k + gap
    return out


def evaluate_busy_only(proba, true_binary, threshold, gap):
    entries = busy_only_entries(proba, threshold, gap)
    n_trades = len(entries)
    if n_trades == 0:
        return {'n_trades': 0, 'win_rate': None, 'ci_lo': None, 'ci_hi': None}
    n_correct = int(sum(true_binary[k] == 1 for k in entries))
    win_rate = n_correct / n_trades
    ci_lo, ci_hi = wilson_ci(n_correct, n_trades)
    return {'n_trades': n_trades, 'n_correct': n_correct, 'win_rate': win_rate,
            'ci_lo': ci_lo, 'ci_hi': ci_hi}


def build_trades_csv(entries, proba, true_binary, dates, odds):
    """
    測試集逐筆交易明細，供報告畫開單時間圖、權益曲線/回撤圖。
    entry_idx 是測試集陣列裡的 0-index，dates 是測試集對應的時間戳序列
    （已用 df.tail(num_test) 對齊，見 main() 裡的推導）。
    """
    rows = []
    equity = 1.0
    peak = 1.0
    for idx in entries:
        outcome = int(true_binary[idx])
        pnl = odds if outcome == 1 else -1.0
        stake = 0.02 * equity  # 每筆固定 2% 權益
        equity = equity + stake * pnl
        peak = max(peak, equity)
        drawdown = (equity - peak) / peak if peak > 0 else 0.0
        rows.append({
            'entry_idx': int(idx),
            'entry_time': str(dates.iloc[idx]),
            'p_up': float(proba[idx]),
            'outcome': outcome,
            'equity': equity,
            'drawdown': drawdown,
        })
    return pd.DataFrame(rows)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default='./dataset/event30m_v3_sel')
    p.add_argument('--odds', type=float, default=0.85, help='事件合約賠率（贏了賺 r 倍本金），決定損益兩平勝率')
    p.add_argument('--min_trades', type=int, default=100,
                    help='叢集化後樣本數天生變少，門檻比逐根版本調降')
    p.add_argument('--percentiles', default='30,20,10,5,3,2,1',
                    help='驗證集上「前 q%%」的候選門檻，逗號分隔')
    p.add_argument('--entry_at', type=int, default=1,
                    help='固定值，不再網格選擇（Round 2 已證明驗證集量級撐不起可靠的 entry_at 選擇）')
    p.add_argument('--gap', type=int, default=None, help='不給則用 meta.json 的 lookahead')
    p.add_argument('--setting', default=None,
                    help='不給則用 meta.json 頂層 setting（= 最近一次 train_event.py 訓練的模型）；'
                         '同一份 meta.json 底下比較過多個超參數組合時（見 compare_checkpoints.py），'
                         '用這個明確指定要評估哪一個 checkpoint')
    return p.parse_args()


def main():
    args = parse_args()
    meta = load_meta(args.data_dir)
    exp, model_args = build_and_load_model(meta, args.data_dir, setting=args.setting)

    gap = args.gap if args.gap is not None else meta['lookahead']
    breakeven = 1.0 / (1.0 + args.odds)
    print(f"gap={gap}（對齊 lookahead={meta['lookahead']}）  entry_at={args.entry_at}（事前固定，非從測試集選出）")
    print(f"損益兩平勝率（賠率 {args.odds}）= {breakeven:.4f}\n")

    print("=== 驗證集：門檻百分位掃描（門檻只能在這裡選）===")
    val_proba, val_true, n_val = collect_predictions(exp, model_args, 'val')
    print(f"驗證集樣本數: {n_val}，正樣本比例: {val_true.mean():.4f}")
    print(f"驗證集機率分佈: min={val_proba.min():.4f} p50={np.percentile(val_proba,50):.4f} "
          f"max={val_proba.max():.4f} std={val_proba.std():.4f}")
    val_pauc = pauc_report(val_true, val_proba)
    print(f"驗證集 AUC={val_pauc['auc']:.4f}  平均 pAUC={val_pauc['mean_pauc']:.4f}  "
          f"（純描述，不影響門檻選擇——門檻仍依 CI 下界，見 Round 4 plan）\n")

    percentiles = [float(q) for q in args.percentiles.split(',')]
    grid = []
    print(f"{'前q%':>6} {'絕對門檻':>10} {'下單數':>8} {'勝率':>8} {'CI下界':>8} {'CI上界':>8}")
    for q in percentiles:
        th = float(np.percentile(val_proba, 100 - q))
        r = evaluate_cluster(val_proba, val_true, th, gap, args.entry_at)
        r['percentile'] = q
        grid.append(r)
        if r['n_trades'] == 0:
            print(f"{q:>6.1f} {th:>10.4f} {0:>8} {'--':>8} {'--':>8} {'--':>8}")
        else:
            print(f"{q:>6.1f} {th:>10.4f} {r['n_trades']:>8} {r['win_rate']:>8.4f} "
                  f"{r['ci_lo']:>8.4f} {r['ci_hi']:>8.4f}")

    eligible = [r for r in grid if r['n_trades'] >= args.min_trades]
    if not eligible:
        print(f"\n⚠️ 沒有任何百分位候選在驗證集上有 >= {args.min_trades} 筆下單，"
              f"樣本量不足以做出可信的選擇。無法通過關卡。")
        chosen = None
    else:
        chosen = max(eligible, key=lambda r: r['ci_lo'])
        print(f"\n✅ 鎖定門檻 = {chosen['threshold']:.4f}（驗證集前 {chosen['percentile']:.1f}%）"
              f"（驗證集：下單 {chosen['n_trades']} 筆，勝率 {chosen['win_rate']:.4f}，"
              f"CI 下界 {chosen['ci_lo']:.4f}）")

    print("\n=== 測試集：只評估這一次 ===")
    test_proba, test_true, n_test = collect_predictions(exp, model_args, 'test')
    print(f"測試集樣本數: {n_test}，正樣本比例: {test_true.mean():.4f}")  # noqa: lookahead - 純描述，決策已在驗證集鎖定
    test_pauc = pauc_report(test_true, test_proba)  # noqa: lookahead - 純描述，不影響任何決策
    print(f"測試集 AUC={test_pauc['auc']:.4f}  平均 pAUC={test_pauc['mean_pauc']:.4f}  "
          f"（Round 4：pAUC 與 AUC 不同尺度，不可跨輪直接比較數字，只能各自對照各自的事前基準）")

    always_up_win_rate = float(test_true.mean())  # noqa: lookahead - 描述性基準，非決策
    majority_baseline = float(max(test_true.mean(), 1 - test_true.mean()))  # noqa: lookahead - 同上
    print(f"基準：全押漲勝率 = {always_up_win_rate:.4f}，多數類別基準 = {majority_baseline:.4f}")

    result = {
        'gap': gap, 'entry_at': args.entry_at, 'breakeven': breakeven, 'min_trades': args.min_trades,
        'val_grid': [{k: v for k, v in r.items() if k != 'entries'} for r in grid],
        'val_pauc_report': val_pauc,
        'test_pauc_report': test_pauc,
        'chosen_percentile': chosen['percentile'] if chosen else None,
        'chosen_threshold': chosen['threshold'] if chosen else None,
        'test': None,
        'busy_only_secondary': None,
        'baselines': {'always_up': always_up_win_rate, 'majority': majority_baseline},
        'passed': False,
        'note': ('門檻用驗證集百分位選定並凍結成絕對值，entry_at 事前固定為 '
                 f'{args.entry_at}（不做任何選擇/掃描）。叢集進場保證交易間隔 > gap，'
                 '彼此不重疊，Wilson CI 站得住腳。busy_only_secondary 是次要參考統計'
                 '（只鎖倉不要求叢集斷開），量化叢集規則本身付出的統計功效代價，不是主結論。'
                 'val_pauc_report/test_pauc_report（Round 4 新增）純描述用，關卡本身仍只看'
                 '勝率 + Wilson CI，不加 pAUC 通關條件——加了等於對測試集咬第二口。'),
    }

    if chosen is not None:
        test_report = evaluate_cluster(test_proba, test_true, chosen['threshold'], gap, args.entry_at)
        entries = test_report.pop('entries')
        result['test'] = test_report

        if len(entries) > 1:
            min_gap_seen = min(b - a for a, b in zip(entries, entries[1:]))
            assert min_gap_seen > gap, f"交易重疊！最小間隔 {min_gap_seen} <= gap {gap}"
            print(f"\n交易不重疊檢查通過（{len(entries)} 筆交易，最小間隔 {min_gap_seen} 根 > gap {gap}）")

        print(f"\n門檻={chosen['threshold']:.4f}, entry_at={args.entry_at} 在測試集上：")
        if test_report['n_trades'] == 0:
            print("  沒有任何下單訊號。")
        else:
            print(f"  下單數: {test_report['n_trades']}")
            print(f"  勝率: {test_report['win_rate']:.4f}")
            print(f"  95% CI: [{test_report['ci_lo']:.4f}, {test_report['ci_hi']:.4f}]")

            passed = bool(test_report['n_trades'] >= args.min_trades and
                          test_report['ci_lo'] > breakeven)
            result['passed'] = passed

            print(f"\n{'✅ 通過' if passed else '❌ 未通過'} 關卡標準："
                  f"下單數 >= {args.min_trades} 且 95% CI 下界 > 損益兩平勝率 {breakeven:.4f}")
            if not passed:
                print("未通過就不建議進到 predict_live.py 常駐監控。")

            # --- 次要參考統計：僅不重疊版本 ---
            busy_report = evaluate_busy_only(test_proba, test_true, chosen['threshold'], gap)
            result['busy_only_secondary'] = busy_report
            if busy_report['n_trades'] > 0:
                print(f"\n[次要參考] 同門檻下「僅不重疊」（不要求叢集斷開）在測試集："
                      f"下單數 {busy_report['n_trades']}，勝率 {busy_report['win_rate']:.4f}，"
                      f"95% CI [{busy_report['ci_lo']:.4f}, {busy_report['ci_hi']:.4f}]")

            # --- 逐筆交易明細，供報告畫圖 ---
            raw = pd.read_csv(os.path.join(args.data_dir, meta['raw_ohlcv_path']))
            test_dates = raw['date'].tail(n_test).reset_index(drop=True)
            trades_df = build_trades_csv(entries, test_proba, test_true, test_dates, args.odds)
            trades_path = os.path.join(args.data_dir, 'trades.csv')
            trades_df.to_csv(trades_path, index=False)
            print(f"\n已存 {trades_path}（{len(trades_df)} 筆交易明細）")

    out_path = os.path.join(args.data_dir, 'evaluate_report.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n已存 {out_path}")


if __name__ == '__main__':
    main()
