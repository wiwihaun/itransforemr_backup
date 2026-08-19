"""
evaluate_event.py 與 predict_live.py 共用的邏輯：
- 從 meta.json（train_event.py 寫入的 setting/train_args）重建 args Namespace
- 建模型、載入 checkpoint、設 eval()
- Wilson score 信賴區間（門檻掃描與樣本數檢定用）
- 叢集進場狀態機（cluster_entries，Round 2 新增）

只依賴 meta.json 裡記下的資訊，不重新猜測訓練時的超參數 —— 這是
run.py 的 checkpoint 路徑與模型結構能被正確重建的唯一保證。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, make_scorer

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_meta(data_dir):
    meta_path = os.path.join(data_dir, 'meta.json')
    with open(meta_path) as f:
        meta = json.load(f)
    if 'setting' not in meta or 'train_args' not in meta:
        raise ValueError(
            f"{meta_path} 沒有 'setting'/'train_args' 欄位。"
            f"請先跑過 train_event.py（哪怕只加 --dry_run）產生這些欄位。"
        )
    return meta


def build_args_namespace(meta, data_dir, setting=None):
    """
    只包含 exp_basic / exp_long_term_forecasting / data_factory / Dataset_Custom /
    iTransformer 在 long_term_forecast 推論路徑上實際會讀到的欄位（已用
    grep 逐一核對過），而不是照抄 run.py 全部 ~80 個 CLI 參數。

    傳 setting 時優先查 meta['train_runs'][setting]（逐 setting 快照，
    compare_checkpoints.py 在同一份 meta.json 底下輪流重建好幾個不同超參數
    組合的模型時要用這個，不能用會被每次訓練覆寫的頂層 train_args/seq_len）；
    不傳或查無此 setting 時退回頂層 meta['train_args']/meta['seq_len']
    （= 目前 evaluate_event.py / predict_live.py 的既有行為，不變）。
    """
    train_runs = meta.get('train_runs', {})
    if setting is not None and setting in train_runs:
        ta = train_runs[setting]
        seq_len = ta['seq_len']
    else:
        ta = meta['train_args']
        seq_len = meta['seq_len']

    n_features = meta['n_features']  # 特徵數，+1 = 含 target 的變數 token 數

    return argparse.Namespace(
        task_name=ta['task_name'],
        model=ta['model'],
        data=ta['data'],
        model_id=ta['model_id'],
        root_path=os.path.abspath(data_dir),
        data_path=meta['data_path'],
        features=ta['features'],
        target=ta['target'],
        freq=ta['freq'],
        checkpoints='./checkpoints/',
        seq_len=seq_len,
        label_len=meta['label_len'],
        pred_len=meta['pred_len'],
        seasonal_patterns='Monthly',
        inverse=False,
        enc_in=n_features + 1,
        dec_in=n_features + 1,
        c_out=1,
        num_class=2,
        d_model=ta['d_model'],
        n_heads=ta['n_heads'],
        e_layers=ta['e_layers'],
        d_layers=ta['d_layers'],
        d_ff=ta['d_ff'],
        factor=ta['factor'],
        distil=ta['distil'],
        dropout=ta['dropout'],
        embed=ta['embed'],
        activation='gelu',
        expand=ta['expand'],
        d_conv=ta['d_conv'],
        num_workers=0,
        batch_size=ta['batch_size'],
        patience=ta['patience'],
        train_epochs=ta['train_epochs'],
        learning_rate=ta['learning_rate'],
        des=ta['des'],
        lradj=ta['lradj'],
        use_amp=False,
        use_gpu=True,
        gpu=0,
        gpu_type=ta.get('gpu_type', 'mps'),
        use_multi_gpu=False,
        devices='0,1,2,3',
        device_ids=[0],
        use_dtw=False,
        augmentation_ratio=0,
        focal_alpha=ta['focal_alpha'],
    )


def build_and_load_model(meta, data_dir, setting=None):
    """
    建 Exp_Long_Term_Forecast（順便完成 _build_model 並放到裝置上）、
    載入 checkpoint 權重、設 eval()。回傳 (exp, args)。

    不傳 setting 時用 meta['setting']（目前最近一次訓練的模型，evaluate_event.py
    / predict_live.py 的既有用法）；傳了就改重建那個指定 setting 的模型
    （compare_checkpoints.py 用來輪流載入好幾個不同超參數組合的 checkpoint）。

    會強制切到 repo 根目錄再匯入 —— exp_basic.py 的 _scan_models_directory()
    掃 'models' 資料夾、checkpoint 路徑都是相對於 cwd，不能假設呼叫端已經
    在正確的目錄下執行。
    """
    os.chdir(REPO_ROOT)
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)

    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast

    target_setting = setting if setting is not None else meta['setting']
    args = build_args_namespace(meta, data_dir, setting=target_setting)
    exp = Exp_Long_Term_Forecast(args)

    ckpt_path = os.path.join(REPO_ROOT, 'checkpoints', target_setting, 'checkpoint.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到 checkpoint: {ckpt_path}，請先跑 train_event.py。")

    state = torch.load(ckpt_path, map_location=exp.device)
    exp.model.load_state_dict(state)
    exp.model.eval()
    return exp, args


def wilson_ci(k, n, z=1.959963984540054):
    """Wilson score interval（95% 預設），回傳 (lo, hi)。n=0 時回傳 (0.0, 1.0)。"""
    if n == 0:
        return 0.0, 1.0
    phat = k / n
    denom = 1 + z ** 2 / n
    center = (phat + z ** 2 / (2 * n)) / denom
    margin = z * ((phat * (1 - phat) / n + z ** 2 / (4 * n ** 2)) ** 0.5) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


# ══════════════════════════════════════════════════════════════
# Partial AUC（Round 4）
#
# 動機：這套系統只交易 p_up 排進前 5~20% 的樣本，全域 AUC 把後面
# 80~95% 完全不會下單的樣本也算進去，是選模指標與實際決策的錯位。
# 改用 sklearn roc_auc_score(..., max_fpr=f)（McClish 標準化，
# no-skill=0.5，已用隨機分數實測 0.4989 確認），把「前 f% 假陽性率」
# 範圍內的排序品質單獨抓出來。事前註冊的主要指標是四個操作點
# （對應會交易的 5/10/15/20%）的平均，不是單一 k——避免挑中剛好在
# 某個 k 上運氣好的候選人。
# ══════════════════════════════════════════════════════════════
PAUC_MAX_FPRS = (0.05, 0.10, 0.15, 0.20)


def partial_auc(y_true, y_score, max_fpr):
    """單一操作點的 partial AUC（McClish 標準化，no-skill=0.5）。"""
    return float(roc_auc_score(y_true, y_score, max_fpr=max_fpr))


def mean_partial_auc(y_true, y_score, max_fprs=PAUC_MAX_FPRS):
    """
    Round 4 事前註冊的主要選模指標：PAUC_MAX_FPRS 四個操作點的平均。
    必須是 module-level 函式（不是 lambda/closure）——
    permutation_importance(n_jobs=-1) 要能 pickle 這個 callable。
    """
    return float(np.mean([roc_auc_score(y_true, y_score, max_fpr=f) for f in max_fprs]))


def pauc_report(y_true, y_score, max_fprs=PAUC_MAX_FPRS):
    """回傳 {'auc':, 'pauc_0.05':, ..., 'mean_pauc':}，供報告與 evaluate_report.json 使用。"""
    out = {'auc': float(roc_auc_score(y_true, y_score))}
    vals = []
    for f in max_fprs:
        v = partial_auc(y_true, y_score, f)
        out[f'pauc_{f:.2f}'] = v
        vals.append(v)
    out['mean_pauc'] = float(np.mean(vals))
    return out


def make_mean_pauc_scorer(max_fprs=PAUC_MAX_FPRS):
    """
    給 sklearn permutation_importance(scoring=...) 用的 scorer。
    用 response_method='predict_proba'（sklearn>=1.4；needs_proba= 已棄用）。
    """
    def _scorer(y_true, y_score):
        return mean_partial_auc(y_true, y_score, max_fprs=max_fprs)
    return make_scorer(_scorer, response_method='predict_proba')


def block_bootstrap_ci(y_true, y_score, metric_fn, block_len, n_boot=1000, seed=0,
                        ci_lo=2.5, ci_hi=97.5):
    """
    Moving-block bootstrap 信賴區間。標籤在 lookahead 根之內高度重疊
    （R3 實測 lag-1 一致率 0.8118，遠高於隨機基準 0.50），一般的
    iid 逐列重抽樣（bootstrap_auc_ci）會嚴重低估區間寬度。這裡改成
    整段連續 block_len 根一起抽（block_len 建議取 meta['lookahead']），
    保留區塊內的相依結構，CI 才誠實。

    y_true/y_score 必須是時間序（不能事先打亂），否則 block 沒有意義。
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    n_blocks = max(1, n // block_len)

    vals = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n - block_len + 1, n_blocks)
        idx = np.concatenate([np.arange(s, s + block_len) for s in starts])
        yt, ys = y_true[idx], y_score[idx]
        if len(np.unique(yt)) < 2:
            vals[b] = np.nan
            continue
        vals[b] = metric_fn(yt, ys)
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, ci_lo)), float(np.percentile(vals, ci_hi))


# ══════════════════════════════════════════════════════════════
# 叢集進場狀態機（backtest 版，陣列 index）
#
# 改編自 wiwihaun/itrans_online_0817 的 stream_entry()
# （toolbywi/r2_experiments.py 第 168-199 行）/ update_cluster_state()
# （live_trading/strategy.py）：cluster/gap/entry_at 判斷邏輯相同，
# 但這裡只認「買漲」單一方向（prob < threshold 的根完全不影響狀態），
# 且 busy 不用像原版那樣往前掃 TP/SL 出場點——事件合約是固定時間結算，
# busy = 進場 index + gap 直接是常數位移，不需要價格資料。
# ══════════════════════════════════════════════════════════════
def cluster_entries(prob, threshold, gap, entry_at=1):
    """
    只認「買漲」方向：prob[k] >= threshold 才算訊號，其餘完全忽略、不影響狀態。

    語意（對齊 delayed-entry skill）：
      - prob[k] < threshold 的根不更新任何狀態（不計入 cluster，也不動 last_sig）
      - 距離上一個訊號 > gap 才算新 cluster
      - entry_at=1 = 叢集第一根就進場，不是「延遲一根」
      - 同一叢集只進場一次；entry_at 之後每一根合格訊號都會繼續嘗試，
        直到成功進場或叢集結束（「持續開放的進場窗口」，不是單次嘗試）
      - busy（尚未結算）時就算輪到 entry_at 也不進場，但不消耗機會——
        entered 仍是 False，叢集內下一根合格訊號還會再試一次

    回傳：進場的 index 清單（由小到大排序）。
    """
    out = []
    last_sig = None
    cluster_count = 0
    entered = False
    busy = -1

    for k in range(len(prob)):
        if prob[k] < threshold:
            continue
        is_new = (last_sig is None) or (k - last_sig > gap)
        if is_new:
            cluster_count, entered = 1, False
        else:
            cluster_count += 1
        last_sig = k  # 不論是否被 busy 擋住都要更新，狀態機才跟串流一致

        if entered or cluster_count < entry_at:
            continue
        if k <= busy:
            continue

        out.append(k)
        busy = k + gap
        entered = True

    return out


def _cluster_entries_reference(prob, threshold, gap, entry_at):
    """
    獨立重寫的參考版本（跟 cluster_entries 語意相同、寫法不同），只用於自我測試。
    刻意不共用程式碼——如果只是複製貼上再改名字，測不出真正的邏輯錯誤；
    這裡先把完整訊號序列切成一個個 cluster，再在每個 cluster 內找
    「entry_at 之後第一個 busy 已過期的訊號」，跟串流版逐步判斷是同一件事，
    但用完全不同的資料流程算出來。
    """
    signal_idx = [k for k in range(len(prob)) if prob[k] >= threshold]

    out = []
    busy = -1
    i = 0
    while i < len(signal_idx):
        cluster = [signal_idx[i]]
        j = i + 1
        while j < len(signal_idx) and signal_idx[j] - cluster[-1] <= gap:
            cluster.append(signal_idx[j])
            j += 1

        if len(cluster) >= entry_at:
            for k in cluster[entry_at - 1:]:
                if k > busy:
                    out.append(k)
                    busy = k + gap
                    break
        i = j

    return out


def _test_cluster_entries_self_consistent():
    """
    合成隨機機率序列，比對 cluster_entries()（串流版）與獨立重寫的參考版本
    是否逐筆一致，比照 delayed-entry skill 的決定性驗證方法論
    （wiwihaun/itrans_online_0817 用同一招驗證 update_cluster_state() 移植版）。
    """
    rng = np.random.default_rng(42)
    n = 2000
    for threshold in (0.5, 0.6, 0.7):
        for gap in (3, 6, 12):
            for entry_at in (1, 2, 3, 6):
                prob = rng.random(n)
                a = cluster_entries(prob, threshold, gap, entry_at)
                b = _cluster_entries_reference(prob, threshold, gap, entry_at)
                assert a == b, (
                    f"cluster_entries 與參考版不一致 threshold={threshold} gap={gap} "
                    f"entry_at={entry_at}\n{a}\nvs\n{b}"
                )
    print("【cluster_entries 自我測試】通過 —— 多組 threshold/gap/entry_at 下，"
          "串流版與獨立重寫的參考版逐筆一致")


# ══════════════════════════════════════════════════════════════
# 叢集進場狀態機（live 版，逐步呼叫）
#
# cluster_entries() 的單步版本，供 predict_live.py 的常駐迴圈使用：每收到
# 新一根 K 棒（bar index k、機率 prob_k）就呼叫一次，state 在迴圈外的
# 記憶體變數裡（單一持續執行的 process，不需要像參考專案的 Firestore
# ——那是因為它們每個 tick 是獨立的 Cloud Run Job）。
# 語意必須跟 cluster_entries() 完全一致，用下面的自我測試逐步驗證。
# ══════════════════════════════════════════════════════════════
def new_cluster_state():
    return {'last_sig': None, 'cluster_count': 0, 'entered': False, 'busy': -1}


def cluster_step(state, k, prob_k, threshold, gap, entry_at=1):
    """
    回傳 (new_state, should_enter)。should_enter=True 代表這根 K 棒符合進場
    條件（只認買漲方向，prob_k < threshold 一律 False 且不動狀態）。
    """
    if prob_k < threshold:
        return dict(state), False

    last_sig = state.get('last_sig')
    is_new = (last_sig is None) or (k - last_sig > gap)
    if is_new:
        cluster_count, entered = 1, False
    else:
        cluster_count, entered = state.get('cluster_count', 0) + 1, state.get('entered', False)

    new_state = dict(state, last_sig=k, cluster_count=cluster_count, entered=entered)

    busy = state.get('busy', -1)
    if entered or cluster_count < entry_at:
        return new_state, False
    if k <= busy:
        return new_state, False

    new_state['entered'] = True
    new_state['busy'] = k + gap
    return new_state, True


def _test_cluster_step_matches_cluster_entries():
    """逐步呼叫 cluster_step() 走完整段序列，跟批次版 cluster_entries() 的
    結果必須逐筆相同——這是 predict_live.py 唯一會用到的介面，若跟評估用的
    批次版邏輯分岔，即時訊號就會跟回測驗證過的統計數字對不上。"""
    rng = np.random.default_rng(7)
    n = 1500
    for threshold in (0.5, 0.55, 0.65):
        for gap in (3, 6, 12):
            for entry_at in (1, 2, 3, 6):
                prob = rng.random(n)
                batch = cluster_entries(prob, threshold, gap, entry_at)

                state = new_cluster_state()
                stepped = []
                for k in range(n):
                    state, should_enter = cluster_step(state, k, prob[k], threshold, gap, entry_at)
                    if should_enter:
                        stepped.append(k)

                assert stepped == batch, (
                    f"cluster_step 逐步版跟 cluster_entries 批次版不一致 "
                    f"threshold={threshold} gap={gap} entry_at={entry_at}\n{stepped}\nvs\n{batch}"
                )
    print("【cluster_step 自我測試】通過 —— 逐步呼叫版與批次版 cluster_entries() 逐筆一致")


if __name__ == '__main__':
    _test_cluster_entries_self_consistent()
    _test_cluster_step_matches_cluster_entries()
