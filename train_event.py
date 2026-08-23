#!/usr/bin/env python3
"""
訓練包裝層 —— 不改 run.py，只是組出訓練參數、算出與 run.py:208-227 逐字元
相同的 setting 字串（之後 evaluate_event.py / predict_live.py 要用它重建
checkpoint 路徑），然後 subprocess 呼叫 run.py。

必須在 repo 根目錄執行（run.py 的 ./checkpoints/ 路徑是相對於 cwd 的）。

用法：
  .venv/bin/python train_event.py --data_dir ./dataset/event30m
  .venv/bin/python train_event.py --data_dir ./dataset/event30m --dry_run   # 只看 setting/指令
"""
import argparse
import json
import os
import re
import subprocess
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default='./dataset/event30m',
                    help='prepare_event_data.py 的輸出目錄（含 meta.json / stock_features.csv）')
    p.add_argument('--model_id', default='BTC_EVENT_30M')
    p.add_argument('--des', default='BTC_EVENT_30M')
    p.add_argument('--seq_len', type=int, default=None,
                    help='不給則用 meta.json 既有的 seq_len；給了會覆寫並寫回 meta.json，'
                         '不用重跑 prepare_event_data.py')
    p.add_argument('--model', default='iTransformer', choices=['iTransformer', 'iTransformerRes'],
                    help='iTransformerRes 是 Round 3 的殘差深層版（見 models/iTransformerRes.py）')
    p.add_argument('--loss_variant', default='focal_g2',
                    choices=['focal_g2', 'focal_g05', 'bce', 'bce_ls', 'brier', 'rank'],
                    help='見 exp/exp_long_term_forecasting.py 的 build_loss_criterion()，'
                         '預設 focal_g2 與 Round 1/2 行為完全相同')
    p.add_argument('--drop_path', type=float, default=0.1,
                    help='只有 --model iTransformerRes 會用到；DropPath 在最後一層的機率')
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--d_ff', type=int, default=512)
    p.add_argument('--e_layers', type=int, default=2)
    p.add_argument('--n_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--learning_rate', type=float, default=1e-4)
    p.add_argument('--lradj', default='type3')
    p.add_argument('--patience', type=int, default=15)
    p.add_argument('--train_epochs', type=int, default=100)
    p.add_argument('--focal_alpha', type=float, default=None,
                    help='不給則用 meta.json 的 suggested_focal_alpha')
    p.add_argument('--gpu_type', default='mps', choices=['cuda', 'mps'])
    p.add_argument('--dry_run', action='store_true', help='只印 setting/指令並寫回 meta.json，不執行訓練')
    p.add_argument('--no_tee', action='store_true',
                    help='Round 4 預設把訓練 stdout tee 進 logs/{setting}.log 並解析出逐 epoch '
                         'loss（供 plot_training_curve.py 畫圖）；加這個旗標還原 Round 1-3 的舊行為'
                         '（直接 subprocess.run，不擷取、不解析）')
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = os.path.abspath(args.data_dir)
    meta_path = os.path.join(data_dir, 'meta.json')
    meta = json.load(open(meta_path))

    # 訓練前先擋一次：特徵表多出任何欄位都會被 Dataset_Custom 當成額外模型
    # 輸入（不遮蔽），若那一欄含未來資訊就是完全洩漏且不會報錯
    sys.path.insert(0, os.path.join(REPO_ROOT, 'toolbywi'))
    from event_common import assert_feature_table_matches_meta
    assert_feature_table_matches_meta(data_dir, meta)

    focal_alpha = args.focal_alpha if args.focal_alpha is not None else meta['suggested_focal_alpha']

    # --- Round 5 防呆：兩個損失變體在極度類別不平衡下會靜默失效。
    #     Round 3 是在 50/50 標籤下比較這 6 個變體的，那個前提在
    #     run_center 標籤（正樣本率約 1.8%）下不再成立。
    #     pos_rate≈0.5 時兩個檢查都不可能觸發，Rounds 1-4 完全不受影響。---
    pos_rate = meta.get('pos_rate')
    if pos_rate is not None:
        if args.loss_variant == 'bce_ls' and pos_rate < 0.10:
            raise SystemExit(
                f"❌ 拒絕執行：loss_variant=bce_ls 在 pos_rate={pos_rate:.4f} 下已失效。\n"
                f"   BCELabelSmoothLoss(smoothing=0.05) 會把負樣本目標推到 0.05，"
                f"比基準率 {pos_rate:.4f} 還高，等於反過來把模型往「多猜正樣本」偏，"
                f"不是正則化。請改用 focal_g2（alpha 會自動由類別比例推導）。")
        if args.loss_variant == 'rank':
            p_empty = (1 - pos_rate) ** args.batch_size
            if p_empty > 0.05:
                raise SystemExit(
                    f"❌ 拒絕執行：loss_variant=rank 在 pos_rate={pos_rate:.4f}、"
                    f"batch_size={args.batch_size} 下，整批沒有正樣本的機率是 "
                    f"{p_empty:.1%}，那些 batch 會直接回傳零梯度（見 PairwiseRankLoss "
                    f"的防禦分支，其 docstring 明講假設 ~50/50 標籤）。\n"
                    f"   請改用 focal_g2，或把 batch_size 提高到 "
                    f"{int(np.ceil(np.log(0.05) / np.log(1 - pos_rate)))} 以上。")

    # --- 固定值，不依賴 run.py 的 argparse 預設，全部在指令列上明確傳入 ---
    task_name = 'long_term_forecast'
    model = args.model
    data = 'custom'
    features = 'MS'
    expand, d_conv, factor, d_layers, embed = 2, 4, 1, 1, 'timeF'
    distil = True  # 對應 run.py 的 --distil (store_false)；不傳這個 flag 就維持 True

    seq_len = args.seq_len if args.seq_len is not None else meta['seq_len']
    label_len = meta['label_len']
    pred_len = meta['pred_len']

    # 與 run.py:208-227 逐字元相同的組法，訓練/評估/推論三支程式共用同一份 setting。
    setting = (
        f"{task_name}_{args.model_id}_{model}_{data}_ft{features}"
        f"_sl{seq_len}_ll{label_len}_pl{pred_len}"
        f"_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{d_layers}"
        f"_df{args.d_ff}_expand{expand}_dc{d_conv}_fc{factor}_eb{embed}"
        f"_dt{distil}_{args.des}_0"
    )

    cmd = [
        sys.executable, '-u', 'run.py',
        '--task_name', task_name, '--is_training', '1',
        '--root_path', data_dir,
        '--data_path', meta['data_path'],
        '--model_id', args.model_id, '--model', model, '--data', data,
        '--features', features, '--target', 'target', '--freq', 't',
        '--seq_len', str(seq_len), '--label_len', str(label_len), '--pred_len', str(pred_len),
        '--d_model', str(args.d_model), '--d_ff', str(args.d_ff),
        '--e_layers', str(args.e_layers), '--n_heads', str(args.n_heads),
        '--dropout', str(args.dropout),
        '--expand', str(expand), '--d_conv', str(d_conv), '--factor', str(factor),
        '--d_layers', str(d_layers), '--embed', embed,
        '--batch_size', str(args.batch_size), '--patience', str(args.patience),
        '--train_epochs', str(args.train_epochs),
        '--learning_rate', str(args.learning_rate), '--lradj', args.lradj,
        '--focal_alpha', str(focal_alpha),
        '--loss_variant', args.loss_variant, '--drop_path', str(args.drop_path),
        '--des', args.des, '--itr', '1', '--num_workers', '0',
        '--use_gpu', '--gpu_type', args.gpu_type,
    ]

    print("=== setting ===")
    print(setting)
    print("\n=== 指令 ===")
    print(' '.join(cmd))

    # 把 setting 與這次訓練用的超參數寫回 meta.json，evaluate_event.py / predict_live.py 要重建
    # 一模一樣的模型與 checkpoint 路徑，全靠這份紀錄。event_common.py 的
    # build_args_namespace() 是直接讀 meta['seq_len']（不是 train_args 裡的），
    # 所以 --seq_len 覆寫也必須寫回這裡，否則評估/即時推論會用錯窗口長度。
    train_args_dict = {
        'model_id': args.model_id, 'des': args.des, 'task_name': task_name,
        'model': model, 'data': data, 'features': features,
        'd_model': args.d_model, 'd_ff': args.d_ff, 'e_layers': args.e_layers,
        'n_heads': args.n_heads, 'dropout': args.dropout,
        'expand': expand, 'd_conv': d_conv, 'factor': factor,
        'd_layers': d_layers, 'embed': embed, 'distil': distil,
        'batch_size': args.batch_size, 'patience': args.patience,
        'train_epochs': args.train_epochs, 'learning_rate': args.learning_rate,
        'lradj': args.lradj, 'focal_alpha': focal_alpha, 'gpu_type': args.gpu_type,
        'target': 'target', 'freq': 't',
        'loss_variant': args.loss_variant, 'drop_path': args.drop_path,
    }

    meta['seq_len'] = seq_len
    meta['setting'] = setting
    meta['train_args'] = train_args_dict

    # 額外用「setting 字串 -> 該次超參數」的字典累積保存（絕不覆寫舊 key，
    # 只新增）——meta['train_args'] 是單一欄位，每跑一次就被蓋掉，只反映
    # 「最近一次」訓練。Round 3 同一份 meta.json 底下要輪流比較好幾個不同
    # 超參數組合訓練出來的 checkpoint（見 compare_checkpoints.py），沒有
    # 這份逐 setting 快照就無法正確重建除了最後一次以外的任何模型架構。
    meta.setdefault('train_runs', {})
    meta['train_runs'][setting] = {**train_args_dict, 'seq_len': seq_len}

    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"\n已把 setting 與訓練參數寫回 {meta_path}")

    if args.dry_run:
        print("\n[dry_run] 不執行訓練。")
        return

    print("\n=== 開始訓練 ===")
    if args.no_tee:
        # Round 1-3 舊行為：不擷取 stdout，也就沒有逐 epoch loss 可畫圖。
        result = subprocess.run(cmd, cwd=REPO_ROOT)
        if result.returncode != 0:
            raise RuntimeError(f"run.py 以非 0 狀態結束: {result.returncode}")
    else:
        returncode = run_and_tee(cmd, REPO_ROOT, data_dir, setting)
        if returncode != 0:
            raise RuntimeError(f"run.py 以非 0 狀態結束: {returncode}")

    ckpt_path = os.path.join(REPO_ROOT, 'checkpoints', setting, 'checkpoint.pth')
    if os.path.exists(ckpt_path):
        print(f"\n✅ 訓練完成，checkpoint: {ckpt_path}")
    else:
        print(f"\n⚠️ 訓練結束但找不到預期的 checkpoint: {ckpt_path}")


# 逐 epoch 三個 loss，對應 exp/exp_long_term_forecasting.py train() 第 273 行左右
# 的 print 格式：「Epoch: {n}, Steps: {s} | Train Loss: {a} Vali Loss: {b} Test Loss: {c}」。
# 沿用 toolbywi/experiment_runner.py 既有的 tee + regex 模式，只是擴充成擷取三個 loss
# 而非只找 vali_loss 最小值，並額外擷取存檔/早停/學習率事件供圖上標註。
_EPOCH_RE = re.compile(
    r'Epoch:\s*(\d+),\s*Steps:\s*(\d+)\s*\|\s*Train Loss:\s*([\d.eE+-]+)\s*'
    r'Vali Loss:\s*([\d.eE+-]+)\s*Test Loss:\s*([\d.eE+-]+)'
)
_SAVED_RE = re.compile(r'Validation loss decreased')
_ES_COUNTER_RE = re.compile(r'EarlyStopping counter:\s*(\d+)\s*out of\s*(\d+)')
_ES_STOP_RE = re.compile(r'^Early stopping')
_LR_RE = re.compile(r'Updating learning rate to\s*([\d.eE+-]+)')


def run_and_tee(cmd, cwd, data_dir, setting):
    """
    Popen(cmd) 並把 stdout 即時印到終端機的同時逐行寫進 logs/{setting}.log，
    再解析出逐 epoch loss 存進 {data_dir}/training_history.json（累加式，
    用 setting 當 key，同 train_event.py 對 meta['train_runs'] 的做法，不覆寫
    其他 setting 的歷史）。

    只加在這支包裝層——exp/exp_long_term_forecasting.py 是 evaluate_event.py /
    compare_checkpoints.py / predict_live.py 重載 R1-R3 checkpoint 的必經路徑，
    不動它才能保證所有歷史 checkpoint 都還載得動。第 87 行的 cmd 已經帶 -u，
    run.py 的 stdout 本來就是不緩衝逐行輸出，這裡才抓得到完整逐行紀錄。
    """
    logs_dir = os.path.join(REPO_ROOT, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f'{setting}.log')

    epochs = []
    saved_epochs = []
    early_stop_epoch = None
    lr_updates = []

    with open(log_path, 'w') as logf:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, cwd=cwd)
        cur_epoch = None
        for line in proc.stdout:
            print(line, end='')
            logf.write(line)

            m = _EPOCH_RE.search(line)
            if m:
                cur_epoch = int(m.group(1))
                epochs.append({
                    'epoch': cur_epoch, 'steps': int(m.group(2)),
                    'train_loss': float(m.group(3)), 'vali_loss': float(m.group(4)),
                    'test_loss': float(m.group(5)),
                })
                continue
            if _SAVED_RE.search(line) and cur_epoch is not None:
                saved_epochs.append(cur_epoch)
                continue
            if _ES_STOP_RE.search(line):
                early_stop_epoch = cur_epoch
                continue
            m = _LR_RE.search(line)
            if m:
                lr_updates.append({'epoch': cur_epoch, 'lr': float(m.group(1))})
        proc.wait()

    print(f"\n訓練 log 已存 {log_path}")

    history_path = os.path.join(data_dir, 'training_history.json')
    existing = {}
    if os.path.exists(history_path):
        existing = json.load(open(history_path))
    existing[setting] = {
        'epochs': epochs,
        'saved_epochs': saved_epochs,          # EarlyStopping 實際存檔（= 目前最佳 vali_loss）的 epoch
        'best_epoch': saved_epochs[-1] if saved_epochs else None,  # checkpoint.pth 對應的 epoch
        'early_stop_epoch': early_stop_epoch,
        'lr_updates': lr_updates,
        'log_path': os.path.relpath(log_path, REPO_ROOT),
    }
    with open(history_path, 'w') as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)
    print(f"訓練歷史（逐 epoch loss）已存 {history_path}")

    return proc.returncode


if __name__ == '__main__':
    main()
