#!/usr/bin/env python3
"""
畫 loss-epoch 圖 —— 讀 train_event.py（Round 4 起預設開啟 tee）寫的
{data_dir}/training_history.json，把某一個 setting 的逐 epoch
train/vali/test loss 畫成一張圖，標出：
  - EarlyStopping 實際存檔（= checkpoint.pth 對應）的 epoch
  - early stop 觸發的 epoch（若有）

只讀資料，不影響任何訓練/評估流程；測試 loss 只用來畫圖觀察，不作為任何
模型選擇依據（EarlyStopping 只看 vali_loss，選擇邏輯不變，見
exp/exp_long_term_forecasting.py 的 train()）。

用法：
  .venv/bin/python plot_training_curve.py --data_dir ./dataset/event30m_v4_sel \
      --setting long_term_forecast_..._R4_pauc_0
  .venv/bin/python plot_training_curve.py --data_dir ./dataset/event30m_v4_sel   # 不給 --setting 用 meta.json 頂層 setting
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data_dir', default='./dataset/event30m_v4_sel')
    p.add_argument('--setting', default=None,
                    help='不給則用 meta.json 頂層 setting（train_event.py 最近一次訓練的模型）')
    p.add_argument('--out', default=None,
                    help='輸出圖檔路徑，不給則存到 {data_dir}/training_curve_{setting短版}.png')
    return p.parse_args()


def main():
    args = parse_args()

    history_path = os.path.join(args.data_dir, 'training_history.json')
    if not os.path.exists(history_path):
        raise RuntimeError(
            f"找不到 {history_path}。只有 train_event.py 在 tee 模式（預設，見 --no_tee）"
            f"下訓練過的模型才會有這份逐 epoch 紀錄；Round 1-3 的舊 checkpoint 沒有。"
        )
    history = json.load(open(history_path))

    setting = args.setting
    if setting is None:
        meta = json.load(open(os.path.join(args.data_dir, 'meta.json')))
        setting = meta['setting']
    if setting not in history:
        raise RuntimeError(
            f"training_history.json 裡沒有 setting={setting}。"
            f"可用的 key：{list(history.keys())}"
        )

    rec = history[setting]
    epochs_data = rec['epochs']
    if not epochs_data:
        raise RuntimeError(f"setting={setting} 的 epochs 是空的，訓練可能在第一個 epoch 前就中斷了。")

    ep = [d['epoch'] for d in epochs_data]
    train_loss = [d['train_loss'] for d in epochs_data]
    vali_loss = [d['vali_loss'] for d in epochs_data]
    test_loss = [d['test_loss'] for d in epochs_data]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(ep, train_loss, marker='o', markersize=4, label='Train Loss', color='#68787C')
    ax.plot(ep, vali_loss, marker='o', markersize=4, label='Vali Loss', color='#AC4A2C')
    ax.plot(ep, test_loss, marker='o', markersize=4, label='Test Loss', color='#2E6B78',
            linestyle='--')

    best_epoch = rec.get('best_epoch')
    if best_epoch is not None and best_epoch in ep:
        i = ep.index(best_epoch)
        ax.scatter([best_epoch], [vali_loss[i]], s=140, facecolors='none',
                    edgecolors='#AC4A2C', linewidths=2, zorder=5)
        ax.annotate(f'checkpoint (epoch {best_epoch})', xy=(best_epoch, vali_loss[i]),
                    xytext=(8, 12), textcoords='offset points', fontsize=9, color='#AC4A2C')

    early_stop_epoch = rec.get('early_stop_epoch')
    if early_stop_epoch is not None:
        ax.axvline(early_stop_epoch, color='#68787C', linestyle=':', linewidth=1.2)
        ax.annotate(f'early stop (epoch {early_stop_epoch})',
                    xy=(early_stop_epoch, max(train_loss + vali_loss + test_loss)),
                    xytext=(4, -4), textcoords='offset points', fontsize=9, color='#68787C',
                    ha='left', va='top')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Loss vs Epoch\n{setting}', fontsize=10)
    ax.set_xticks(ep)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()

    out_path = args.out
    if out_path is None:
        short = setting.split('_R4_')[-1] if '_R4_' in setting else setting[-24:]
        out_path = os.path.join(args.data_dir, f'training_curve_{short}.png')
    fig.savefig(out_path, dpi=150)
    print(f"已存 {out_path}")

    print(f"\nsetting: {setting}")
    print(f"epochs: {ep[0]}..{ep[-1]}（共 {len(ep)} 個）")
    print(f"checkpoint（EarlyStopping 實際存檔）epoch: {best_epoch}")
    print(f"early stop epoch: {early_stop_epoch}")
    if rec.get('lr_updates'):
        print(f"學習率調整: {rec['lr_updates']}")


if __name__ == '__main__':
    main()
