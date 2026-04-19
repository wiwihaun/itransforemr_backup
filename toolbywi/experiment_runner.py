"""
experiment_runner.py — One-call pipeline orchestrator for iTransformer 5min experiments.

Usage in Colab:
    from experiment_runner import run_experiment, display_results
    results = run_experiment(config)
    display_results(results)
"""

import os
import sys
import subprocess
import shutil
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.special import expit

# ── Auto-detect project root (works whether called from /content/itransformer_5min or toolbywi/) ──
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.dirname(_THIS_DIR)   # itransformer_5min/
_TOOLBYWI  = _THIS_DIR                    # itransformer_5min/toolbywi/

for _p in [_PROJ_ROOT, _TOOLBYWI]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ───────────────────────────────────────────────────────────────────────────────
# DEFAULT CONFIG
# ───────────────────────────────────────────────────────────────────────────────
DEFAULT_CONFIG = {
    # === 資料設定 ===
    'symbol':        'BTCUSDT',
    'alt_symbols':   ['ETHUSDT', 'BNBUSDT', 'SOLUSDT'],   # 空 list = 不加 alt
    'start_year':    2025,  'start_month':  6,
    'end_year':      2026,  'end_month':    3,

    # === 目標標籤 ===
    'direction':     'long',   # 'long' or 'short'
    'tp_pct':        0.02,     # 止盈（同時作為回測 take_profit_pct）
    'sl_pct':        0.01,     # 止損（同時作為回測 stop_loss_pct）
    'lookahead':     288,      # 未來幾根 K 棒（288×5min = 24h）

    # === 模型架構 ===
    'seq_len':       576,      # 2 天
    'd_model':       256,
    'd_ff':          512,
    'n_heads':       8,
    'e_layers':      2,
    'dropout':       0.2,

    # === 訓練設定 ===
    'batch_size':    64,
    'learning_rate': 0.0001,
    'lradj':         'type3',
    'patience':      15,
    'train_epochs':  100,

    # === 回測設定（TP/SL 自動沿用訓練設定，不需重複填）===
    'prob_threshold':   0.60,
    'position_size':    3,       # 每次下注佔資本的倍率（Kelly-like）
    'max_hold_days':    1,
    'cooldown_hours':   4,
    'initial_capital':  1000.0,
}


# ───────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ───────────────────────────────────────────────────────────────────────────────

def _merge_config(user_config):
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(user_config)
    return cfg


def _download_data(cfg):
    from download_binance_monthly_batch import binance_load_5min

    print(f"📥 下載 {cfg['symbol']} 5min 資料...")
    df_btc = binance_load_5min(
        symbol=cfg['symbol'],
        start_year=cfg['start_year'], start_month=cfg['start_month'],
        end_year=cfg['end_year'],   end_month=cfg['end_month'],
    )
    df_btc = df_btc.sort_values('date').reset_index(drop=True)

    alt_dfs = {}
    for sym in cfg.get('alt_symbols', []):
        print(f"📥 下載 {sym} 5min 資料...")
        df_alt = binance_load_5min(
            symbol=sym,
            start_year=cfg['start_year'], start_month=cfg['start_month'],
            end_year=cfg['end_year'],   end_month=cfg['end_month'],
        )
        alt_dfs[sym.replace('USDT', '')] = df_alt

    print(f"✅ 下載完成 — BTC:{len(df_btc)} rows" +
          ''.join(f"  {k}:{len(v)}" for k, v in alt_dfs.items()))
    return df_btc, alt_dfs


def _build_features(df_btc, alt_dfs, cfg):
    from feature_scale import features, scaler, alt_features
    from target_calulate import target_long, target_short

    # 1. Target label
    if cfg['direction'] == 'long':
        df_btc['target'] = target_long(
            df_btc, lookahead=cfg['lookahead'],
            tp_pct=cfg['tp_pct'], sl_pct=cfg['sl_pct'])
    else:
        df_btc['target'] = target_short(
            df_btc, lookahead=cfg['lookahead'],
            tp_pct=cfg['tp_pct'], sl_pct=cfg['sl_pct'])

    n0 = (df_btc['target'] == 0).sum()
    n1 = (df_btc['target'] == 1).sum()
    focal_alpha = round(n0 / (n0 + n1), 4)
    print(f"🎯 Target: {n0} zeros / {n1} ones | focal_alpha={focal_alpha:.3f}")

    # 2. BTC technical indicators
    df_btc = features(df_btc)

    # 3. Alt-coin features (optional)
    if alt_dfs:
        df_btc = alt_features(df_btc, alt_dfs)

    # 4. Z-Score scale (fit only on train 70%)
    df_btc = scaler(df_btc, train_ratio=0.7)

    # 5. Target must be last column
    cols = [c for c in df_btc.columns if c != 'target']
    df_btc = df_btc[cols + ['target']]

    # 6. Drop raw OHLCV (not model input, already saved separately)
    drop_raw = ['Open', 'High', 'Low', 'Quote_asset_volume', 'Taker_buy_quote_asset_volume']
    df_btc = df_btc.drop(columns=[c for c in drop_raw if c in df_btc.columns])

    feature_cols = [c for c in df_btc.columns if c not in ['date', 'target']]
    enc_in = len(feature_cols) + 1   # +1 for target column
    print(f"✅ 特徵數: {len(feature_cols)}  |  enc_in={enc_in}")
    return df_btc, feature_cols, enc_in, focal_alpha


def _save_csvs(df_btc_origin, df_btc, dataset_dir):
    os.makedirs(dataset_dir, exist_ok=True)
    raw_path  = os.path.join(dataset_dir, 'raw_ohlcv.csv')
    feat_path = os.path.join(dataset_dir, 'stock_features.csv')
    df_btc_origin.to_csv(raw_path,  index=False)
    df_btc.to_csv(feat_path, index=False)
    print(f"💾 raw_ohlcv.csv ({len(df_btc_origin)} rows) + stock_features.csv ({df_btc.shape}) saved")
    return raw_path, feat_path


def _train_model(cfg, enc_in, focal_alpha, proj_root, exp_id):
    """Call run.py via subprocess and capture stdout."""
    cmd = [
        'python', '-u', os.path.join(proj_root, 'run.py'),
        '--task_name',    'long_term_forecast',
        '--is_training',  '1',
        '--root_path',    os.path.join(proj_root, 'dataset', 'stock') + os.sep,
        '--data_path',    'stock_features.csv',
        '--model_id',     exp_id,
        '--model',        'iTransformer',
        '--data',         'custom',
        '--features',     'MS',
        '--target',       'target',
        '--freq',         't',
        '--seq_len',      str(cfg['seq_len']),
        '--label_len',    '48',
        '--pred_len',     '1',
        '--enc_in',       str(enc_in),
        '--dec_in',       str(enc_in),
        '--c_out',        '1',
        '--e_layers',     str(cfg['e_layers']),
        '--d_model',      str(cfg['d_model']),
        '--d_ff',         str(cfg['d_ff']),
        '--n_heads',      str(cfg['n_heads']),
        '--dropout',      str(cfg['dropout']),
        '--batch_size',   str(cfg['batch_size']),
        '--patience',     str(cfg['patience']),
        '--train_epochs', str(cfg['train_epochs']),
        '--learning_rate',str(cfg['learning_rate']),
        '--lradj',        cfg['lradj'],
        '--focal_alpha',  str(focal_alpha),
        '--des',          exp_id,
        '--itr',          '1',
    ]
    print(f"\n🚀 開始訓練 [{exp_id}] ...")
    print(f"   enc_in={enc_in}  focal_alpha={focal_alpha:.3f}  seq_len={cfg['seq_len']}")
    print(f"   d_model={cfg['d_model']} d_ff={cfg['d_ff']} batch={cfg['batch_size']} lr={cfg['learning_rate']}\n")

    log_lines = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, cwd=proj_root)
    for line in proc.stdout:
        print(line, end='')
        log_lines.append(line)
    proc.wait()
    return log_lines, proc.returncode


def _parse_training_log(log_lines):
    """Extract epoch count and best val_loss from training log."""
    import re
    epochs_done = 0
    best_val    = float('inf')
    for line in log_lines:
        m = re.search(r'Epoch:\s*(\d+).*?Vali Loss:\s*([\d.]+)', line)
        if m:
            epochs_done = int(m.group(1))
            val = float(m.group(2))
            if val < best_val:
                best_val = val
    return epochs_done, best_val


def _run_backtest(cfg, feat_csv, raw_csv, result_dir):
    from backtest import load_data, run_backtest as _run_bt, print_stats, plot_results

    df_test, pred_probs = load_data(
        csv_path=feat_csv,
        prob_threshold=cfg['prob_threshold'],
        result_dir=result_dir,
        price_csv_path=raw_csv,
    )
    df_test, buy_r, tp_r, sl_r, te_r = _run_bt(
        df_test,
        initial_capital=cfg['initial_capital'],
        position_size=cfg['position_size'],
        take_profit_pct=cfg['tp_pct'],
        stop_loss_pct=cfg['sl_pct'],
        max_hold_days=cfg['max_hold_days'],
        cooldown_hours=cfg['cooldown_hours'],
    )
    return df_test, buy_r, tp_r, sl_r, te_r, pred_probs


def _compute_classification(result_dir, threshold):
    """Compute precision, recall, F1 from pred.npy + true.npy."""
    try:
        folders = [os.path.join(result_dir, f) for f in os.listdir(result_dir)
                   if os.path.isdir(os.path.join(result_dir, f))]
        latest = max(folders, key=os.path.getmtime)
        pred_logits = np.load(os.path.join(latest, 'pred.npy')).flatten()
        true_vals   = np.load(os.path.join(latest, 'true.npy')).flatten()

        pred_bin = (expit(pred_logits) > threshold).astype(int)
        true_bin = (true_vals > 0).astype(int)

        tp_ = ((pred_bin == 1) & (true_bin == 1)).sum()
        fp_ = ((pred_bin == 1) & (true_bin == 0)).sum()
        fn_ = ((pred_bin == 0) & (true_bin == 1)).sum()
        tn_ = ((pred_bin == 0) & (true_bin == 0)).sum()

        accuracy  = (tp_ + tn_) / len(true_bin)
        precision = tp_ / (tp_ + fp_) if (tp_ + fp_) > 0 else 0
        recall    = tp_ / (tp_ + fn_) if (tp_ + fn_) > 0 else 0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return dict(accuracy=accuracy, precision=precision,
                    recall=recall, f1=f1,
                    tp=int(tp_), fp=int(fp_), fn=int(fn_), tn=int(tn_))
    except Exception as e:
        print(f"⚠️  分類指標計算失敗: {e}")
        return {}


# ───────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ───────────────────────────────────────────────────────────────────────────────

def run_experiment(user_config: dict, exp_id: str = None) -> dict:
    """
    One-call pipeline: download → features → target → scale → train → backtest.

    Parameters
    ----------
    user_config : dict  — see DEFAULT_CONFIG for all keys
    exp_id      : str   — experiment name (auto-generated if None)

    Returns
    -------
    dict with keys:
        config, feature_cols, enc_in, focal_alpha,
        epochs_done, best_val_loss, return_code,
        df_test, buy_records, tp_records, sl_records, te_records, pred_probs,
        roi, total_trades, win_rate, max_drawdown,
        classification (accuracy/precision/recall/f1),
        proj_root, result_dir, raw_csv, feat_csv
    """
    cfg = _merge_config(user_config)

    if exp_id is None:
        ts = time.strftime('%m%d_%H%M')
        exp_id = f"exp_{cfg['direction']}_tp{int(cfg['tp_pct']*100)}_sl{int(cfg['sl_pct']*100)}_{ts}"

    proj_root   = _PROJ_ROOT
    dataset_dir = os.path.join(proj_root, 'dataset', 'stock')
    result_dir  = os.path.join(proj_root, 'results')

    # 1. Download
    df_btc_raw, alt_dfs = _download_data(cfg)
    df_btc_origin = df_btc_raw.copy()

    # 2. Build features
    df_btc, feature_cols, enc_in, focal_alpha = _build_features(df_btc_raw, alt_dfs, cfg)

    # 3. Save CSVs
    raw_csv, feat_csv = _save_csvs(df_btc_origin, df_btc, dataset_dir)

    # 4. Clear old results
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)
    os.makedirs(result_dir, exist_ok=True)

    # 5. Train
    log_lines, return_code = _train_model(cfg, enc_in, focal_alpha, proj_root, exp_id)
    epochs_done, best_val  = _parse_training_log(log_lines)

    if return_code != 0:
        print("❌ 訓練失敗！請檢查上方錯誤訊息。")
        return {'return_code': return_code, 'config': cfg}

    # 6. Classification metrics
    clf_metrics = _compute_classification(result_dir, cfg['prob_threshold'])

    # 7. Backtest
    df_test, buy_r, tp_r, sl_r, te_r, pred_probs = _run_backtest(
        cfg, feat_csv, raw_csv, result_dir)

    # 8. Summary stats
    final_eq    = df_test['Equity'].iloc[-1]
    roi         = (final_eq - cfg['initial_capital']) / cfg['initial_capital']
    total_trades= len(buy_r)
    win_rate    = len(tp_r) / total_trades if total_trades > 0 else 0
    peak        = df_test['Equity'].cummax()
    max_dd      = ((df_test['Equity'] - peak) / peak).min()

    results = dict(
        config=cfg, exp_id=exp_id,
        feature_cols=feature_cols, enc_in=enc_in, focal_alpha=focal_alpha,
        epochs_done=epochs_done, best_val_loss=best_val, return_code=return_code,
        df_test=df_test,
        buy_records=buy_r, tp_records=tp_r, sl_records=sl_r, te_records=te_r,
        pred_probs=pred_probs,
        roi=roi, total_trades=total_trades, win_rate=win_rate, max_drawdown=max_dd,
        classification=clf_metrics,
        proj_root=proj_root, result_dir=result_dir,
        raw_csv=raw_csv, feat_csv=feat_csv,
    )

    # Quick summary
    print(f"\n{'='*50}")
    print(f"✅ 實驗完成 [{exp_id}]")
    print(f"   特徵數={len(feature_cols)}  enc_in={enc_in}  epochs={epochs_done}")
    print(f"   val_loss={best_val:.4f}  ROI={roi:.2%}  勝率={win_rate:.2%}")
    if clf_metrics:
        print(f"   Precision={clf_metrics.get('precision',0):.2%}  "
              f"Recall={clf_metrics.get('recall',0):.2%}  "
              f"F1={clf_metrics.get('f1',0):.2%}")
    print(f"{'='*50}")
    return results


def display_results(results: dict):
    """Print summary stats + plot training loss curve + backtest equity curve."""
    if not results or results.get('return_code', 1) != 0:
        print("❌ 無可顯示結果（訓練失敗）")
        return

    cfg    = results['config']
    clf    = results.get('classification', {})

    # ── Text summary ──
    print("\n" + "="*55)
    print("📊  實驗結果摘要")
    print("="*55)
    print(f"實驗 ID     : {results['exp_id']}")
    print(f"方向        : {cfg['direction'].upper()}  |  "
          f"TP={cfg['tp_pct']*100:.1f}%  SL={cfg['sl_pct']*100:.1f}%")
    print(f"特徵數       : {len(results['feature_cols'])}  |  enc_in={results['enc_in']}")
    print(f"focal_alpha : {results['focal_alpha']:.3f}")
    print(f"訓練 Epochs  : {results['epochs_done']}  |  最佳 val_loss={results['best_val_loss']:.4f}")
    print("-"*55)
    if clf:
        print(f"Accuracy    : {clf.get('accuracy',0):.2%}")
        print(f"Precision   : {clf.get('precision',0):.2%}  (預測=1 中真正是1的比例)")
        print(f"Recall      : {clf.get('recall',0):.2%}  (真實=1 中被抓到的比例)")
        print(f"F1 Score    : {clf.get('f1',0):.2%}")
    print("-"*55)
    total_t = results['total_trades']
    tp_n    = len(results['tp_records'])
    sl_n    = len(results['sl_records'])
    te_n    = len(results['te_records'])
    print(f"ROI         : {results['roi']:.2%}")
    print(f"最大回撤     : {results['max_drawdown']:.2%}")
    print(f"總交易數     : {total_t}")
    if total_t > 0:
        print(f"  止盈       : {tp_n} ({tp_n/total_t:.1%})")
        print(f"  止損       : {sl_n} ({sl_n/total_t:.1%})")
        print(f"  時間出場   : {te_n} ({te_n/total_t:.1%})")
        print(f"純停利勝率  : {results['win_rate']:.2%}")
    print("="*55)

    # ── Charts ──
    df = results['df_test']
    buy_r, tp_r, sl_r, te_r = (results['buy_records'], results['tp_records'],
                                results['sl_records'], results['te_records'])

    import seaborn as sns
    sns.set_theme(style='whitegrid')
    fig, axes = plt.subplots(2, 1, figsize=(16, 10),
                             gridspec_kw={'height_ratios': [2, 1]}, sharex=True)
    fig.suptitle(f"Backtest — {results['exp_id']} | ROI={results['roi']:.2%}  勝率={results['win_rate']:.2%}",
                 fontsize=14, fontweight='bold')

    axes[0].plot(df['date'], df['Close'], color='gray', alpha=0.5, label='BTC Close')
    if buy_r:
        bd, bp = zip(*buy_r)
        axes[0].scatter(bd, bp, marker='^', color='green', s=100, label='Buy', zorder=5)
    if tp_r:
        td, tp_, _ = zip(*tp_r)
        axes[0].scatter(td, tp_, marker='*', color='gold', s=200,
                        edgecolor='k', label=f'TP +{cfg["tp_pct"]*100:.0f}%', zorder=6)
    if sl_r:
        sd, sp, _ = zip(*sl_r)
        axes[0].scatter(sd, sp, marker='X', color='magenta', s=150,
                        edgecolor='k', label=f'SL -{cfg["sl_pct"]*100:.0f}%', zorder=6)
    if te_r:
        ed, ep, _ = zip(*te_r)
        axes[0].scatter(ed, ep, marker='v', color='saddlebrown', s=80, label='Time Exit', zorder=5)
    axes[0].set_ylabel('Price'); axes[0].legend(loc='upper left', fontsize=10)

    axes[1].plot(df['date'], df['Equity'], color='royalblue', lw=2, label='Equity')
    axes[1].axhline(cfg['initial_capital'], color='red', ls='--', alpha=0.6, label='Initial')
    axes[1].set_ylabel('Capital'); axes[1].set_xlabel('Date')
    axes[1].legend(loc='upper left', fontsize=10)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def save_best_model(results: dict, name: str = 'best_model'):
    """Copy latest checkpoint.pth to Google Drive or current dir."""
    proj_root = results.get('proj_root', _PROJ_ROOT)
    ckpt_dir  = os.path.join(proj_root, 'checkpoints')
    if not os.path.exists(ckpt_dir):
        print("❌ checkpoints/ not found")
        return

    folders = [os.path.join(ckpt_dir, f) for f in os.listdir(ckpt_dir)
               if os.path.isdir(os.path.join(ckpt_dir, f))]
    if not folders:
        print("❌ No checkpoint folders")
        return

    latest    = max(folders, key=os.path.getmtime)
    ckpt_file = os.path.join(latest, 'checkpoint.pth')

    # Try Google Drive first
    drive_dir = '/content/drive/MyDrive/Quant_Models/iTransformer_Binary/'
    if os.path.exists('/content/drive/MyDrive/'):
        os.makedirs(drive_dir, exist_ok=True)
        dest = os.path.join(drive_dir, f'{name}.pth')
    else:
        dest = os.path.join(proj_root, f'{name}.pth')

    import shutil
    shutil.copy(ckpt_file, dest)
    print(f"✅ 模型已儲存至: {dest}")
    return dest
