"""
機械化因果性測試：證明 features() / alt_features() 沒有任何欄位偷看未來。

方法：對原始 df 算一次特徵，再把第 N 列之後的 OHLCV 換成隨機值後重算一次，
若任何特徵是因果的（只用得到 <= t 的資料），兩次計算在 [0, N) 應該逐位元相同。
若有欄位在擾動後、N 之前的值也變了，代表該欄用到了未來資料。

用法：
    .venv/bin/python toolbywi/test_causality.py
"""
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from feature_scale import features, microstructure_features, alt_features, extra_features


def make_synthetic_ohlcv(n=800, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2025-01-01', periods=n, freq='5min')
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0, 1.0, n)
    low = close - rng.uniform(0, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)
    volume = rng.uniform(10, 100, n)
    quote_vol = volume * close
    taker_quote_vol = quote_vol * rng.uniform(0.3, 0.7, n)
    return pd.DataFrame({
        'date': dates, 'Open': open_, 'High': high, 'Low': low, 'Close': close,
        'Volume': volume, 'Quote_asset_volume': quote_vol,
        'Taker_buy_quote_asset_volume': taker_quote_vol,
    })


def test_features_causal():
    n = 800
    split = 500
    df_orig = make_synthetic_ohlcv(n=n, seed=1)

    df_perturb = df_orig.copy()
    rng = np.random.default_rng(999)
    m = n - split
    perturb_close = 100 + np.cumsum(rng.normal(0, 5.0, m))  # very different random walk
    df_perturb.loc[split:, 'Close'] = perturb_close
    df_perturb.loc[split:, 'High'] = perturb_close + rng.uniform(0, 5.0, m)
    df_perturb.loc[split:, 'Low'] = perturb_close - rng.uniform(0, 5.0, m)
    df_perturb.loc[split:, 'Open'] = perturb_close + rng.normal(0, 2.0, m)
    df_perturb.loc[split:, 'Volume'] = rng.uniform(500, 2000, m)
    df_perturb.loc[split:, 'Quote_asset_volume'] = df_perturb.loc[split:, 'Volume'] * perturb_close
    df_perturb.loc[split:, 'Taker_buy_quote_asset_volume'] = (
        df_perturb.loc[split:, 'Quote_asset_volume'] * rng.uniform(0.3, 0.7, m)
    )

    f_orig = features(df_orig)
    f_perturb = features(df_perturb)

    # features() does df.dropna().reset_index(drop=True) internally, so row N
    # in the input isn't necessarily row N in the output. Join on 'date' instead
    # of relying on positional alignment.
    common_dates = f_orig.loc[f_orig['date'] < df_orig['date'].iloc[split], 'date']
    a = f_orig[f_orig['date'].isin(common_dates)].set_index('date').sort_index()
    b = f_perturb[f_perturb['date'].isin(common_dates)].set_index('date').sort_index()

    assert list(a.columns) == list(b.columns), "欄位不一致，無法比較"

    bad_cols = []
    for col in a.columns:
        if not np.allclose(a[col].values, b[col].values, equal_nan=True, atol=1e-9):
            bad_cols.append(col)

    if bad_cols:
        raise AssertionError(
            f"因果性測試失敗！以下欄位在擾動未來資料後、過去的值也變了（洩漏未來資料）：{bad_cols}"
        )

    print(f"✅ features() 因果性測試通過（{len(a.columns)} 個欄位，{len(a)} 列比對，全部一致）")


def test_alt_features_causal():
    n = 800
    split = 500
    df_btc = make_synthetic_ohlcv(n=n, seed=2)
    df_eth = make_synthetic_ohlcv(n=n, seed=3)

    df_eth_perturb = df_eth.copy()
    rng = np.random.default_rng(998)
    m = n - split
    perturb_close = 50 + np.cumsum(rng.normal(0, 3.0, m))
    df_eth_perturb.loc[split:, 'Close'] = perturb_close
    df_eth_perturb.loc[split:, 'Volume'] = rng.uniform(500, 2000, m)

    merged_orig = alt_features(df_btc, {'ETH': df_eth})
    merged_perturb = alt_features(df_btc, {'ETH': df_eth_perturb})

    common_dates = merged_orig.loc[merged_orig['date'] < df_btc['date'].iloc[split], 'date']
    a = merged_orig[merged_orig['date'].isin(common_dates)].set_index('date').sort_index()
    b = merged_perturb[merged_perturb['date'].isin(common_dates)].set_index('date').sort_index()

    new_cols = [c for c in a.columns if c not in df_btc.columns]
    bad_cols = []
    for col in new_cols:
        if not np.allclose(a[col].values, b[col].values, equal_nan=True, atol=1e-9):
            bad_cols.append(col)

    if bad_cols:
        raise AssertionError(
            f"因果性測試失敗！alt_features 以下欄位偷看了未來資料：{bad_cols}"
        )

    print(f"✅ alt_features() 因果性測試通過（{len(new_cols)} 個新欄位，{len(a)} 列比對，全部一致）")


def make_synthetic_extra_sources(dates, seed=0):
    """
    合成 metrics / funding / premium / mark 四組來源，時間網格對齊 dates
    （funding 除外，它每 96 根才更新一次，模擬真實的 8 小時結算）。
    """
    rng = np.random.default_rng(seed)
    n = len(dates)

    metrics_df = pd.DataFrame({
        'date': dates,
        'sum_open_interest': 100000 + np.cumsum(rng.normal(0, 50, n)),
        'sum_open_interest_value': 6_000_000_000 + np.cumsum(rng.normal(0, 1e6, n)),
        'count_toptrader_long_short_ratio': 1.0 + rng.normal(0, 0.1, n),
        'sum_toptrader_long_short_ratio': 1.5 + rng.normal(0, 0.1, n),
        'count_long_short_ratio': 1.2 + rng.normal(0, 0.1, n),
        'sum_taker_long_short_vol_ratio': 1.0 + rng.normal(0, 0.3, n),
    })

    funding_idx = np.arange(0, n, 96)  # 每 96 根（8 小時）結算一次
    funding_df = pd.DataFrame({
        'date': dates[funding_idx],
        'funding_interval_hours': 8,
        'last_funding_rate': rng.normal(0, 0.0001, len(funding_idx)),
    })

    premium_df = pd.DataFrame({
        'date': dates,
        'Premium': rng.normal(0, 0.0005, n),
    })

    mark_df = pd.DataFrame({
        'date': dates,
        'Mark_Close': 100 + np.cumsum(rng.normal(0, 0.5, n)),
    })

    return metrics_df, funding_df, premium_df, mark_df


def test_microstructure_features_causal():
    """
    Round 4：microstructure_features() 的 6 個特徵都是逐根 pointwise 轉換
    （ATR_Pct_14 例外，重用 features() 已算好且已通過因果測試的 ATR_14），
    理論上零 warmup、擾動未來完全不該影響過去任何一列——用跟既有測試相同
    的「擾動尾端、比對前段」框架驗證。
    """
    n = 800
    split = 500
    df_orig = make_synthetic_ohlcv(n=n, seed=6)

    df_perturb = df_orig.copy()
    rng = np.random.default_rng(996)
    m = n - split
    perturb_close = 100 + np.cumsum(rng.normal(0, 5.0, m))
    df_perturb.loc[split:, 'Close'] = perturb_close
    df_perturb.loc[split:, 'High'] = perturb_close + rng.uniform(0, 5.0, m)
    df_perturb.loc[split:, 'Low'] = perturb_close - rng.uniform(0, 5.0, m)
    df_perturb.loc[split:, 'Open'] = perturb_close + rng.normal(0, 2.0, m)
    df_perturb.loc[split:, 'Volume'] = rng.uniform(500, 2000, m)
    df_perturb.loc[split:, 'Quote_asset_volume'] = df_perturb.loc[split:, 'Volume'] * perturb_close
    df_perturb.loc[split:, 'Taker_buy_quote_asset_volume'] = (
        df_perturb.loc[split:, 'Quote_asset_volume'] * rng.uniform(0.3, 0.7, m)
    )

    f_orig = microstructure_features(features(df_orig))
    f_perturb = microstructure_features(features(df_perturb))

    common_dates = f_orig.loc[f_orig['date'] < df_orig['date'].iloc[split], 'date']
    a = f_orig[f_orig['date'].isin(common_dates)].set_index('date').sort_index()
    b = f_perturb[f_perturb['date'].isin(common_dates)].set_index('date').sort_index()

    new_cols = ['HL_Range', 'OC_Ret', 'Up_Wick', 'Low_Wick', 'BarVWAP_Dev', 'ATR_Pct_14']
    bad_cols = []
    for col in new_cols:
        if not np.allclose(a[col].values, b[col].values, equal_nan=True, atol=1e-9):
            bad_cols.append(col)

    if bad_cols:
        raise AssertionError(
            f"因果性測試失敗！microstructure_features 以下欄位偷看了未來資料：{bad_cols}"
        )

    print(f"✅ microstructure_features() 因果性測試通過（{len(new_cols)} 個新欄位，{len(a)} 列比對，全部一致）")


def test_zero_volume_bar():
    """
    Round 4：1m/冷門時段可能出現 Volume==0 的 bar（BTC 在 5m 極少見，但不
    排除；alt 幣種更常見）。BarVWAP_Dev 除以 Volume，若沒有這層防護會產生
    inf/nan，讓下游 StandardScaler.fit() 直接炸掉——驗證零成交時的正確值
    是 0.0（無成交時 Open==High==Low==Close，VWAP 未定義，0 是唯一合理值，
    不是隨便挑的 sentinel），且輸出全程 finite，不留給下游猜測。
    """
    df = make_synthetic_ohlcv(n=200, seed=7)
    df_feat = features(df)  # dropna() 內部已丟掉 warmup 頭部（ROC_30 需要 30 根），
                             # 剩餘列重新 reset_index(drop=True)，故用相對位置而非絕對列號

    zero_idx = len(df_feat) // 2
    df_feat.loc[zero_idx, ['Open', 'High', 'Low', 'Close']] = 12345.6
    df_feat.loc[zero_idx, 'Volume'] = 0.0
    df_feat.loc[zero_idx, 'Quote_asset_volume'] = 0.0

    out = microstructure_features(df_feat)
    new_cols = ['HL_Range', 'OC_Ret', 'Up_Wick', 'Low_Wick', 'BarVWAP_Dev', 'ATR_Pct_14']

    assert np.isfinite(out.loc[zero_idx, new_cols].values.astype(float)).all(), (
        "Volume=0 的 bar 產生了非有限值（inf/nan）——BarVWAP_Dev 的除以零沒被正確攔截"
    )
    assert out.loc[zero_idx, 'BarVWAP_Dev'] == 0.0, (
        f"Volume=0 時 BarVWAP_Dev 應該是 0.0（VWAP 未定義的正確處理），"
        f"實際是 {out.loc[zero_idx, 'BarVWAP_Dev']}"
    )
    assert np.isfinite(out[new_cols].values.astype(float)).all(), "整張表都必須是有限值，不只是那一列"

    print("✅ 零成交 bar 測試通過（Volume=0 時 BarVWAP_Dev=0.0，全表無 inf/nan）")


def test_extra_features_causal():
    n = 800
    split = 500
    df_btc = make_synthetic_ohlcv(n=n, seed=4)
    dates = df_btc['date']

    metrics_df, funding_df, premium_df, mark_df = make_synthetic_extra_sources(dates, seed=5)

    rng = np.random.default_rng(997)
    m = n - split
    split_date = dates.iloc[split]

    metrics_p = metrics_df.copy()
    tail_mask_m = metrics_p['date'] >= split_date
    tm = tail_mask_m.sum()
    metrics_p.loc[tail_mask_m, 'sum_open_interest'] = rng.uniform(5e4, 5e5, tm)
    metrics_p.loc[tail_mask_m, 'sum_open_interest_value'] = rng.uniform(1e9, 1e10, tm)
    metrics_p.loc[tail_mask_m, 'count_toptrader_long_short_ratio'] = rng.uniform(0.5, 3, tm)
    metrics_p.loc[tail_mask_m, 'sum_toptrader_long_short_ratio'] = rng.uniform(0.5, 3, tm)
    metrics_p.loc[tail_mask_m, 'count_long_short_ratio'] = rng.uniform(0.5, 3, tm)
    metrics_p.loc[tail_mask_m, 'sum_taker_long_short_vol_ratio'] = rng.uniform(0.1, 5, tm)

    funding_p = funding_df.copy()
    tail_mask_f = funding_p['date'] >= split_date
    tf = tail_mask_f.sum()
    funding_p.loc[tail_mask_f, 'last_funding_rate'] = rng.uniform(-0.01, 0.01, tf)

    premium_p = premium_df.copy()
    tail_mask_p = premium_p['date'] >= split_date
    tp = tail_mask_p.sum()
    premium_p.loc[tail_mask_p, 'Premium'] = rng.uniform(-0.01, 0.01, tp)

    mark_p = mark_df.copy()
    tail_mask_mk = mark_p['date'] >= split_date
    tmk = tail_mask_mk.sum()
    mark_p.loc[tail_mask_mk, 'Mark_Close'] = rng.uniform(1000, 5000, tmk)

    df_feat = features(df_btc)

    orig = extra_features(df_feat, metrics_df, funding_df, premium_df, mark_df)
    perturb = extra_features(df_feat, metrics_p, funding_p, premium_p, mark_p)

    common_dates = orig.loc[orig['date'] < split_date, 'date']
    a = orig[orig['date'].isin(common_dates)].set_index('date').sort_index()
    b = perturb[perturb['date'].isin(common_dates)].set_index('date').sort_index()

    assert len(a) > 0, "分割點之前沒有剩餘資料可比對，測試設計有問題"

    new_cols = [c for c in a.columns if c not in df_feat.columns]
    bad_cols = []
    for col in new_cols:
        if not np.allclose(a[col].values, b[col].values, equal_nan=True, atol=1e-9):
            bad_cols.append(col)

    if bad_cols:
        raise AssertionError(
            f"因果性測試失敗！extra_features 以下欄位偷看了未來資料：{bad_cols}"
        )

    print(f"✅ extra_features() 因果性測試通過（{len(new_cols)} 個新欄位，{len(a)} 列比對，全部一致）")


if __name__ == '__main__':
    test_features_causal()
    test_alt_features_causal()
    test_microstructure_features_causal()
    test_zero_volume_bar()
    test_extra_features_causal()
    print("✅ 全部因果性測試通過")
