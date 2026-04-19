import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def features(df):
    """
    全新 18 特徵組（基於五篇高引用論文）— 5min K 線版
    來源：arXiv 2311.14759 / 2410.06935 / 2511.00665 / MDPI TFT / GitHub baruch1192
    分類：原始價量(2) + 趨勢(4) + 動量(5) + 波動(3) + 量能(4)
    5min 調整：EMA_6→EMA_72 (6h)，ADX_13→ADX_156 (13h)，其餘不變
    """
    print("🛠️ 開始計算全新 18 特徵組 [5min 版]...")
    df = df.copy()

    # ── 共用 True Range 計算（供 ADX、ATR 使用）─────────
    high_low  = df['High'] - df['Low']
    high_prev = (df['High'] - df['Close'].shift(1)).abs()
    low_prev  = (df['Low']  - df['Close'].shift(1)).abs()
    tr = pd.concat([high_low, high_prev, low_prev], axis=1).max(axis=1)

    # ==========================================
    # 📌 1. 趨勢類（4 個）
    # 5min 版：span × 12 以維持等效時間（1h EMA_6 = 6h = 5min EMA_72）
    # ==========================================
    df['EMA_72']    = df['Close'].ewm(span=72,  adjust=False).mean()  # 6h（等效 1h EMA_6）
    df['EMA_95']    = df['Close'].ewm(span=95, adjust=False).mean()   # ~8h（原 1h EMA_95，仍合理）
    df['EMA_Cross'] = df['EMA_72'] - df['EMA_95']   # 黃金/死亡交叉強度

    # ADX(156)：13h 等效（1h ADX_13 × 12）
    plus_dm  = (df['High'] - df['High'].shift(1)).clip(lower=0)
    minus_dm = (df['Low'].shift(1) - df['Low']).clip(lower=0)
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr156   = tr.ewm(span=156, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(span=156, adjust=False).mean()  / (atr156 + 1e-8)
    minus_di = 100 * minus_dm.ewm(span=156, adjust=False).mean() / (atr156 + 1e-8)
    dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-8)
    df['ADX_156'] = dx.ewm(span=156, adjust=False).mean()

    # ==========================================
    # 📌 2. 動量類（5 個）
    # 論文：arXiv 2410.06935 Chi-Squared Top3：RSI30、MACD、MOM30
    #       arXiv 2511.00665：MACD(17, 21, 15)
    # ==========================================
    # RSI(14)
    d14   = df['Close'].diff()
    g14   = d14.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    l14   = (-d14.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    df['RSI_14'] = 100 - (100 / (1 + g14 / (l14 + 1e-8)))

    # RSI(30)
    d30   = df['Close'].diff()
    g30   = d30.clip(lower=0).ewm(alpha=1/30, adjust=False).mean()
    l30   = (-d30.clip(upper=0)).ewm(alpha=1/30, adjust=False).mean()
    df['RSI_30'] = 100 - (100 / (1 + g30 / (l30 + 1e-8)))

    # MACD(17, 21) + Signal(15)
    ema17 = df['Close'].ewm(span=17, adjust=False).mean()
    ema21 = df['Close'].ewm(span=21, adjust=False).mean()
    df['MACD']        = ema17 - ema21
    df['MACD_Signal'] = df['MACD'].ewm(span=15, adjust=False).mean()

    # MOM(30)：動能
    df['MOM_30'] = df['Close'] - df['Close'].shift(30)

    # ==========================================
    # 📌 3. 波動類（3 個）
    # ==========================================
    df['ATR_14']  = tr.rolling(14).mean()

    sma20  = df['Close'].rolling(20).mean()
    std20  = df['Close'].rolling(20).std()
    upper  = sma20 + 2 * std20
    lower  = sma20 - 2 * std20
    df['BB_Width'] = (upper - lower) / (sma20 + 1e-8) * 100
    df['BB_PB']    = (df['Close'] - lower) / (upper - lower + 1e-8)

    # ==========================================
    # 📌 4. 量能類（4 個）
    # 論文：MDPI TFT — 主動買量最具預測力
    #       GitHub baruch1192 — CMF 取代 MFI
    # ==========================================
    # OBV
    obv_vals = np.where(df['Close'] > df['Close'].shift(1),  df['Volume'],
               np.where(df['Close'] < df['Close'].shift(1), -df['Volume'], 0))
    df['OBV'] = pd.Series(obv_vals, index=df.index).cumsum()

    # CMF(14)：柴金資金流量（比 MFI 更純粹）
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) \
          / (df['High'] - df['Low'] + 1e-8)
    df['CMF_14'] = (clv * df['Volume']).rolling(14).sum() \
                   / (df['Volume'].rolling(14).sum() + 1e-8)

    # Volume_Ratio：相對成交量（過濾異常放量）
    df['Volume_Ratio'] = df['Volume'] / (df['Volume'].rolling(20).mean() + 1e-8)

    # Taker_Ratio：主動買方比例（Binance 獨有，反映多空力道）
    if 'Taker_buy_quote_asset_volume' in df.columns \
            and 'Quote_asset_volume' in df.columns:
        df['Taker_Ratio'] = df['Taker_buy_quote_asset_volume'] \
                            / (df['Quote_asset_volume'] + 1e-8)

    # ==========================================
    # 📌 5. 新增 BTC 自身指標（7 個）
    # Stochastic(14,3)、Williams %R(14)、CCI(20)、ROC(10,30)、VWAP Deviation
    # ==========================================

    # Stochastic %K/%D (14,3)
    low14  = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = (df['Close'] - low14) / (high14 - low14 + 1e-9) * 100
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # Williams %R (14)
    df['Williams_R'] = (high14 - df['Close']) / (high14 - low14 + 1e-9) * -100

    # CCI (20) — Commodity Channel Index
    tp = (df['High'] + df['Low'] + df['Close']) / 3
    sma20_cci = tp.rolling(20).mean()
    mad20     = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['CCI_20'] = (tp - sma20_cci) / (0.015 * mad20 + 1e-9)

    # Rate of Change (10, 30)
    df['ROC_10'] = df['Close'].pct_change(10) * 100
    df['ROC_30'] = df['Close'].pct_change(30) * 100

    # VWAP Deviation — 每日重置，衡量離日內均價的偏差
    if 'date' in df.columns:
        _date_str = df['date'].astype(str).str[:10]
    else:
        _date_str = pd.Series(df.index, index=df.index).astype(str)
    _pv  = (df['Close'] * df['Volume']).groupby(_date_str).cumsum()
    _vol = df['Volume'].groupby(_date_str).cumsum()
    _vwap = _pv / (_vol + 1e-9)
    df['VWAP_Dev'] = df['Close'] / (_vwap + 1e-9) - 1

    # ==========================================
    # 清除 rolling 產生的初期空值
    df = df.dropna().reset_index(drop=True)

    feature_count = len([c for c in df.columns if c not in ['date', 'target']])
    print(f"✅ 完成！共 {feature_count} 個特徵（不含 date、target）。")
    return df



def alt_features(btc_df, alt_dfs):
    """
    為 BTC DataFrame 附加多幣種特徵（ETH / BNB / SOL 等）。
    每個幣種提取 3 個特徵：Return（5min 漲跌幅）、RSI_14、Volume_Ratio。
    透過 date 欄左連接，ffill 補缺後 dropna。

    Parameters
    ----------
    btc_df   : pd.DataFrame，包含 'date' 欄
    alt_dfs  : dict，例如 {'ETH': df_eth, 'BNB': df_bnb, 'SOL': df_sol}

    Returns
    -------
    pd.DataFrame，原 btc_df 欄位 + 每幣種 3 個新欄位
    """
    print("🛠️ 開始合併多幣種特徵...")
    merged = btc_df.copy()

    for sym, df_alt in alt_dfs.items():
        df_a = df_alt[['date', 'Close', 'Volume']].copy()
        df_a['date'] = pd.to_datetime(df_a['date'])
        df_a = df_a.sort_values('date').reset_index(drop=True)

        # Return (1-bar pct change)
        df_a[f'{sym}_Return'] = df_a['Close'].pct_change() * 100

        # RSI_14 (Wilder EWM)
        _d   = df_a['Close'].diff()
        _g   = _d.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        _l   = (-_d.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        df_a[f'{sym}_RSI14'] = 100 - (100 / (1 + _g / (_l + 1e-8)))

        # Volume_Ratio (relative to 20-bar rolling mean)
        df_a[f'{sym}_VolRatio'] = df_a['Volume'] / (df_a['Volume'].rolling(20).mean() + 1e-8)

        # Keep only date + 3 feature columns
        cols = ['date', f'{sym}_Return', f'{sym}_RSI14', f'{sym}_VolRatio']
        df_a = df_a[cols].dropna()

        # Merge on date (left join)
        merged['date'] = pd.to_datetime(merged['date'])
        merged = merged.merge(df_a, on='date', how='left')
        merged[[f'{sym}_Return', f'{sym}_RSI14', f'{sym}_VolRatio']] = \
            merged[[f'{sym}_Return', f'{sym}_RSI14', f'{sym}_VolRatio']].ffill()

    merged = merged.dropna().reset_index(drop=True)
    new_cols = [c for c in merged.columns if c not in btc_df.columns]
    print(f"✅ 多幣種合併完成！新增 {len(new_cols)} 個特徵：{new_cols}")
    return merged


def scaler(df, train_ratio=0.7):
    print("🛠️ 開始執行統一 Z-Score 縮放 (防未來洩漏機制啟動)...")
    df_scaled = df.copy()

    # 計算訓練集的截止邊界
    train_end_idx = int(len(df_scaled) * train_ratio)

    # ==========================================
    # 📌 統一對所有數值特徵進行 Z-Score（排除 date 和 target）
    # ==========================================
    feature_cols = [c for c in df_scaled.columns if c not in ['date', 'target']]

    if feature_cols:
        sc = StandardScaler()
        # 🚨 絕對關鍵：只用訓練集 (0 ~ train_end_idx) 來 fit 計算均值和標準差！
        sc.fit(df_scaled.loc[:train_end_idx, feature_cols])

        # transform 套用到全部資料 (包含測試集)
        df_scaled[feature_cols] = sc.transform(df_scaled[feature_cols])

    print(f"✅ 統一 Z-Score 完成！共縮放 {len(feature_cols)} 個特徵。")
    return df_scaled


