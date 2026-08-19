import pandas as pd
import numpy as np
import joblib
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


def microstructure_features(df):
    """
    Round 4 新增：從被 prepare_event_data.py 丟棄的 5 個原始欄
    （Open/High/Low/Quote_asset_volume/Taker_buy_quote_asset_volume）
    衍生出的 6 個平穩微結構特徵。全部是逐根 pointwise 轉換（ATR_Pct_14
    例外，重用 features() 已算好的 ATR_14），不新增 warmup。

    設計依據（見 Round 4 plan）：
    - Open/High/Low 原始值與 Close 相關 0.99997+，是非平穩價格水位的複本，
      不可直接當特徵；轉成相對於 Close 的 %，實測 train->test 漂移全部
      < 0.24 sd（對照 EMA_72 的 -2.51 sd、OBV 的 -2.84 sd）。
    - QVol_Ratio（Quote_asset_volume 的滾動相對量）已剔除：與既有
      Volume_Ratio 相關 0.99998（Quote_asset_volume ≈ Volume * Close，
      20 根滾動比值把價格因子約掉了），加入只會讓兩者在 permutation
      importance 裡互相瓜分。
    - Gap_Open（跳空）已剔除：永續合約 24/7 無收盤，實測 51.8% 為精確 0，
      std 僅 7e-4%，是把股票概念套到沒有交易時段的市場。
    - Taker_buy_quote_asset_volume 已由既有 features() 的 Taker_Ratio 消化，
      不重複處理。

    要求 df 已經過 features()（需要 Open/High/Low/Close/Volume/
    Quote_asset_volume、以及 ATR_14）。
    """
    print("🛠️ 開始計算 Round 4 微結構特徵（6 個）...")
    df = df.copy()

    df['HL_Range'] = (df['High'] - df['Low']) / df['Close'] * 100
    df['OC_Ret'] = (df['Close'] - df['Open']) / df['Open'] * 100
    df['Up_Wick'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close'] * 100
    df['Low_Wick'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close'] * 100

    # BarVWAP_Dev：bar 內成交均價（Quote_asset_volume / Volume）與 Close 的偏離。
    # Volume==0（無成交的冷門 bar）時 Open==High==Low==Close，VWAP 未定義，
    # 正確值是 0（不是 sentinel）——否則除以 0 產生 inf，會讓下游
    # StandardScaler.fit() 炸掉（同 extra_features() 的 n_inf 守衛要防的問題）。
    bar_vwap = df['Quote_asset_volume'] / df['Volume'].replace(0, np.nan)
    df['BarVWAP_Dev'] = np.where(
        df['Volume'] > 0,
        (df['Close'] - bar_vwap) / df['Close'] * 100,
        0.0,
    )

    # ATR_Pct_14：ATR_14（features() 已算好，R3 選中的 15 個特徵之一）是原始
    # 美元水位、隨幣價漂移（實測 -0.611 sd）；轉成 % 是零成本的平穩替代
    # （實測 -0.235 sd）。
    df['ATR_Pct_14'] = df['ATR_14'] / df['Close'] * 100

    feature_count = 6
    print(f"✅ 微結構特徵完成！新增 {feature_count} 個特徵："
          f"['HL_Range', 'OC_Ret', 'Up_Wick', 'Low_Wick', 'BarVWAP_Dev', 'ATR_Pct_14']")
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


def extra_features(df, metrics_df, funding_df, premium_df, mark_df):
    """
    Round 3 新特徵來源：未平倉量+多空比（metrics）、資金費率
    （fundingRate）、溢價指數/標記價格（premium/mark），共 19 個新特徵。

    對齊規則（因果性）：
    - metrics / premium / mark 與 K 線同一個 5 分鐘網格，直接依 'date'
      內連接（inner join）——任一來源缺某根就整根丟棄，不猜測補值。
    - funding 每 8 小時才更新一次，用 merge_asof(backward) 只取「已結算」
      的費率（calc_time <= date），嚴格因果。

    Parameters
    ----------
    df          : pd.DataFrame，含 'date'、'Close' 欄（已經過 features()/alt_features()）
    metrics_df  : toolbywi.binance_extra.load_metrics() 或
                  toolbywi.binance_live.get_live_metrics() 的輸出
    funding_df  : load_funding_rate() 或 get_live_funding() 的輸出
    premium_df  : load_premium_index() 或 get_live_premium_index() 的輸出
    mark_df     : load_mark_price() 或 get_live_mark_price() 的輸出

    Returns
    -------
    pd.DataFrame，原 df 欄位 + 19 個新欄位
    """
    print("🛠️ 開始合併 Round 3 新特徵來源（metrics / funding / premium / mark）...")
    merged = df.copy()
    merged['date'] = pd.to_datetime(merged['date'])
    n_before = len(merged)

    # ── A. metrics（未平倉量 + 多空比，12 個）──────────────────
    m = metrics_df.copy()
    m['date'] = pd.to_datetime(m['date'])
    merged = merged.merge(m, on='date', how='inner')

    merged['OI'] = merged['sum_open_interest']
    merged['OI_Value'] = merged['sum_open_interest_value']
    merged['OI_Chg_1'] = merged['OI'].pct_change(1) * 100
    merged['OI_Chg_6'] = merged['OI'].pct_change(6) * 100
    merged['OI_Chg_72'] = merged['OI'].pct_change(72) * 100
    merged['TopTrader_Pos_LS'] = merged['sum_toptrader_long_short_ratio']
    merged['TopTrader_Acct_LS'] = merged['count_toptrader_long_short_ratio']
    merged['Global_Acct_LS'] = merged['count_long_short_ratio']
    merged['Taker_LS_Vol'] = merged['sum_taker_long_short_vol_ratio']
    merged['TopTrader_Pos_LS_Chg6'] = merged['TopTrader_Pos_LS'].diff(6)
    merged['Global_Acct_LS_Chg6'] = merged['Global_Acct_LS'].diff(6)
    roc_6 = merged['Close'].pct_change(6) * 100
    merged['OI_Price_Div'] = merged['OI_Chg_6'] * roc_6  # 多空擠壓：OI 與價格背離幅度

    merged = merged.drop(columns=['sum_open_interest', 'sum_open_interest_value',
                                   'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
                                   'count_long_short_ratio', 'sum_taker_long_short_vol_ratio'])

    # ── C. premiumIndex / markPrice（溢價與基差，4 個）───────────
    p = premium_df.copy()
    p['date'] = pd.to_datetime(p['date'])
    mk = mark_df.copy()
    mk['date'] = pd.to_datetime(mk['date'])
    merged = merged.merge(p, on='date', how='inner').merge(mk, on='date', how='inner')

    merged['Premium_Mean6'] = merged['Premium'].rolling(6).mean()
    merged['Basis'] = (merged['Mark_Close'] - merged['Close']) / merged['Close'] * 100
    merged['Basis_Chg6'] = merged['Basis'].diff(6)
    merged = merged.drop(columns=['Mark_Close'])

    # ── B. fundingRate（資金費率，3 個）──────────────────────────
    f = funding_df.copy()
    f['date'] = pd.to_datetime(f['date'])
    f = f.sort_values('date').reset_index(drop=True)
    merged = merged.sort_values('date').reset_index(drop=True)
    merged = pd.merge_asof(
        merged, f[['date', 'last_funding_rate', 'funding_interval_hours']],
        on='date', direction='backward'
    )
    merged = merged.rename(columns={'last_funding_rate': 'Funding_Rate'})
    merged['Funding_Rate'] = merged['Funding_Rate'].ffill()
    # 相鄰結算差：ffill 後同一結算週期內都相同，只有換到新結算值那一刻才非零
    merged['Funding_Chg'] = merged['Funding_Rate'].diff()

    # Bars_To_Funding：由時鐘決定（不看任何未來資料），結算時刻的集合從
    # funding_df 本身的歷史資料推導，不硬編 [0,8,16]，以防 Binance 改制。
    funding_hours = sorted(pd.to_datetime(funding_df['date']).dt.hour.unique().tolist())
    if not funding_hours:
        funding_hours = [0, 8, 16]
    funding_minutes = sorted(h * 60 for h in funding_hours)

    # 用 >= 而非 >：結算那一根本身要算 0，不能跳到下一輪的 96
    # （用 > 會讓 00:00/08:00/16:00 這幾根誤算成距下次結算 96 根）。
    minutes_of_day = (merged['date'].dt.hour * 60 + merged['date'].dt.minute).values
    bars_to_funding = np.empty(len(minutes_of_day), dtype=float)
    for i, cur in enumerate(minutes_of_day):
        nxt = [fm for fm in funding_minutes if fm >= cur]
        delta = (min(nxt) - cur) if nxt else (1440 - cur + funding_minutes[0])
        bars_to_funding[i] = delta // 5
    merged['Bars_To_Funding'] = bars_to_funding

    merged = merged.drop(columns=['funding_interval_hours'])

    # metrics 資料極少數時間點會回報 sum_open_interest=0（交易所端回報缺口，
    # 不是真實未平倉量歸零——BTCUSDT 這種主力合約不會真的變成 0），
    # pct_change() 除以 0 會產生 inf，讓下游 StandardScaler.fit() 直接炸掉。
    # 實測 165,719 列裡只有個位數列受影響，換成 NaN 後交給既有的
    # dropna() 一起丟掉，不必為了這幾根改動特徵定義本身。
    n_inf = int(np.isinf(merged.select_dtypes(include=[np.number]).values).sum())
    if n_inf > 0:
        merged = merged.replace([np.inf, -np.inf], np.nan)
        print(f"⚠️ 偵測到 {n_inf} 個 inf 值（多半是 OI 資料缺口造成的除以 0），已轉成 NaN 交給 dropna() 處理")

    merged = merged.dropna().reset_index(drop=True)
    new_cols = [c for c in merged.columns if c not in df.columns]
    print(f"✅ 新特徵來源合併完成！{n_before} → {len(merged)} 列，新增 {len(new_cols)} 個特徵：{new_cols}")
    return merged


def scaler(df, train_ratio=0.7, return_scaler=False):
    print("🛠️ 開始執行統一 Z-Score 縮放 (防未來洩漏機制啟動)...")
    df_scaled = df.copy()

    # 計算訓練集的截止邊界
    train_end_idx = int(len(df_scaled) * train_ratio)

    # ==========================================
    # 📌 統一對所有數值特徵進行 Z-Score（排除 date 和 target）
    # ==========================================
    feature_cols = [c for c in df_scaled.columns if c not in ['date', 'target']]

    sc = None
    if feature_cols:
        sc = StandardScaler()
        # 🚨 絕對關鍵：只用訓練集 (0 ~ train_end_idx) 來 fit 計算均值和標準差！
        sc.fit(df_scaled.loc[:train_end_idx, feature_cols])

        # transform 套用到全部資料 (包含測試集)
        df_scaled[feature_cols] = sc.transform(df_scaled[feature_cols])

    print(f"✅ 統一 Z-Score 完成！共縮放 {len(feature_cols)} 個特徵。")

    if return_scaler:
        return df_scaled, sc, feature_cols
    return df_scaled


def save_scaler(path, sc, feature_cols):
    """持久化 fitted StandardScaler + 對應的特徵欄序，供即時推論重用。"""
    joblib.dump({'scaler': sc, 'feature_cols': list(feature_cols)}, path)
    print(f"💾 scaler 已存至 {path}")


def load_scaler(path):
    """讀回 save_scaler() 存的 (scaler, feature_cols)。"""
    obj = joblib.load(path)
    return obj['scaler'], obj['feature_cols']


def apply_scaler(df, sc, feature_cols):
    """
    用已 fit 好的 scaler 套用到新資料（例如即時推論的最新視窗）。
    要求 df 必須含 feature_cols 的全部欄位，欄位順序不需相同（內部依名稱取值），
    但缺任何一欄會直接報錯，而不是靜默補值。
    """
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"apply_scaler: df 缺少欄位 {missing}，無法套用 scaler。")

    df_scaled = df.copy()
    df_scaled[feature_cols] = sc.transform(df_scaled[feature_cols])
    return df_scaled


