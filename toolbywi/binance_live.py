"""
即時 5min K 線抓取（Binance USDT-M 永續合約公開 REST，無需 API key）。

與 toolbywi/download_binance_monthly_batch.py 的離線資料下載互補：
- 離線（data.binance.vision）：訓練資料，落後約 1 天，適合大量歷史回補
- 這裡（fapi.binance.com REST）：即時推論，適合抓最新一小段視窗

安全機制（對應計畫中的 L2 洩漏路徑）：
- 一律丟棄最後一根尚未收盤的 K 線
- 陳舊資料檢查：最新已收盤 K 線若距現在太久，視為異常並拒絕輸出訊號
"""
import time
import requests
import pandas as pd

BASE_URL = "https://fapi.binance.com/fapi/v1/klines"
TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/price"

_INTERVAL_MS = {
    '1m': 60_000, '3m': 180_000, '5m': 300_000, '15m': 900_000,
    '30m': 1_800_000, '1h': 3_600_000,
}

_RAW_COLUMNS = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
                'Close_time', 'Quote_asset_volume', 'Number_of_trades',
                'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore']

_NUMERIC_COLS = ['Open', 'High', 'Low', 'Close', 'Volume',
                  'Quote_asset_volume', 'Taker_buy_quote_asset_volume']


class StaleDataError(RuntimeError):
    """最新已收盤 K 線距現在太久，資料可能斷線或延遲，不安全用於下單決策。"""
    pass


def fetch_recent_klines(symbol="BTCUSDT", interval="5m", total_bars=1524, timeout=10):
    """
    抓取最近 total_bars 根 K 線（由舊到新排序），自動分頁（單次上限 1500）。
    回傳的最後一根可能是尚未收盤的當前 K 線 —— 這裡只負責忠實抓資料，
    是否丟棄未收盤那根由 get_latest_closed_klines() 決定。
    """
    if interval not in _INTERVAL_MS:
        raise ValueError(f"不支援的 interval: {interval}")

    chunks = []
    end_time = None  # None = 抓到現在為止
    remaining = total_bars

    while remaining > 0:
        batch_limit = min(remaining, 1500)
        params = {'symbol': symbol, 'interval': interval, 'limit': batch_limit}
        if end_time is not None:
            params['endTime'] = end_time

        resp = requests.get(BASE_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break

        df_chunk = pd.DataFrame(data, columns=_RAW_COLUMNS)
        chunks.append(df_chunk)
        remaining -= len(df_chunk)

        earliest_open = int(df_chunk['Open_time'].iloc[0])
        end_time = earliest_open - 1
        if len(df_chunk) < batch_limit:
            break  # 伺服器已無更早資料

    if not chunks:
        return None

    final = pd.concat(chunks, ignore_index=True)
    final = final.drop_duplicates(subset='Open_time').sort_values('Open_time').reset_index(drop=True)

    for c in _NUMERIC_COLS:
        final[c] = final[c].astype(float)
    final['Open_time'] = final['Open_time'].astype('int64')
    final['Close_time'] = final['Close_time'].astype('int64')

    final['date'] = pd.to_datetime(final['Open_time'], unit='ms')
    final['close_time_dt'] = pd.to_datetime(final['Close_time'], unit='ms')

    out = final[['date', 'Open', 'High', 'Low', 'Close', 'Volume',
                 'Quote_asset_volume', 'Taker_buy_quote_asset_volume', 'close_time_dt']].copy()

    return out.tail(total_bars).reset_index(drop=True)


def get_latest_closed_klines(symbol="BTCUSDT", interval="5m", total_bars=1524,
                              staleness_minutes=6, timeout=10):
    """
    高階介面：抓最新 K 線 → 丟棄未收盤那根 → 陳舊性檢查 → 回傳與離線
    loader 相同 schema 的 DataFrame：
    [date, Open, High, Low, Close, Volume, Quote_asset_volume, Taker_buy_quote_asset_volume]

    total_bars 是「含未收盤那根」的抓取數量，所以回傳結果會比 total_bars 少 1 根。
    """
    df = fetch_recent_klines(symbol=symbol, interval=interval,
                              total_bars=total_bars, timeout=timeout)
    if df is None or len(df) == 0:
        raise RuntimeError(f"{symbol} 未抓到任何 K 線資料。")

    now = pd.Timestamp.now('UTC').tz_localize(None)
    last_close_time = df['close_time_dt'].iloc[-1]

    # 最後一根是否仍在形成中：其收盤時間還沒到
    if last_close_time > now:
        df = df.iloc[:-1].copy()

    if len(df) == 0:
        raise RuntimeError(f"{symbol} 丟棄未收盤 K 線後沒有剩餘資料，稍後再試。")

    latest_closed_time = df['close_time_dt'].iloc[-1]
    age_minutes = (now - latest_closed_time).total_seconds() / 60.0
    if age_minutes > staleness_minutes:
        raise StaleDataError(
            f"{symbol} 最新已收盤 K 線是 {latest_closed_time} UTC，"
            f"距現在 {age_minutes:.1f} 分鐘（門檻 {staleness_minutes} 分鐘）。"
            f"資料可能延遲或連線異常，拒絕輸出訊號。"
        )

    return df.drop(columns=['close_time_dt']).reset_index(drop=True)


def get_current_price(symbol="BTCUSDT", timeout=5):
    """即時參考價（非 K 線收盤價），用於檢查訊號是否已過期。"""
    resp = requests.get(TICKER_URL, params={'symbol': symbol}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    return float(data['price'])


# ══════════════════════════════════════════════════════════════
# Round 3 新特徵來源的即時版本 —— 欄名必須與 toolbywi/binance_extra.py
# 的歷史 loader 完全相同（訓練/上線一致性的唯一保證，已用交叉比對驗證
# 對應關係，見計畫文件）。
# ══════════════════════════════════════════════════════════════
_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_PREMIUM_KLINES_URL = "https://fapi.binance.com/fapi/v1/premiumIndexKlines"
_MARK_KLINES_URL = "https://fapi.binance.com/fapi/v1/markPriceKlines"


def _fetch_futures_data_paginated(path, symbol, period, total_bars, timeout=10):
    """
    通用分頁抓取器，適用於 openInterestHist / topLongShortAccountRatio /
    topLongShortPositionRatio / globalLongShortAccountRatio /
    takerlongshortRatio —— 這五個端點的參數與分頁行為完全相同（實測 limit
    可達 1000，文件寫的 500 上限不是硬限制）。
    """
    chunks = []
    end_time = None
    remaining = total_bars
    while remaining > 0:
        batch_limit = min(remaining, 1000)
        params = {'symbol': symbol, 'period': period, 'limit': batch_limit}
        if end_time is not None:
            params['endTime'] = end_time
        resp = requests.get(f"https://fapi.binance.com{path}", params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        chunks.extend(data)
        remaining -= len(data)
        earliest_ts = min(int(d['timestamp']) for d in data)
        end_time = earliest_ts - 1
        if len(data) < batch_limit:
            break
    return chunks


def _records_to_df(records, value_cols_map):
    """records（list of dict）依 value_cols_map={來源欄名: 目標欄名} 轉成
    [date, 目標欄名...] 的 DataFrame，數值欄轉 float、依 date 去重排序。"""
    if not records:
        return None
    df = pd.DataFrame(records)
    df['date'] = pd.to_datetime(df['timestamp'].astype('int64'), unit='ms')
    out = df[['date'] + list(value_cols_map.keys())].rename(columns=value_cols_map)
    for c in value_cols_map.values():
        out[c] = pd.to_numeric(out[c], errors='coerce')
    return out.dropna().drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)


def get_live_metrics(symbol="BTCUSDT", total_bars=1012, timeout=10):
    """
    未平倉量 + 多空比即時版，對應 binance_extra.load_metrics() 的六個欄位。
    五個端點各自抓、依 date 內連接合併——任一端點缺某根就整根丟棄，
    不猜測補值。
    """
    oi = _fetch_futures_data_paginated("/futures/data/openInterestHist", symbol, "5m", total_bars, timeout)
    top_acct = _fetch_futures_data_paginated("/futures/data/topLongShortAccountRatio", symbol, "5m", total_bars, timeout)
    top_pos = _fetch_futures_data_paginated("/futures/data/topLongShortPositionRatio", symbol, "5m", total_bars, timeout)
    glob_acct = _fetch_futures_data_paginated("/futures/data/globalLongShortAccountRatio", symbol, "5m", total_bars, timeout)
    taker = _fetch_futures_data_paginated("/futures/data/takerlongshortRatio", symbol, "5m", total_bars, timeout)

    df_oi = _records_to_df(oi, {'sumOpenInterest': 'sum_open_interest',
                                 'sumOpenInterestValue': 'sum_open_interest_value'})
    df_ta = _records_to_df(top_acct, {'longShortRatio': 'count_toptrader_long_short_ratio'})
    df_tp = _records_to_df(top_pos, {'longShortRatio': 'sum_toptrader_long_short_ratio'})
    df_ga = _records_to_df(glob_acct, {'longShortRatio': 'count_long_short_ratio'})
    df_tk = _records_to_df(taker, {'buySellRatio': 'sum_taker_long_short_vol_ratio'})

    parts = {'openInterestHist': df_oi, 'topLongShortAccountRatio': df_ta,
             'topLongShortPositionRatio': df_tp, 'globalLongShortAccountRatio': df_ga,
             'takerlongshortRatio': df_tk}
    missing = [k for k, v in parts.items() if v is None]
    if missing:
        raise RuntimeError(f"get_live_metrics: 端點沒抓到資料: {missing}")

    out = df_oi
    for f in (df_ta, df_tp, df_ga, df_tk):
        out = out.merge(f, on='date', how='inner')
    return out.reset_index(drop=True)


def get_live_funding(symbol="BTCUSDT", limit=60, timeout=10):
    """
    資金費率即時版，對應 binance_extra.load_funding_rate()。即時端點不像
    歷史 dump 回傳 funding_interval_hours，用歷史資料實測跨 4 個月份確認
    恆為 8 小時的結果直接填常數（見計畫文件的確認事項）。
    """
    resp = requests.get(_FUNDING_URL, params={'symbol': symbol, 'limit': limit}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not data:
        raise RuntimeError(f"{symbol} 沒抓到 fundingRate 資料。")

    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['fundingTime'].astype('int64'), unit='ms')
    df['last_funding_rate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
    df['funding_interval_hours'] = 8

    out = df[['date', 'funding_interval_hours', 'last_funding_rate']].dropna()
    return out.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)


def _fetch_kline_style_paginated(url, symbol, total_bars, value_col, timeout=10):
    """
    premiumIndexKlines / markPriceKlines 共用：跟主 K 線一樣是標準 12 欄
    kline 陣列格式、單次上限 1500，分頁邏輯與 fetch_recent_klines() 相同。
    回傳 [date, value_col]，**不主動丟棄未收盤那根**——留給呼叫端跟主
    K 線用 date 內連接時自然對齊（主 K 線已經丟過未收盤那根）。
    """
    chunks = []
    end_time = None
    remaining = total_bars
    while remaining > 0:
        batch_limit = min(remaining, 1500)
        params = {'symbol': symbol, 'interval': '5m', 'limit': batch_limit}
        if end_time is not None:
            params['endTime'] = end_time
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        chunks.extend(data)
        remaining -= len(data)
        earliest_open = int(data[0][0])
        end_time = earliest_open - 1
        if len(data) < batch_limit:
            break

    if not chunks:
        return None

    df = pd.DataFrame(chunks)
    df['date'] = pd.to_datetime(df[0].astype('int64'), unit='ms')
    df[value_col] = pd.to_numeric(df[4], errors='coerce')  # index 4 = close
    out = df[['date', value_col]].dropna()
    return out.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)


def get_live_premium_index(symbol="BTCUSDT", total_bars=1012, timeout=10):
    """溢價指數即時版，對應 binance_extra.load_premium_index()，回傳 [date, Premium]。"""
    out = _fetch_kline_style_paginated(_PREMIUM_KLINES_URL, symbol, total_bars, 'Premium', timeout)
    if out is None:
        raise RuntimeError(f"{symbol} 沒抓到 premiumIndexKlines 資料。")
    return out


def get_live_mark_price(symbol="BTCUSDT", total_bars=1012, timeout=10):
    """標記價格即時版，對應 binance_extra.load_mark_price()，回傳 [date, Mark_Close]。"""
    out = _fetch_kline_style_paginated(_MARK_KLINES_URL, symbol, total_bars, 'Mark_Close', timeout)
    if out is None:
        raise RuntimeError(f"{symbol} 沒抓到 markPriceKlines 資料。")
    return out


if __name__ == '__main__':
    df = get_latest_closed_klines('BTCUSDT', total_bars=20)
    print(df.tail(5))
    print('current price:', get_current_price('BTCUSDT'))
