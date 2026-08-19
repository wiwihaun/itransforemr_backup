"""
Round 3 新特徵來源的歷史資料下載：未平倉量/多空比（metrics）、資金費率
（fundingRate）、溢價指數/標記價格（premiumIndexKlines / markPriceKlines）。

沿用 download_binance_monthly_batch.py 的月包優先 + 日包補齊風格，但這幾組
來源各自的月包/日包可用性不同（metrics 只有 daily；funding/premium/mark
只有 monthly），所以各自實作對應的下載函式，不硬套同一個 fast wrapper。

欄名刻意保留 Binance 官方 CSV 原始欄名（僅 create_time/calc_time/open_time
統一改名為 date），這樣 toolbywi/binance_live.py 的即時版本才能用完全相同
的欄名對齊——見 feature_scale.extra_features() 的用法。
"""
import pandas as pd
import requests
import zipfile
import io


def load_metrics(symbol="BTCUSDT", start_year=2025, start_month=1,
                  end_year=2026, end_month=8):
    """
    未平倉量 + 多空比，5 分鐘原生粒度。只有 daily 封存（無 monthly），
    逐日下載。回傳欄位與官方 CSV 一致，只把 create_time 改名為 date：
    [date, sum_open_interest, sum_open_interest_value,
     count_toptrader_long_short_ratio, sum_toptrader_long_short_ratio,
     count_long_short_ratio, sum_taker_long_short_vol_ratio]
    """
    print(f"🚀 開始下載 {symbol} metrics 日包資料 ({start_year}-{start_month:02d} 到 {end_year}-{end_month:02d})...")

    periods = pd.date_range(start=f"{start_year}-{start_month:02d}-01",
                             end=f"{end_year}-{end_month:02d}-01", freq='D')

    frames = []
    for dt in periods:
        y, m, d = dt.year, dt.month, dt.day
        file_name = f"{symbol}-metrics-{y}-{m:02d}-{d:02d}.zip"
        url = f"https://data.binance.vision/data/futures/um/daily/metrics/{symbol}/{file_name}"
        try:
            resp = requests.get(url)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"❌ 下載 {y}-{m:02d}-{d:02d} metrics 時發生錯誤: {e}")

    if not frames:
        print("⚠️ 沒有抓到任何 metrics 資料。")
        return None

    df = pd.concat(frames, ignore_index=True)
    df['create_time'] = pd.to_datetime(df['create_time'], errors='coerce')
    df = df.dropna(subset=['create_time'])

    keep = ['create_time', 'sum_open_interest', 'sum_open_interest_value',
            'count_toptrader_long_short_ratio', 'sum_toptrader_long_short_ratio',
            'count_long_short_ratio', 'sum_taker_long_short_vol_ratio']
    df = df[keep].rename(columns={'create_time': 'date'})
    for c in df.columns:
        if c != 'date':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna().drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)

    print(f"🎉 metrics 下載完成！共 {len(df)} 筆（{df['date'].min()} ~ {df['date'].max()}）。")
    return df


def load_funding_rate(symbol="BTCUSDT", start_year=2025, start_month=1,
                       end_year=2026, end_month=8):
    """
    資金費率，8 小時結算一次。只有 monthly 封存。回傳：
    [date, funding_interval_hours, last_funding_rate]
    date 取自 calc_time（即時端對應 fundingTime）。
    """
    print(f"🚀 開始下載 {symbol} fundingRate 月包資料 ({start_year}-{start_month:02d} 到 {end_year}-{end_month:02d})...")

    periods = pd.date_range(start=f"{start_year}-{start_month:02d}-01",
                             end=f"{end_year}-{end_month:02d}-01", freq='MS')

    frames = []
    for dt in periods:
        y, m = dt.year, dt.month
        file_name = f"{symbol}-fundingRate-{y}-{m:02d}.zip"
        url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{symbol}/{file_name}"
        try:
            resp = requests.get(url)
            if resp.status_code == 404:
                print(f"⚠️ 找不到 {y}-{m:02d} 的 fundingRate 月包")
                continue
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    frames.append(pd.read_csv(f))
        except Exception as e:
            print(f"❌ 下載 {y}-{m:02d} fundingRate 時發生錯誤: {e}")

    if not frames:
        print("⚠️ 沒有抓到任何 fundingRate 資料。")
        return None

    df = pd.concat(frames, ignore_index=True)
    df['calc_time'] = pd.to_numeric(df['calc_time'], errors='coerce')
    df = df.dropna(subset=['calc_time'])
    df['date'] = pd.to_datetime(df['calc_time'].astype('int64'), unit='ms')

    df = df[['date', 'funding_interval_hours', 'last_funding_rate']].copy()
    df['funding_interval_hours'] = pd.to_numeric(df['funding_interval_hours'], errors='coerce')
    df['last_funding_rate'] = pd.to_numeric(df['last_funding_rate'], errors='coerce')
    df = df.dropna().drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)

    print(f"🎉 fundingRate 下載完成！共 {len(df)} 筆（{df['date'].min()} ~ {df['date'].max()}）。")
    return df


def _load_5m_kline_style(symbol, dataset, value_col, start_year, start_month, end_year, end_month):
    """
    premiumIndexKlines 與 markPriceKlines 都是標準 12 欄 K 線格式（只是數值
    含義不同），且都有 monthly 封存，共用同一份下載邏輯。
    """
    print(f"🚀 開始下載 {symbol} {dataset} 5m 月包資料 ({start_year}-{start_month:02d} 到 {end_year}-{end_month:02d})...")

    columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Close_time', 'Quote_asset_volume', 'Number_of_trades',
               'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore']

    periods = pd.date_range(start=f"{start_year}-{start_month:02d}-01",
                             end=f"{end_year}-{end_month:02d}-01", freq='MS')

    frames = []
    for dt in periods:
        y, m = dt.year, dt.month
        file_name = f"{symbol}-5m-{y}-{m:02d}.zip"
        url = f"https://data.binance.vision/data/futures/um/monthly/{dataset}/{symbol}/5m/{file_name}"
        try:
            resp = requests.get(url)
            if resp.status_code == 404:
                print(f"⚠️ 找不到 {y}-{m:02d} 的 {dataset} 月包")
                continue
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                with z.open(z.namelist()[0]) as f:
                    frames.append(pd.read_csv(f, header=None, names=columns))
        except Exception as e:
            print(f"❌ 下載 {y}-{m:02d} {dataset} 時發生錯誤: {e}")

    if not frames:
        print(f"⚠️ 沒有抓到任何 {dataset} 資料。")
        return None

    df = pd.concat(frames, ignore_index=True)
    df['Open_time'] = pd.to_numeric(df['Open_time'], errors='coerce')
    df = df.dropna(subset=['Open_time'])
    df['date'] = pd.to_datetime(df['Open_time'].astype('int64'), unit='ms')
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')

    df = df[['date', 'Close']].rename(columns={'Close': value_col})
    df = df.dropna().drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)

    print(f"🎉 {dataset} 下載完成！共 {len(df)} 筆（{df['date'].min()} ~ {df['date'].max()}）。")
    return df


def load_premium_index(symbol="BTCUSDT", start_year=2025, start_month=1,
                        end_year=2026, end_month=8):
    """溢價指數，5 分鐘。回傳 [date, Premium]。"""
    return _load_5m_kline_style(symbol, "premiumIndexKlines", "Premium",
                                 start_year, start_month, end_year, end_month)


def load_mark_price(symbol="BTCUSDT", start_year=2025, start_month=1,
                     end_year=2026, end_month=8):
    """標記價格，5 分鐘。回傳 [date, Mark_Close]。"""
    return _load_5m_kline_style(symbol, "markPriceKlines", "Mark_Close",
                                 start_year, start_month, end_year, end_month)
