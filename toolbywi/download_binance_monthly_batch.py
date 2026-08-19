import pandas as pd
import requests
import zipfile
import io

def download_binance_monthly_batch(symbol="BTCUSDT", interval="1h", start_year=2024, start_month=1, end_year=2026, end_month=2):
    print(f"🚀 開始下載 {symbol} {interval} 月包資料 ({start_year}-{start_month:02d} 到 {end_year}-{end_month:02d})...")

    # 幣安預設的 12 個欄位
    columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Close_time', 'Quote_asset_volume', 'Number_of_trades',
               'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore']

    all_dataframes = []

    # 自動產生要下載的年月清單
    periods = pd.date_range(start=f"{start_year}-{start_month}-01",
                            end=f"{end_year}-{end_month}-01",
                            freq='MS')

    for dt in periods:
        year = dt.year
        month = dt.month
        file_name = f"{symbol}-{interval}-{year}-{month:02d}.zip"
        url = f"https://data.binance.vision/data/futures/um/monthly/klines/{symbol}/{interval}/{file_name}"

        try:
            response = requests.get(url)
            if response.status_code == 404:
                print(f"⚠️ 找不到 {year}-{month:02d} 的月包 (官方可能尚未釋出)")
                continue

            response.raise_for_status()

            # 在記憶體中直接解壓縮讀取 CSV
            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df_month = pd.read_csv(f, header=None, names=columns)
                    all_dataframes.append(df_month)
                    

        except Exception as e:
            print(f"❌ 下載 {year}-{month:02d} 時發生錯誤: {e}")

    if not all_dataframes:
        print("⚠️ 沒有抓到任何資料，請檢查網路。")
        return None

    # 將所有月份資料合併
    final_df = pd.concat(all_dataframes, ignore_index=True)

    # 【關鍵修復】：強制把 Open_time 轉為數字，將混在裡面的英文字母表頭清掉
    final_df['Open_time'] = pd.to_numeric(final_df['Open_time'], errors='coerce')
    final_df = final_df.dropna(subset=['Open_time'])

    # 整理時間與轉換資料型態
    final_df['Open_time'] = pd.to_datetime(final_df['Open_time'], unit='ms')
    final_df.set_index('Open_time', inplace=True)

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume','Quote_asset_volume','Taker_buy_quote_asset_volume']
    final_df = final_df[numeric_cols].astype(float)

    print(f"\n🎉 合併完成！總共取得 {len(final_df)} 筆 K 線資料。")

    # Time-Series-Library 預設需要一個名為 'date' 的時間欄位
    final_df.reset_index(inplace=True)
    final_df.rename(columns={'Open_time': 'date'}, inplace=True)

    return final_df


def binance_load_5min(symbol="BTCUSDT", start_year=2024, start_month=1,
                      end_year=2026, end_month=2):
    """
    下載幣安 5 分鐘 K 線資料（使用 daily 壓縮檔）。
    輸出格式與 download_binance_monthly_batch() 完全一致。

    參數:
        symbol     (str): 交易對，預設 'BTCUSDT'
        start_year (int): 起始年
        start_month(int): 起始月
        end_year   (int): 結束年
        end_month  (int): 結束月
    回傳:
        DataFrame: columns = [date, Open, High, Low, Close, Volume,
                              Quote_asset_volume, Taker_buy_quote_asset_volume]
    """
    interval = "5m"
    print(f"🚀 開始下載 {symbol} {interval} 日包資料 ({start_year}-{start_month:02d} 到 {end_year}-{end_month:02d})...")

    columns = ['Open_time', 'Open', 'High', 'Low', 'Close', 'Volume',
               'Close_time', 'Quote_asset_volume', 'Number_of_trades',
               'Taker_buy_base_asset_volume', 'Taker_buy_quote_asset_volume', 'Ignore']

    all_dataframes = []

    # 產生每日日期清單
    periods = pd.date_range(start=f"{start_year}-{start_month:02d}-01",
                            end=f"{end_year}-{end_month:02d}-01",
                            freq='D')

    for dt in periods:
        y, m, d = dt.year, dt.month, dt.day
        file_name = f"{symbol}-{interval}-{y}-{m:02d}-{d:02d}.zip"
        url = (f"https://data.binance.vision/data/futures/um/daily/klines/"
               f"{symbol}/{interval}/{file_name}")

        try:
            response = requests.get(url)
            if response.status_code == 404:
                # 未來日期或尚未釋出，靜默跳過
                continue

            response.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                csv_filename = z.namelist()[0]
                with z.open(csv_filename) as f:
                    df_day = pd.read_csv(f, header=None, names=columns)
                    all_dataframes.append(df_day)
                

        except Exception as e:
            print(f"❌ 下載 {y}-{m:02d}-{d:02d} 時發生錯誤: {e}")

    if not all_dataframes:
        print("⚠️ 沒有抓到任何資料，請檢查網路。")
        return None

    final_df = pd.concat(all_dataframes, ignore_index=True)

    # 清理混入的表頭列
    final_df['Open_time'] = pd.to_numeric(final_df['Open_time'], errors='coerce')
    final_df = final_df.dropna(subset=['Open_time'])

    final_df['Open_time'] = pd.to_datetime(final_df['Open_time'], unit='ms')
    final_df.set_index('Open_time', inplace=True)

    numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume',
                    'Quote_asset_volume', 'Taker_buy_quote_asset_volume']
    final_df = final_df[numeric_cols].astype(float)

    print(f"\n🎉 合併完成！總共取得 {len(final_df)} 筆 5 分鐘 K 線資料。")

    final_df.reset_index(inplace=True)
    final_df.rename(columns={'Open_time': 'date'}, inplace=True)

    return final_df


def binance_load_5min_fast(symbol="BTCUSDT", start_year=2025, start_month=1,
                            end_year=2026, end_month=8):
    """
    5min 資料快速下載：優先用月包（快，一次一整月），月包 404 的部分
    （通常是最近 1-2 個月尚未封存，或未來月份）自動改用日包
    （binance_load_5min）逐日補齊，確保區間內完全不缺資料。

    參數的 start/end 採「月包」慣例：inclusive，即 end_year-end_month
    當月會被納入（與 download_binance_monthly_batch 一致）。

    輸出格式與 binance_load_5min() 完全一致：
    [date, Open, High, Low, Close, Volume, Quote_asset_volume, Taker_buy_quote_asset_volume]
    """
    print(f"🚀 [快速模式] {symbol} 5m：先嘗試月包 ({start_year}-{start_month:02d} ~ {end_year}-{end_month:02d})...")
    df_monthly = download_binance_monthly_batch(
        symbol=symbol, interval="5m",
        start_year=start_year, start_month=start_month,
        end_year=end_year, end_month=end_month,
    )

    expected_months = pd.date_range(
        start=f"{start_year}-{start_month:02d}-01",
        end=f"{end_year}-{end_month:02d}-01",
        freq='MS',
    )

    got_months = set()
    if df_monthly is not None and len(df_monthly) > 0:
        got_months = set(pd.to_datetime(df_monthly['date']).dt.to_period('M'))

    missing = [p for p in expected_months if p.to_period('M') not in got_months]

    frames = [df_monthly] if df_monthly is not None else []

    if missing:
        print(f"⚠️ 月包缺 {len(missing)} 個月，改用日包補齊：{[str(p.to_period('M')) for p in missing]}")
        for p in missing:
            y, m = p.year, p.month
            # binance_load_5min 的 end_year/end_month 只涵蓋到 end 月份第 1 天，
            # 要 +1 個月才能抓到該月份完整資料。
            end_y, end_m = (y, m + 1) if m < 12 else (y + 1, 1)
            df_day = binance_load_5min(symbol=symbol,
                                        start_year=y, start_month=m,
                                        end_year=end_y, end_month=end_m)
            if df_day is not None:
                frames.append(df_day)

    if not frames:
        print("⚠️ 沒有抓到任何資料。")
        return None

    final_df = pd.concat(frames, ignore_index=True)
    final_df = final_df.drop_duplicates(subset='date').sort_values('date').reset_index(drop=True)
    print(f"🎉 [快速模式] 合併完成！總共 {len(final_df)} 筆 5 分鐘 K 線資料 "
          f"({final_df['date'].min()} ~ {final_df['date'].max()})。")
    return final_df
