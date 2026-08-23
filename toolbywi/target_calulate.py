import numpy as np
import pandas as pd

def target_direction(df, lookahead=6):
    """
    滾動方向標籤（供固定時間結算的事件合約使用，例如 30 分鐘 = 5min K 線 × 6 根）。
    target[i] = 1 若 Close[i+lookahead] > Close[i]，否則 0。
    向量化實作（無雙層迴圈），並回傳有效遮罩：
    尾端 lookahead 根因為看不到未來結算價，標籤無效，呼叫端必須依 valid 篩掉，
    不能留著（會被當成 0 而混進訓練/評估）。

    回傳
    ----
    targets : np.ndarray[int]，形狀與 df 同長
    valid   : np.ndarray[bool]，True 代表該列標籤可用
    """
    close = df['Close'].values
    n = len(df)

    future_close = np.full(n, np.nan)
    future_close[:n - lookahead] = close[lookahead:]

    valid = ~np.isnan(future_close)
    targets = np.zeros(n, dtype=int)
    targets[valid] = (future_close[valid] > close[valid]).astype(int)

    return targets, valid


# ══════════════════════════════════════════════════════════════
# Round 5：幅度門檻標籤 + 連續段取中間
#
# 動機：Round 1-4 的標籤是「N 根之後有沒有漲」，正樣本率約 50%，其中
# 大量樣本是貼近 0% 的擲硬幣雜訊。Round 5 只把「強勁且持續的上漲」標成
# 1：先用幅度門檻（漲幅 >= min_pct）濾掉微幅波動，再對連續段只保留
# 中間位置——實測中間那根的 1 小時報酬（+0.756%）明顯高於段落起點
# （+0.483%）與終點（+0.471%），是整段最強的位置。
#
# 重要：這裡產生的是「訓練標籤」，不是「合約勝負」。事件合約仍然是漲就
# 贏（見 contract_target），兩者必須分開，不可混用——用稀疏標籤算勝率
# 會得到 1.8% 這種與損益兩平 54.05% 完全不可比的數字。
# ══════════════════════════════════════════════════════════════
def run_center_transform(y, min_run):
    """
    對 0/1 序列做「連續段只保留中間」的轉換。

    規則（對每個長度 L >= min_run 的極大連續 1 段，起點 s）：
      - L 為奇數 → 只有 s + (L-1)//2 這 1 根設為 1
      - L 為偶數 → s + L//2 - 1 與 s + L//2 這 2 根都設為 1
    長度 < min_run 的段落全部歸 0。

    偶數取 2 根是刻意的（使用者指定）：長度 6 的段落沒有單一中點，
    取中間兩根才對稱。這個規則會影響正樣本率約 1.5 倍，是行為常數，
    改動前要先確認呼叫端的預期。

    範例
    ----
    >>> ''.join(map(str, run_center_transform([0,0,1,1,1,1,1,0,1,0], 5)))
    '0000100000'
    >>> ''.join(map(str, run_center_transform([0,1,1,1,1,1,1,0,0], 5)))
    '000110000'

    參數
    ----
    y       : array-like[int]，0/1 序列。必須是「連續等間隔」的時間網格，
              否則「連續」沒有意義（見 prepare_event_data.py 的呼叫點註解）。
    min_run : int，段落長度下限（含）

    回傳
    ----
    np.ndarray[int]，與 y 同長
    """
    y = np.asarray(y, dtype=int)
    z = np.zeros_like(y)
    start = None
    # 尾端補一個 0，讓最後一段也能被同一套邏輯收尾（不改變輸入）
    for i, v in enumerate(np.append(y, 0)):
        if v == 1 and start is None:
            start = i
        elif v != 1 and start is not None:
            end = i - 1
            length = end - start + 1
            if length >= min_run:
                if length % 2 == 1:
                    z[start + (length - 1) // 2] = 1
                else:
                    z[start + length // 2 - 1] = 1
                    z[start + length // 2] = 1
            start = None
    return z


def trailing_run_start(y):
    """
    尾端連續 1 段的起始 index；若 y[-1] != 1 則回傳 len(y)（代表沒有被截斷的尾段）。
    用於 Round 5 的尾端設限處理（見 target_run_center 的 R2 規則）。
    """
    y = np.asarray(y, dtype=int)
    n = len(y)
    if n == 0 or y[-1] != 1:
        return n
    i = n - 1
    while i >= 0 and y[i] == 1:
        i -= 1
    return i + 1


def target_pct_threshold(df, lookahead=12, min_pct=0.003):
    """
    幅度門檻方向標籤：target[i] = 1 若 Close[i+lookahead] >= Close[i] * (1 + min_pct)。

    回傳
    ----
    targets : np.ndarray[int]
    valid   : np.ndarray[bool]，尾端 lookahead 根為 False（看不到結算價）
    fwd_ret : np.ndarray[float]，Close[i+lookahead]/Close[i] - 1，無效處為 nan
    """
    close = df['Close'].values.astype(float)
    n = len(df)

    future_close = np.full(n, np.nan)
    future_close[:n - lookahead] = close[lookahead:]  # noqa: lookahead - 標籤建構，設計如此

    valid = ~np.isnan(future_close)
    fwd_ret = np.full(n, np.nan)
    fwd_ret[valid] = future_close[valid] / close[valid] - 1.0

    targets = np.zeros(n, dtype=int)
    targets[valid] = (fwd_ret[valid] >= min_pct).astype(int)

    return targets, valid, fwd_ret


def target_run_center(df, lookahead=12, min_pct=0.003, min_run=5, min_pct_floor=None):
    """
    Round 5 訓練標籤：幅度門檻 + 連續段取中間，含三層尾端設限處理。

    尾端規則（順序執行）：
      R1 視野尾端：fwd_ret 在最後 lookahead 根未定義 → valid=False
      R2 被截斷的尾段：若剩餘序列最後一格是 1，該連續段是右設限的
         （真實長度與中點都未知）→ 整段 valid=False。注意是「丟棄」不是
         「歸零」——歸零會在測試集尾端注入假陰性，因為時間序列的測試集
         就在尾端，這正是最不該被污染的地方。
      R3 網格共用設限：給 min_pct_floor 時，改用門檻 min_pct_floor 去找
         尾段。因為 y_pct 對 min_pct 單調（p1<p2 ⟹ y_p1 >= y_p2 逐元素），
         用網格最小的門檻設限一次即可支配所有更嚴格的門檻，讓整個參數
         網格共用同一組有效列、同一份 scaler、同一組切分，比較才公平。

    回傳
    ----
    targets : np.ndarray[int]，取中間後的稀疏訓練標籤
    valid   : np.ndarray[bool]
    fwd_ret : np.ndarray[float]
    """
    targets_raw, valid, fwd_ret = target_pct_threshold(df, lookahead, min_pct)

    # 只在 valid 區間上做連續段判定——尾端無效列不能參與「連續」的認定
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return np.zeros(len(df), dtype=int), valid, fwd_ret
    lo, hi = idx[0], idx[-1] + 1
    y_valid = targets_raw[lo:hi]

    # R2/R3：決定尾端要再砍掉哪些列
    floor_pct = min_pct if min_pct_floor is None else min_pct_floor
    y_floor = np.zeros(hi - lo, dtype=int)
    fr = fwd_ret[lo:hi]
    y_floor[~np.isnan(fr)] = (fr[~np.isnan(fr)] >= floor_pct).astype(int)
    cut = trailing_run_start(y_floor)

    valid = valid.copy()
    if cut < len(y_floor):
        valid[lo + cut:hi] = False

    z_valid = run_center_transform(y_valid, min_run)
    if cut < len(y_floor):
        z_valid[cut:] = 0

    targets = np.zeros(len(df), dtype=int)
    targets[lo:hi] = z_valid
    targets[~valid] = 0

    return targets, valid, fwd_ret


def target_long(df, lookahead=96, tp_pct=0.06, sl_pct=0.02):
    # 將需要運算的欄位轉換為純 NumPy 陣列以最大化效能
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)

    # 建立預設為 0 的 target 陣列
    targets = np.zeros(n, dtype=int)

    # 遍歷每一根 K 線
    for i in range(n):
        base_price = closes[i]

        # 依照當前 Close 設定止贏與止損價位
        target_up = base_price * (1 + tp_pct)
        target_down = base_price * (1 - sl_pct)

        # 確保不會超出資料邊界
        end_idx = min(n, i + 1 + lookahead)

        # 往未來 96 根檢查
        for j in range(i + 1, end_idx):
            # 保守原則：若同一根 K 線同時觸及止盈與止損，視為打止損 (0)
            if lows[j] <= target_down and highs[j] >= target_up:
                break
            # 先打止損 (標記 0 並結束此輪檢查)
            elif lows[j] <= target_down:
                break
            # 先達止贏 (標記 1 並結束此輪檢查)
            elif highs[j] >= target_up:
                targets[i] = 1
                break

    return targets

def target_short(df, lookahead=96, tp_pct=0.06, sl_pct=0.02):
    """
    計算做空 (Short) 用的目標標籤。
    - 標記 1: 優先觸及止盈價 (向下 tp_pct)
    - 標記 0: 優先觸及止損價 (向上 sl_pct) 或是兩者同時觸發(保守看待)，或是最終都未觸及
    """
    # 將需要運算的欄位轉換為純 NumPy 陣列以最大化效能
    # (已將 df_btc 修正為使用傳入的 df 變數)
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)

    # 建立預設為 0 的 target 陣列
    targets = np.zeros(n, dtype=int)

    # 遍歷每一根 K 線
    for i in range(n):
        base_price = closes[i]

        # 依照做空邏輯，當前 Close 往『下』是止贏，往『上』是止損
        target_down = base_price * (1 - tp_pct) # 止盈價 (跌下去才賺錢)
        target_up = base_price * (1 + sl_pct)   # 止損價 (漲上去就虧錢)

        # 確保不會超出資料邊界
        end_idx = min(n, i + 1 + lookahead)

        # 往未來 lookahead 根檢查
        for j in range(i + 1, end_idx):
            # 保守原則：若同一根 K 線內最高價碰到止損、最低價也碰到止盈，視為先打掉止損 (0)
            if highs[j] >= target_up and lows[j] <= target_down:
                break
            # 先打止損 (標記 0 並結束此輪檢查)
            elif highs[j] >= target_up:
                break
            # 先達止盈 (標記 1 並結束此輪檢查)
            elif lows[j] <= target_down:
                targets[i] = 1
                break

    return targets
