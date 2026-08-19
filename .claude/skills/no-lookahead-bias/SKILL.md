---
name: no-lookahead-bias
description: Prevents and audits look-ahead bias (future-data leakage) in time-series ML, forecasting, and backtesting pipelines. Use this skill whenever the work involves chronological train/test splits, rolling or expanding windows, feature engineering from price/sensor/event history, fitting scalers or probability calibrators, selecting features or hyperparameters or decision thresholds, walk-forward validation, or evaluating a trading strategy or forecast — and especially when results look suspiciously good, when a metric flips sign after a small parameter change, or when someone asks "is this backtest realistic?". Also use when reviewing or debugging existing pipeline code for future functions, even if the user does not say "look-ahead" or "leakage" by name.
---

# 訓練資料不能偷看未來數據

## 為什麼這件事沒有折衷空間

洩漏不是讓數字「偏樂觀」，是讓數字**失去意義**。

一個有洩漏的 0.85 AUC 不代表「實際大概 0.7」，它代表「不知道」——因為你不知道模型有多少比例的表現來自那個它實際上拿不到的資訊。所以發現洩漏時唯一正確的反應是**修好後重跑**，不是打個折扣沿用，也不是在報告裡加註「可能略微高估」。

這也是為什麼要在寫 pipeline 的當下就防，而不是等結果出來再回頭查：等到數字漂亮了，人會傾向替它找理由。

---

## 洩漏的四個層次

由淺入深、由易查到難查。前兩層靜態掃描抓得到，後兩層不行。

| 層次 | 內容 | 為什麼容易漏掉 |
|---|---|---|
| **L1 特徵洩漏** | 時點 t 的特徵用到 t 之後的資料 | 最好查，靜態掃描就能抓 |
| **L2 標籤洩漏** | 標籤刻意看未來（正常），但尾端窗口不完整必須剔除；標籤重疊造成樣本不獨立 | 「標籤本來就看未來」讓人放鬆警覺 |
| **L3 統計量洩漏** | scaler / 校準器 / 門檻 / 特徵選擇用到模型當下拿不到的資料 | 程式碼看起來完全正常 |
| **L4 決策洩漏** | 看過測試集結果後才決定用哪個模型、哪個門檻、哪組特徵 | 沒有任何一行程式碼有錯，錯在流程 |

### L1 — 特徵洩漏

特徵在時點 t 的值，只能由 t 及之前的資料計算。危險樣式：

- `shift(-n)`、`diff(-n)` — 負向位移直接取未來值
- `rolling(n, center=True)` — 置中窗口有一半在未來
- `.bfill()`、`fillna(method='bfill')` — 用未來值回填
- 對整段序列取 `.max()` / `.min()` / `.mean()` 拿來做正規化
- `df.iloc[::-1]` 之後再做 rolling — 反向等於前看

多數技術指標（EMA、RSI、ATR、rolling mean）預設就是回看的，問題通常出在自己寫的那幾行。

### L2 — 標籤洩漏

標籤看未來是**設計**，不是 bug。真正要處理的是兩件事：

**尾端剔除。** 若標籤用未來 N 根計算，最後 N 筆的窗口不完整，結果會被系統性地偏向某一類。時間序列的測試集就在尾端，所以這些壞標籤全部落在最該乾淨的地方。

```python
df['target'] = make_label(df, lookahead=N)
df = df.iloc[:-N].reset_index(drop=True)   # 沒有這行，測試集就毀了
```

**重疊造成的假樣本數。** lookahead=N 時，相鄰兩筆的未來窗口重疊 (N−1)/N。這不是洩漏，但會讓「1000 筆測試樣本」實際上只等於幾十個獨立觀測，所有統計檢定的檢定力都被高估。查法是算相鄰標籤的相同比例，跟隨機基準比。

順帶一提：如果標籤欄同時被當成輸入特徵餵進模型（多變量框架常見），那才是真的洩漏——模型直接抄上一筆標籤就能答對。輸入窗口裡的標籤欄要遮蔽掉。

### L3 — 統計量洩漏

任何「從資料估出來的參數」都必須問：**模型在做這個決策的當下，真的拿得到這個統計量嗎？**

| 東西 | 只能用什麼資料擬合 |
|---|---|
| StandardScaler / MinMaxScaler | 訓練集 |
| 機率校準器（Platt / isotonic / temperature） | 驗證集 |
| 決策門檻 | 驗證集，且在進入測試期前就固定 |
| 特徵選擇 | 訓練集 + 驗證集 |
| 缺值填補的統計量 | 訓練集 |

最常見的一種是門檻：

```python
# ✗ 用了整個測試期的分佈——實時在第 k 根時不可能知道
th = np.percentile(test_prob, 80)

# ✓ 訓練結束時就從驗證集固定下來
th = np.percentile(val_prob, 80)
```

### L4 — 決策洩漏

這是最常見、也最難自覺的一種。沒有任何一行程式碼有錯，錯在流程：你比較了 N 種設定在測試集上的表現，然後選了最好的那個。

判準只有一句：**如果我沒看過測試集結果，我還會做同樣的選擇嗎？**

具體形態：

- 掃 10 個門檻，報告最好的那個
- 比較 3 種模型架構，選測試 AUC 最高的
- 試了 4 種進場規則，選賺最多的
- 「我只是看了一眼測試集，沒有真的用它調參」——看了就是用了

代價可以量化。比較 12 種設定時，即使真實效果為零，最小 p 值的期望值大約是 1/(12+1) ≈ 0.077。所以在這種情況下拿到 p=0.062 **完全不構成證據**。

補救方式不是假裝沒發生，而是：
1. 事前宣告操作點（在看結果之前就決定用哪個門檻）
2. 或保留一份從未動過的最終測試集
3. 或誠實報告「這個數字是從 N 種設定中選出來的」並據此打折看待

---

## 靜態稽核

跑掃描器：

```bash
python .claude/skills/no-lookahead-bias/scripts/audit_lookahead.py <檔案或目錄>
```

它會輸出 `檔案:行號  嚴重度  樣式  說明`，有高嚴重度命中時 exit code 非 0（可掛 pre-commit）。刻意前看的程式碼（標籤建構）用行內註解豁免：

```python
future_high = df['High'].shift(-n)   # noqa: lookahead - 標籤建構，設計如此
```

**掃描器只涵蓋 L1 和部分 L2。** 乾淨的掃描結果不代表沒有洩漏——L3 和 L4 必須靠下一節的方法。

---

## 決定性驗證：串流等價測試

這是這份 skill 裡最有價值的一招。靜態掃描抓不到 L3/L4，人的直覺也不可靠——看起來可疑的常常是安全的，看起來正常的常常有問題。

方法：**重寫一份嚴格串流的版本**——逐個時點推進，每個決策只能讀當下與過去——然後 assert 它與向量化版本逐筆相同。

```python
def stream_version(threshold, gap=6):
    """嚴格串流：逐個時點推進，每個決策只用當下與過去的資訊。

    threshold 必須是進入測試期前就固定的常數，不能是測試集的統計量。
    """
    out, busy, last_signal = [], -1, None
    for j in range(len(prob)):          # 只能往前走，不能回頭改
        t = index[j]
        if prob[j] < threshold:
            continue
        # 「距離上一個訊號超過 gap」——純粹回看，實時可算
        is_new = (last_signal is None) or (t - last_signal > gap)
        last_signal = t                 # 不論有沒有採用都要更新狀態
        if not is_new or t <= busy:
            continue
        exit_t, result = simulate(t)    # 模擬「會發生什麼」，不是做決策
        out.append((t, exit_t, result))
        busy = exit_t
    return out

assert stream_version(th) == vectorized_version(th), '兩者不一致，決策邏輯有未來函數'
```

把這個 assert 留在程式裡當迴歸測試。它幾乎不花時間，卻能擋住日後任何人把未來函數改回去。

三個要點：

1. **只維護實時取得的狀態。** 上面的 `last_signal`、`busy` 都是純粹的過去狀態。如果你發現需要「先掃過全部資料才能算出來的東西」，那就是洩漏。
2. **狀態更新不能有條件遺漏。** `last_signal = t` 必須對每個訊號都執行，即使那個訊號因為別的原因被跳過——否則狀態機的行為會跟實時不同。
3. **模擬未來 ≠ 洩漏。** `simulate(t)` 往前掃找出場點是正常的，它在回答「如果那時進場會怎樣」，不是在替進場做決策。

### 反例：先驗證再下結論

這個專案曾經懷疑「叢集第一根」的進場規則是未來函數——直覺上「要知道這是叢集的第一根，不是得先看到整個叢集嗎？」

寫了串流版本驗證後發現：**不是**。「叢集第一根」的定義是「前 6 根內沒有訊號」，純粹回看。串流版與向量版逐筆完全相同 33 筆。

真正的洩漏在同一份程式碼的另一個地方——門檻用了測試集百分位。

教訓：可疑的地方要驗證，不要憑感覺改。憑感覺改會同時做錯兩件事——改掉沒問題的、放過有問題的。

---

## 常見誤判

過度反應會讓 pipeline 變得又慢又難懂，而且會掩蓋真正的問題。以下都**不是**洩漏：

- **回測模擬中向前掃描找出場點。** 那是在模擬「會發生什麼」，不是在做決策。
- **標籤刻意看未來。** 那是設計。要處理的是尾端剔除，不是不准看。
- **用整段資料算敘述統計拿來印出來看。** 只要不餵進任何決策就沒問題。報告裡寫「測試集機率範圍 [0.19, 0.42]」是描述，不是決策。
- **在訓練集上用未來資訊。** 訓練集本來就是全部給模型看的。要管的是「訓練集的邊界有沒有伸進驗證/測試集」。

判準始終是同一個：**這個值有沒有影響到一個決策？如果有，做這個決策的當下拿得到它嗎？**

---

## 報告紀律

如果測試集已經被污染（例如比較過 N 種設定），照實說明，並且區分兩種發現：

- **結構性改善**——有機制解釋、不是用目標指標挑出來的。例如「改掉重複進場後最大回撤從 −21.6% 降到 −11.5%，因為不再反覆撞同一個市場狀態，且在三個門檻上都成立」。這類發現比較可信。
- **報酬數字**——可能是選出來的。要標明「這是從 N 種設定中選出的最佳值，不能當作預期報酬」。

可以直接用的句型：

> 這些數字是在同一個測試集上比較了 N 種設定之後才選定的，**這個選擇本身就用掉了測試集的資訊**，所以不能當作已驗證的預期表現。比較可信的是 X 的改善——那是結構性後果，不是用報酬率挑出來的參數。要驗證，得等 walk-forward 或全新的樣本外資料。

還有一種要特別小心的自我欺騙：**換一個選擇規則直到得到想要的答案**。這個專案曾經為了選校準器連續換了三種規則（驗證集 Brier → 交叉驗證 Brier → AUC 守門），三種給出三種答案。正確做法是採用一個有文獻依據、事前就能說明理由的規則（例如 one-standard-error rule：交叉驗證分數在最佳者的 1 個標準誤內視為打平，此時選較簡單的模型），然後接受它給出的答案。

---

## 檢查清單

寫完 pipeline 後逐項確認：

- [ ] 切分是依時間順序的，沒有 shuffle（`train_test_split(..., shuffle=False)` 或手動切）
- [ ] 交叉驗證用 `TimeSeriesSplit` 而非 `KFold`
- [ ] scaler / 填補統計量只在訓練集 fit
- [ ] 標籤尾端 lookahead 筆已剔除
- [ ] 標籤欄若同時是輸入特徵，輸入窗口裡已遮蔽
- [ ] 特徵選擇只用訓練 + 驗證
- [ ] 校準器只在驗證集 fit
- [ ] 決策門檻在進入測試期前就固定
- [ ] 掃描器跑過，高嚴重度命中都已處理或豁免
- [ ] 關鍵決策邏輯有串流等價測試，且 assert 留在程式裡
- [ ] 若比較過多組設定，報告裡已據實說明

深入案例（8 個真實 bug 的症狀、成因、修正前後程式碼與影響量級）見 `references/case-studies.md`。遇到具體情境拿不準時去查那份，裡面每個案例都有實測數字。
