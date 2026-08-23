"""
Round 5 標籤驗證：連續段取中間轉換、幅度門檻、尾端設限、標籤/合約分離。

取中間轉換是本輪唯一的新演算法，必須在任何資料流程跑之前單獨驗證通過。
沿用 toolbywi/test_causality.py 的慣例：module-level test_* 函式 + 裸 assert，
不引入 pytest。

用法：
    .venv/bin/python toolbywi/test_labels.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from target_calulate import (run_center_transform, trailing_run_start,
                              target_pct_threshold, target_run_center, target_direction)


def _s(arr):
    """0/1 陣列轉字串，方便肉眼比對測試案例。"""
    return ''.join(map(str, np.asarray(arr, dtype=int)))


def _parse(s):
    return np.array([int(c) for c in s], dtype=int)


# ── T1：手算表（使用者提供的兩個原例必須逐位元符合）──────────────
def test_run_center_exact_examples():
    cases = [
        # (輸入, min_run, 期望輸出, 說明)
        ('0011111010', 5, '0000100000', '使用者原例1：段長5（奇數）取中間1根，孤立的1不合格歸0'),
        ('011111100',  5, '000110000',  '使用者原例2：段長6（偶數）取中間2根'),
        ('111',        3, '010',        '奇數最小合格段'),
        ('1111',       3, '0110',       '偶數段取2根（釘住規則）'),
        ('11111',      3, '00100',      ''),
        ('111111',     3, '001100',     ''),
        ('11',         3, '00',         '長度不足全歸0'),
        ('1110',       3, '0100',       '段落貼齊陣列開頭'),
        ('0111',       3, '0010',       '段落貼齊陣列結尾'),
        ('1110111',    3, '0100010',    '兩段各自獨立處理'),
        ('0000000000', 5, '0000000000', '全0'),
        ('1111111111', 5, '0000110000', '全1（長度10，偶數）→ 取 index 4,5'),
        ('111111111',  5, '000010000',  '全1（長度9，奇數）→ 取 index 4'),
        ('0101010',    1, '0101010',    'min_run=1 且全是長度1的段 → 恆等'),
    ]
    for src, mr, expected, note in cases:
        got = _s(run_center_transform(_parse(src), mr))
        assert got == expected, (
            f"取中間轉換錯誤：{src} (min_run={mr})\n"
            f"  得到 {got}\n  期望 {expected}\n  {note}"
        )
    print(f"✅ T1 取中間手算表通過（{len(cases)} 個案例，含使用者提供的兩個原例）")


# ── T2：不變量（隨機測試）────────────────────────────────────
def _maximal_runs(y):
    out, s = [], None
    for i, v in enumerate(np.append(np.asarray(y, dtype=int), 0)):
        if v == 1 and s is None:
            s = i
        elif v != 1 and s is not None:
            out.append((s, i - 1))
            s = None
    return out


def test_run_center_invariants():
    rng = np.random.default_rng(20260823)
    for trial in range(1000):
        n = int(rng.integers(1, 120))
        # 偏向產生較長的連續段，否則隨機序列幾乎都是長度1
        p = rng.uniform(0.2, 0.85)
        y = (rng.random(n) < p).astype(int)
        mr = int(rng.integers(1, 9))
        z = run_center_transform(y, mr)

        assert len(z) == len(y), '長度必須一致'
        assert set(np.unique(z)).issubset({0, 1}), '輸出必須是 0/1'
        assert np.all(z <= y), 'z[i]=1 必須蘊含 y[i]=1（不能無中生有）'

        runs = _maximal_runs(y)
        expected_total = 0
        for s, e in runs:
            L = e - s + 1
            seg = z[s:e + 1]
            if L >= mr:
                k = 1 if L % 2 == 1 else 2
                expected_total += k
                assert seg.sum() == k, (
                    f'trial={trial} 段[{s},{e}] 長度{L} 應有 {k} 個 1，實得 {seg.sum()}')
                ones = np.flatnonzero(seg)
                if k == 2:
                    assert ones[1] - ones[0] == 1, '偶數段的兩個 1 必須相鄰'
                # 對稱性：兩側剩餘長度相差不超過 1
                assert abs(ones[0] - (L - 1 - ones[-1])) <= 1, '必須置中'
            else:
                assert seg.sum() == 0, f'trial={trial} 不合格段 [{s},{e}] 必須全 0'
        assert z.sum() == expected_total, '總數必須等於各合格段的貢獻和'
    print('✅ T2 取中間不變量通過（1000 組隨機序列）')


# ── T3：雙實作比對（結構刻意不同）────────────────────────────
def _run_center_reference(y, min_run):
    """
    獨立重寫版：用 np.diff 找邊界，而非雙指標掃描。
    刻意不共用任何程式碼——沿用 event_common.py 的
    _cluster_entries_reference 慣例，兩份實作逐位元相等才算可信。
    """
    y = np.asarray(y, dtype=int)
    z = np.zeros_like(y)
    if len(y) == 0:
        return z
    padded = np.concatenate(([0], y, [0]))
    d = np.diff(padded)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1) - 1
    for s, e in zip(starts, ends):
        L = e - s + 1
        if L < min_run:
            continue
        if L & 1:
            z[s + (L - 1) // 2] = 1
        else:
            z[s + L // 2 - 1] = 1
            z[s + L // 2] = 1
    return z


def test_run_center_two_implementations():
    rng = np.random.default_rng(77)
    for _ in range(1000):
        n = int(rng.integers(0, 150))
        y = (rng.random(n) < rng.uniform(0.1, 0.9)).astype(int)
        mr = int(rng.integers(1, 10))
        a = run_center_transform(y, mr)
        b = _run_center_reference(y, mr)
        assert np.array_equal(a, b), f'兩份實作不一致\n  y={_s(y)}\n  min_run={mr}'
    print('✅ T3 雙實作比對通過（1000 組隨機序列，np.diff 版 vs 雙指標版）')


def test_trailing_run_start():
    assert trailing_run_start([0, 0, 0]) == 3, '沒有尾段時應回傳 len'
    assert trailing_run_start([1, 1, 1]) == 0, '全 1 時尾段從 0 開始'
    assert trailing_run_start([0, 1, 0, 1, 1]) == 3
    assert trailing_run_start([]) == 0
    assert trailing_run_start([1, 0]) == 2, '最後一格是 0 → 無尾段'
    print('✅ trailing_run_start 通過')


def _synth_close(n=400, seed=0, tail_rally_bars=0, H=12, pct=0.01):
    """合成收盤價序列；tail_rally_bars>0 時讓尾端強制形成一段未結束的漲勢。"""
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.05, n))
    if tail_rally_bars > 0:
        # 讓最後 tail_rally_bars 根的 fwd_ret 都遠超過門檻：
        # fwd_ret[i] = close[i+H]/close[i]-1，要讓 i 從 n-H-tail_rally_bars 起都成立
        start = n - H - tail_rally_bars
        for i in range(start, n):
            close[i] = close[start] * (1 + pct * 3 * (i - start + 1) / max(1, H))
    return pd.DataFrame({'Close': close})


# ── T4：尾端設限 ─────────────────────────────────────────────
def test_tail_censoring():
    H, K = 12, 20
    df = _synth_close(n=400, seed=3, tail_rally_bars=K, H=H, pct=0.01)

    y_raw, valid_h, fwd = target_pct_threshold(df, lookahead=H, min_pct=0.003)
    assert (~valid_h[-H:]).all(), 'R1：最後 lookahead 根必須 valid=False'
    assert valid_h[:-H].all(), 'R1：其餘必須全部 valid'
    assert y_raw[-1] == 0, '無效區的 target 必須是 0'

    tgt, valid, fwd2 = target_run_center(df, lookahead=H, min_pct=0.003, min_run=3)
    n = len(df)
    assert (~valid[-H:]).all(), 'R1 仍必須成立'

    # R2：被截斷的尾段必須整段作廢，且作廢區內不得有任何 target=1
    censored = ~valid
    assert tgt[censored].sum() == 0, 'R2：設限區內不得有任何 target=1'
    n_censored_extra = int((~valid).sum() - H)
    assert n_censored_extra > 0, (
        f'測試設計問題：合成資料應該要造成尾端設限，實際額外砍掉 {n_censored_extra} 根')

    # R3 單調性：較寬鬆門檻的設限區必須包含較嚴格門檻的設限區
    _, v_lo, _ = target_run_center(df, lookahead=H, min_pct=0.001, min_run=3,
                                    min_pct_floor=0.001)
    _, v_hi, _ = target_run_center(df, lookahead=H, min_pct=0.005, min_run=3,
                                    min_pct_floor=0.001)
    assert np.array_equal(v_lo, v_hi), (
        'R3：給定同一個 min_pct_floor，所有 min_pct 必須共用完全相同的 valid 遮罩，'
        '否則 25 組網格的切分會不一致、無法公平比較')

    # 尾段有正常結束時，不應額外砍任何列
    df2 = _synth_close(n=400, seed=5, tail_rally_bars=0)
    _, valid2, _ = target_run_center(df2, lookahead=H, min_pct=0.003, min_run=3)
    assert int((~valid2).sum()) == H, (
        f'尾段已正常結束時只該砍 R1 的 {H} 根，實際砍了 {int((~valid2).sum())} 根')
    print('✅ T4 尾端設限通過（R1 視野尾端／R2 截斷尾段整段作廢／R3 網格共用遮罩）')


# ── T5：標籤與合約分離 ───────────────────────────────────────
def test_label_contract_separation():
    H = 12
    df = _synth_close(n=3000, seed=11)

    tgt, valid, fwd = target_run_center(df, lookahead=H, min_pct=0.003, min_run=5)
    contract_ref, valid_dir = target_direction(df, lookahead=H)
    contract = np.zeros(len(df), dtype=int)
    contract[valid] = (fwd[valid] > 0).astype(int)

    # (a) contract_target 必須與既有 target_direction 在有效區完全一致
    both = valid & valid_dir
    assert np.array_equal(contract[both], contract_ref[both]), (
        'contract_target 必須與 target_direction 的定義完全相同')

    # (b) 訓練標籤=1 必然蘊含合約獲勝（漲幅 >= 0.3% 必然 > 0%）
    pos = (tgt == 1)
    if pos.sum() > 0:
        assert contract[pos].all(), 'target=1 必須 100% 蘊含 contract_target=1'
        assert (fwd[pos] >= 0.003 - 1e-12).all(), 'target=1 的漲幅必須都達到門檻'

    # (c) 稀疏度：訓練標籤必須遠低於合約正樣本率
    if pos.sum() > 0:
        assert tgt[valid].mean() < contract[valid].mean() / 3, (
            f'訓練標籤 {tgt[valid].mean():.4f} 應遠低於合約 {contract[valid].mean():.4f}')
    print('✅ T5 標籤/合約分離通過（定義一致、target=1 蘊含合約獲勝、稀疏度正確）')


# ── T7：切分對齊（對照 Dataset_Custom 本身，不是重推公式）─────────
def test_split_index_alignment():
    import argparse
    import json
    import shutil
    import tempfile

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data_provider.data_loader import Dataset_Custom
    from event_common import split_start_index

    n, seq_len, label_len, pred_len = 3000, 64, 8, 1
    tmp = tempfile.mkdtemp()
    try:
        # target 欄放位置追蹤器 0,1,2,...,n-1，這樣模型「預測目標」的值
        # 直接就是它在全域表中的列號，對齊錯了立刻看得出來
        dates = pd.date_range('2025-01-01', periods=n, freq='5min')
        df = pd.DataFrame({'date': dates.astype(str),
                            'f1': np.arange(n, dtype=float) * 0.001,
                            'target': np.arange(n, dtype=float)})
        df.to_csv(os.path.join(tmp, 'stock_features.csv'), index=False)

        num_train = int(n * 0.7)
        num_test = int(n * 0.2)
        num_vali = n - num_train - num_test
        meta = {'n_rows': n, 'num_train': num_train, 'num_vali': num_vali,
                'num_test': num_test, 'data_path': 'stock_features.csv',
                'feature_cols': ['f1']}

        for flag in ('train', 'val', 'test'):
            ds = Dataset_Custom(args=argparse.Namespace(augmentation_ratio=0), root_path=tmp,
                                 data_path='stock_features.csv', flag=flag,
                                 size=[seq_len, label_len, pred_len],
                                 features='MS', target='target', scale=False,
                                 timeenc=1, freq='t')
            start = split_start_index(meta, flag, len(ds))
            for j in (0, 1, len(ds) // 2, len(ds) - 1):
                _, seq_y, _, _ = ds[j]
                got = float(seq_y[-1, -1])
                want = float(start + j)
                assert got == want, (
                    f'切分對齊錯誤 flag={flag} j={j}: 模型看到的目標列號 {got}，'
                    f'split_start_index 推導出 {want}')
        print('✅ T7 切分對齊通過（train/val/test 逐樣本對照 Dataset_Custom 實際輸出）')
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_old_tail_alignment_was_broken():
    """把既有 bug 釘成測試：raw_ohlcv.csv 的 tail() 對齊確實不等於正確答案。
    這個測試存在的目的是防止有人「順手改回去」。"""
    from event_common import split_start_index
    d = 'dataset/event30m_v4'
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dd = os.path.join(repo, d)
    if not os.path.exists(os.path.join(dd, 'meta.json')):
        print('⏭  T7b 跳過（找不到 dataset/event30m_v4）')
        return
    meta = json.load(open(os.path.join(dd, 'meta.json')))
    n_test = meta['num_test']
    raw_tail = pd.read_csv(os.path.join(dd, 'raw_ohlcv.csv'),
                            usecols=['date'])['date'].tail(n_test).reset_index(drop=True)
    feat = pd.read_csv(os.path.join(dd, 'stock_features.csv'), usecols=['date'])['date']
    start = split_start_index(meta, 'test', n_test)
    correct = feat.iloc[start:start + n_test].reset_index(drop=True)
    match = (raw_tail.values == correct.values).mean()
    assert match < 0.5, (
        f'預期舊寫法應該是錯的，但符合率是 {match:.4f}——若資料結構改變讓兩者一致了，'
        f'請重新確認這個測試還有沒有意義')
    print(f'✅ T7b 已釘住既有 bug（raw_ohlcv.csv tail() 對齊符合率僅 {match*100:.2f}%，'
          f'正確做法是從 stock_features.csv 取）')


# ── T6/T8：特徵表 schema 與 labels.csv 對齊（對真實資料目錄）─────────
def _round5_dirs():
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = []
    for d in ('dataset/event1h_v5', 'dataset/event1h_v5_sel'):
        p = os.path.join(repo, d)
        if os.path.exists(os.path.join(p, 'meta.json')):
            out.append(p)
    return out


def test_feature_table_schema():
    from event_common import assert_feature_table_matches_meta
    dirs = _round5_dirs()
    if not dirs:
        print('⏭  T6 跳過（尚未產生 Round 5 資料目錄）')
        return
    for d in dirs:
        meta = json.load(open(os.path.join(d, 'meta.json')))
        assert_feature_table_matches_meta(d, meta)
        cols = list(pd.read_csv(os.path.join(d, meta['data_path']), nrows=0).columns)
        for bad in ('contract_target', 'fwd_ret'):
            assert bad not in cols, f'{d}: {bad} 混進特徵表 = 未來資料洩漏！'
        assert not [c for c in cols if c.startswith('y_bp')], \
            f'{d}: 網格標籤欄混進特徵表 = 未來資料洩漏！'
        print(f'  ✓ {os.path.basename(d)}: {len(cols)} 欄 = n_features({len(meta["feature_cols"])})+2')
    print('✅ T6 特徵表 schema 通過（無標籤欄洩漏成第 N+1 個模型輸入）')


def test_labels_csv_alignment():
    dirs = _round5_dirs()
    if not dirs:
        print('⏭  T8 跳過（尚未產生 Round 5 資料目錄）')
        return
    for d in dirs:
        meta = json.load(open(os.path.join(d, 'meta.json')))
        feat = pd.read_csv(os.path.join(d, meta['data_path']), usecols=['date', 'target'])
        lab = pd.read_csv(os.path.join(d, meta.get('labels_path', 'labels.csv')))
        assert len(feat) == len(lab) == meta['n_rows'], f'{d}: 長度不一致'
        assert (feat['date'].values == lab['date'].values).all(), f'{d}: date 不對齊'
        assert (feat['target'].values == lab['target'].values).all(), \
            f'{d}: labels.csv 的 target 快照與特徵表不一致'
        # target=1 必須蘊含 contract_target=1
        pos = lab['target'] == 1
        assert (lab.loc[pos, 'contract_target'] == 1).all(), \
            f'{d}: 有 target=1 但 contract_target=0，標籤定義被破壞'
        if meta.get('label_mode') == 'direction':
            assert (lab['contract_target'].values == lab['target'].values).all(), \
                f'{d}: direction 模式下兩者必須逐值相等'
    print('✅ T8 labels.csv 對齊通過（等長同序、target 蘊含 contract_target）')


if __name__ == '__main__':
    import json
    test_run_center_exact_examples()
    test_run_center_invariants()
    test_run_center_two_implementations()
    test_trailing_run_start()
    test_tail_censoring()
    test_label_contract_separation()
    test_split_index_alignment()
    test_old_tail_alignment_was_broken()
    test_feature_table_schema()
    test_labels_csv_alignment()
    print('\n✅ Round 5 標籤測試全部通過')
