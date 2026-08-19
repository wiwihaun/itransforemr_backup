#!/usr/bin/env python3
"""未來函數（look-ahead bias）靜態掃描器。

只用標準函式庫，不假設專案裝了什麼。

用法:
    python audit_lookahead.py <檔案或目錄> [...]
    python audit_lookahead.py . --severity high
    python audit_lookahead.py src/ --quiet          # 只印摘要

豁免：在該行加上 `# noqa: lookahead` 行內註解（可附理由）。
標籤建構這類刻意前看的程式碼應該用它，例如：

    future_high = df['High'].shift(-n)   # noqa: lookahead - 標籤建構，設計如此

退出碼：有未豁免的 high 命中時回傳 1，否則 0。可掛 pre-commit 或 CI。

重要限制：靜態掃描只涵蓋「特徵層」的洩漏（SKILL.md 的 L1，以及部分 L2）。
統計量洩漏（L3）與決策洩漏（L4）在程式碼上完全正常，掃不出來——那兩層
必須靠串流等價測試與流程紀律，見 SKILL.md。
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (嚴重度, 樣式名稱, 命中 regex, 排除 regex 或 None, 說明)
#
# 「排除」很重要：誤報率高的規則比沒有規則更糟——它會訓練使用者忽略輸出。
# 例如 rolling(14).min() 是合法的回看極值，不該跟 x / x.max() 這種
# 全序列正規化混為一談。
RULES: list[tuple[str, str, str, str | None, str]] = [
    # ── L1 特徵洩漏 ───────────────────────────────────────
    ('high', 'negative-shift', r'\.shift\(\s*-\s*\d', None,
     '負向位移直接取未來值'),
    ('high', 'negative-diff', r'\.diff\(\s*-\s*\d', None,
     '負向差分用到未來值'),
    ('high', 'centered-rolling', r'rolling\([^)]*center\s*=\s*True', None,
     '置中窗口有一半落在未來'),
    ('high', 'backfill', r'\.bfill\(|fillna\(\s*method\s*=\s*[\'"]b(fill|ackfill)', None,
     '用未來值回填缺值'),
    ('high', 'reversed-series', r'\.iloc\[\s*::\s*-\s*1\s*\]|\[::-1\][^#]*rolling', None,
     '反轉序列後做滾動運算等於前看'),
    # 結構性比對而非變數名比對——df['Close'] 這種下標寫法讓 \w+ 派不上用場
    ('high', 'minmax-full-series',
     r'\.min\(\)\s*\)\s*/\s*\(?[^)]*\.max\(\)',
     r'rolling\(|expanding\(|np\.maximum\.accumulate',
     '用全序列極值做 min-max 正規化——訓練時拿不到未來的極值'),
    ('medium', 'divide-by-full-stat',
     r'/\s*\(?\s*\w+(\[[^\]]*\])?\.(max|std)\(\)',
     r'rolling\(|expanding\(|np\.maximum\.accumulate|\.groupby\(',
     '除以全序列統計量——確認這是訓練集的統計量而非全體'),

    # ── 切分方式 ──────────────────────────────────────────
    ('high', 'random-split', r'train_test_split\((?![^)]*shuffle\s*=\s*False)', None,
     '時間序列不可隨機切分，需 shuffle=False 或手動依時間切'),
    ('high', 'kfold-on-timeseries', r'\b(KFold|StratifiedKFold)\s*\(', None,
     '時序資料應改用 TimeSeriesSplit；若對象是非時序的 pointwise 轉換'
     '（如機率校準器）請豁免並註明理由'),

    # ── L3 統計量洩漏 ─────────────────────────────────────
    ('high', 'threshold-from-test',
     r'(percentile|quantile)\s*\([^)]*\b(test|tst|holdout|oos)\w*', None,
     '門檻用測試集分佈計算——實時拿不到未來的分佈'),
    ('high', 'threshold-from-pred',
     r'(np\.percentile|\.quantile)\s*\(\s*(prob|pred|score|proba|logit)\w*', None,
     '門檻用預測結果的分佈計算——確認這份預測來自驗證集而非測試集'),
    ('medium', 'stat-on-test',
     r'\b(test|tst|holdout|oos)\w*\.(mean|std|median|var|quantile)\(', None,
     '對測試集取統計量——確認只用於描述輸出而非任何決策'),
    ('medium', 'fit-on-full',
     r'\.fit(_transform)?\(\s*(df|data|X|features)\s*[,)]',
     r'X_?tr|X_?train|train_|\[tr\]|\[train\]',
     'scaler/編碼器在完整資料上 fit——應只用訓練切片'),
    ('medium', 'scaler-fit-transform',
     r'(Scaler|Encoder|Imputer)\([^)]*\)\.fit_transform\(', None,
     'fit_transform 一次做完會把全體統計量帶進來'),
]

NOQA = re.compile(r'#\s*noqa\s*:\s*lookahead', re.I)
SKIP_DIRS = {'.git', '.venv', 'venv', 'node_modules', '__pycache__',
             '.mypy_cache', '.pytest_cache', 'site-packages'}
ORDER = {'high': 0, 'medium': 1}
COLOR = {'high': '\033[31m', 'medium': '\033[33m'}
RESET = '\033[0m'


def scan_file(path: Path) -> list[tuple[int, str, str, str, str]]:
    """回傳 [(行號, 嚴重度, 樣式, 說明, 該行原文)]"""
    try:
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
    except OSError:
        return []
    hits = []
    for i, line in enumerate(lines, 1):
        stripped = line.lstrip()
        if stripped.startswith('#'):          # 純註解行不掃
            continue
        if NOQA.search(line):                 # 明示豁免
            continue
        for sev, name, pattern, unless, msg in RULES:
            if re.search(pattern, line):
                if unless and re.search(unless, line):
                    continue          # 該行有排除樣式，屬合法用法
                hits.append((i, sev, name, msg, line.strip()))

    # 同一行若已有 high 命中，就不再報該行的 medium——通常是同一個問題
    # 被兩條規則抓到，重複列出只會稀釋訊號。
    high_lines = {h[0] for h in hits if h[1] == 'high'}
    return [h for h in hits if h[1] == 'high' or h[0] not in high_lines]


def iter_targets(paths: list[str]):
    for p in paths:
        path = Path(p)
        if path.is_file():
            yield path
        elif path.is_dir():
            for f in sorted(path.rglob('*.py')):
                if not any(part in SKIP_DIRS for part in f.parts):
                    yield f
        else:
            print(f'找不到路徑：{p}', file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description='未來函數靜態掃描器')
    ap.add_argument('paths', nargs='+', help='要掃描的檔案或目錄')
    ap.add_argument('--severity', choices=['high', 'all'], default='all',
                    help='只顯示 high，或全部顯示（預設 all）')
    ap.add_argument('--quiet', action='store_true', help='只印摘要')
    ap.add_argument('--no-color', action='store_true')
    a = ap.parse_args()

    use_color = sys.stdout.isatty() and not a.no_color
    total = {'high': 0, 'medium': 0}
    files_scanned = 0

    for path in iter_targets(a.paths):
        files_scanned += 1
        hits = [h for h in scan_file(path)
                if a.severity == 'all' or h[1] == 'high']
        if not hits:
            continue
        hits.sort(key=lambda h: (ORDER[h[1]], h[0]))
        if not a.quiet:
            print(f'\n{path}')
        for lineno, sev, name, msg, src in hits:
            total[sev] += 1
            if a.quiet:
                continue
            tag = f'{COLOR[sev]}{sev:>6}{RESET}' if use_color else f'{sev:>6}'
            print(f'  {lineno:>5}  {tag}  {name:<22} {msg}')
            print(f'         {src[:110]}')

    print(f'\n掃描 {files_scanned} 個檔案：'
          f'high {total["high"]} 處、medium {total["medium"]} 處')
    if total['high'] or total['medium']:
        print('豁免方式：在該行加 `# noqa: lookahead - 理由`（標籤建構等刻意前看的程式碼）')
    print('\n涵蓋範圍：只掃得到「向量化寫法」的特徵層洩漏（L1）與部分統計量洩漏（L3）。')
    print('掃不到的有三類——')
    print('  1. 手寫迴圈的前看（如 for j in range(i+1, i+N) 取未來 K 棒）')
    print('  2. 統計量跨切分（變數命名看不出是訓練還是測試集時）')
    print('  3. 決策洩漏（看過測試集結果才選門檻/模型/特徵，程式碼完全正常）')
    print('這三類要靠串流等價測試與流程紀律，見 SKILL.md。')
    return 1 if total['high'] else 0


if __name__ == '__main__':
    sys.exit(main())
