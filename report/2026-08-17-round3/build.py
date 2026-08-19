"""source.html (UTF-8, 可編輯) -> btc_event30m_round2.html (純 ASCII, 可安全發布)

為什麼要轉成純 ASCII：Artifact 的包裝層不保證帶 charset header，
瀏覽器猜錯編碼時中文會變亂碼。轉成數字實體參照後，任何編碼假設下都正確。
CSS 不解析 HTML 實體，所以 <style> 區塊必須原封不動、且本身不能有非 ASCII。
"""
import re
import sys

SRC, OUT = 'source.html', 'btc_event30m_round3.html'
src = open(SRC, encoding='utf-8').read()

for m in re.finditer(r'<style>.*?</style>', src, re.S):
    bad = sorted({c for c in m.group(0) if ord(c) > 127})
    if bad:
        sys.exit(f'錯誤：<style> 內有非 ASCII 字元 {bad}，CSS 不解析實體，請改用 ASCII')

parts, last = [], 0
for m in re.finditer(r'<style>.*?</style>', src, re.S):
    parts.append(('esc', src[last:m.start()]))
    parts.append(('raw', m.group(0)))
    last = m.end()
parts.append(('esc', src[last:]))

esc = lambda t: ''.join(c if ord(c) < 128 else f'&#x{ord(c):X};' for c in t)
out = '<meta charset="utf-8">\n' + ''.join(esc(t) if k == 'esc' else t for k, t in parts)

assert all(ord(c) < 128 for c in out), '仍有非 ASCII 殘留'

# SVG 結構檢查
import xml.etree.ElementTree as ET
for i, m in enumerate(re.finditer(r'<svg.*?</svg>', out, re.S), 1):
    ET.fromstring(m.group(0).replace('class="mono"', 'data-c="mono"'))

open(OUT, 'w', encoding='ascii').write(out)
print(f'{OUT}: {len(out)} bytes, 純 ASCII, SVG x{i} 結構完整')
