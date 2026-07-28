"""查找并清理小说中的网站水印"""
import sys
sys.path.insert(0, 'src')
import re
from database import get_db_path, load_chunks, save_chunks, load_chapters, save_chapters
from pathlib import Path

novel_dir = Path("novels/天龙八部（世纪新修版）")

# 读取原文（GBK）
text = open(novel_dir / "original.txt", encoding='gb18030', errors='replace').read()

# 查找所有 99lib 相关模式
patterns = re.findall(r'.{0,20}99lib.{0,20}', text)
print(f"原文中 '99lib' 出现 {len(patterns)} 次")
for p in patterns[:10]:
    print(f"  {repr(p)}")

# 查找其他可能的垃圾标记
junk_patterns = [
    r'<q>.*?</q>',
    r'99lib\.\w+',
    r'www\.\w+\.\w+',
    r'\w+\.net',
    r'\w+\.com',
]
print("\n--- 垃圾模式统计 ---")
for jp in junk_patterns:
    found = re.findall(jp, text)
    if found:
        print(f"  {jp}: {len(found)} 处")
        for f in found[:3]:
            print(f"    {repr(f)}")
