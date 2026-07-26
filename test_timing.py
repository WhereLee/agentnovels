"""计时测试：各步骤耗时"""
import sys, time
sys.path.insert(0, 'src')
from pathlib import Path
from encoding_detector import read_file_auto_encoding
from text_cleaner import clean_text
from chapter_detector import detect_chapters
from chunker import chunk_chapters, chunk_chapters_sentence

# 找到新上传的小说
novels_dir = Path("novels")
target = None
for d in novels_dir.iterdir():
    if d.is_dir() and (d / "original.txt").exists():
        meta = d / "meta.json"
        if meta.exists():
            import json
            m = json.loads(meta.read_text(encoding='utf-8'))
            if m.get("total_chars", 0) > 5000000:
                target = d
                break

if not target:
    print("未找到540万字小说")
    sys.exit(1)

print(f"目标: {target.name}")
print("=" * 50)

# 1. 编码检测 + 读取
t0 = time.perf_counter()
text, enc, conf = read_file_auto_encoding(str(target / "original.txt"))
t1 = time.perf_counter()
print(f"编码检测+读取: {t1-t0:.3f}s ({len(text):,} 字)")

# 2. 文本清洗
t2 = time.perf_counter()
text, stats = clean_text(text)
t3 = time.perf_counter()
print(f"文本清洗:     {t3-t2:.3f}s (清除 {stats['total']} 处)")

# 3. 章节检测
t4 = time.perf_counter()
result = detect_chapters(text)
t5 = time.perf_counter()
print(f"章节检测:     {t5-t4:.3f}s ({len(result.chapters)} 章)")

# 4. fixed 切块
t6 = time.perf_counter()
fixed = chunk_chapters(result.chapters)
t7 = time.perf_counter()
print(f"fixed切块:    {t7-t6:.3f}s ({len(fixed)} 块)")

# 5. sentence 切块
t8 = time.perf_counter()
sentence = chunk_chapters_sentence(result.chapters)
t9 = time.perf_counter()
print(f"sentence切块: {t9-t8:.3f}s ({len(sentence)} 块)")

print("=" * 50)
print(f"总耗时:       {t9-t0:.3f}s")
