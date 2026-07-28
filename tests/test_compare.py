"""对比两种切块策略"""
import sys
sys.path.insert(0, 'src')
from database import get_db_path, load_chapters
from chunker import chunk_chapters, chunk_chapters_sentence

chapters = load_chapters(get_db_path("novels/龙族2·悼亡者之瞳"))

fixed = chunk_chapters(chapters)
sentence = chunk_chapters_sentence(chapters)

print(f"fixed:    {len(fixed)} 块, 平均 {sum(c['chars'] for c in fixed)//len(fixed)} 字/块")
print(f"sentence: {len(sentence)} 块, 平均 {sum(c['chars'] for c in sentence)//len(sentence)} 字/块")

# 检查 sentence 策略是否有句子被切断
import re
bad = 0
for c in sentence:
    content = c['content'].strip()
    if content and content[-1] not in '。！？…\u201d」』】\n':
        bad += 1
print(f"\nsentence 策略中末尾非句末标点的块: {bad}/{len(sentence)}")

# 抽样对比
print("\n--- fixed 第10块 ---")
print(fixed[10]['content'][:100])
print(f"\n--- sentence 第10块 ---")
print(sentence[10]['content'][:100])
