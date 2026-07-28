"""
章节概要生成测试：取一个章节 → LLM 概要 → 保存结果

用途：验证概要提取效果，观察 LLM 对小说体裁的处理质量
后续：确认效果后集成到 Wiki 模块管线中
"""
import sys
import json
import time
sys.path.insert(0, 'src')

from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL
from database import get_db_path, load_chapters

# === 配置 ===
NOVEL = "龙族2·悼亡者之瞳"
CHAPTER_INDEX = 0  # 测试第一章（前言/引子）
OUTPUT_FILE = "test_summary_output.txt"

# === 概要生成 prompt ===
SUMMARY_PROMPT = """阅读小说中某一章节，你需要提取出其中的所有实体名，但不包括笼统代指名词。随后对该章节进行概括，要求包含所有情节内容，特别注意上述你所提取出的实体相关的情节需要避免过度失真。字数不做限制，但是建议2000字左右。

章节标题：{title}

章节正文：
{content}"""


def main():
    # 1. 加载章节
    db_path = get_db_path(f"novels/{NOVEL}")
    chapters = load_chapters(db_path)

    if CHAPTER_INDEX >= len(chapters):
        print(f"错误: 章节索引 {CHAPTER_INDEX} 超出范围 (共 {len(chapters)} 章)")
        return

    chapter = chapters[CHAPTER_INDEX]
    title = chapter["title"]
    content = chapter["content"]
    print(f"章节: 第{CHAPTER_INDEX}章 · {title}")
    print(f"原文字数: {len(content)}")
    print("=" * 50)

    # 2. 调用 LLM 生成概要
    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=120)

    prompt = SUMMARY_PROMPT.format(title=title, content=content)

    print("正在生成概要...")
    t0 = time.perf_counter()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,  # 概要需要准确性，不要太发散
        stream=False,
    )

    elapsed = time.perf_counter() - t0
    summary = response.choices[0].message.content

    print(f"生成完成: {elapsed:.1f}s")
    print(f"概要字数: {len(summary)}")
    print(f"压缩比: {len(content)/len(summary):.1f}x")
    print("=" * 50)

    # 3. 保存结果
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(f"小说: {NOVEL}\n")
        f.write(f"章节: 第{CHAPTER_INDEX}章 · {title}\n")
        f.write(f"原文字数: {len(content)}\n")
        f.write(f"概要字数: {len(summary)}\n")
        f.write(f"耗时: {elapsed:.1f}s\n")
        f.write(f"{'='*50}\n\n")
        f.write(summary)

    print(f"结果已保存: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
