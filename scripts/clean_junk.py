"""清理小说中的网站水印，重新切块入库"""
import sys
sys.path.insert(0, 'src')
import re
from pathlib import Path
from encoding_detector import read_file_auto_encoding
from chapter_detector import detect_chapters
from chunker import chunk_chapters
from database import get_db_path, init_db, save_chapters, save_chunks

# 垃圾模式（按优先级排列）
JUNK_PATTERNS = [
    re.compile(r'<(?:q|abbr|tt|details|summary|mark|cite|code|kbd|samp|var)>.*?</(?:q|abbr|tt|details|summary|mark|cite|code|kbd|samp|var)>', re.DOTALL),
    re.compile(r'<(?:q|abbr|tt|details|summary|mark|cite|code|kbd|samp|var)>[^<]*'),  # 未闭合的
    re.compile(r'</(?:q|abbr|tt|details|summary|mark|cite|code|kbd|samp|var)>'),       # 孤立闭合标签
    re.compile(r'99lib[.\w]*'),        # 裸 99lib 残留
    re.compile(r'www\.tianyabook\.com'),
    re.compile(r'tianyabook\.com'),
]

def clean_text(text: str) -> tuple:
    """清理垃圾内容，返回 (清理后文本, 清理次数)"""
    total = 0
    for pattern in JUNK_PATTERNS:
        text, count = pattern.subn('', text)
        total += count
    # 清理残留的多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text, total

# === 处理所有小说 ===
NOVELS_DIR = Path("novels")

for novel_dir in sorted(NOVELS_DIR.iterdir()):
    if not novel_dir.is_dir():
        continue
    original = novel_dir / "original.txt"
    if not original.exists():
        continue

    print(f"处理: {novel_dir.name}")
    
    # 读取
    text, encoding, confidence = read_file_auto_encoding(str(original))
    original_len = len(text)
    
    # 清理
    cleaned, removed_count = clean_text(text)
    
    if removed_count == 0:
        print(f"  无垃圾内容，跳过")
        continue
    
    print(f"  清理 {removed_count} 处垃圾标记")
    print(f"  字数: {original_len:,} → {len(cleaned):,}")
    
    # 重新检测章节
    result = detect_chapters(cleaned)
    print(f"  章节: {len(result.chapters)}")
    
    # 重新切块
    chunks = chunk_chapters(result.chapters)
    print(f"  切块: {len(chunks)}")
    
    # 写入数据库
    db_path = get_db_path(str(novel_dir))
    init_db(db_path)
    save_chapters(db_path, result.chapters)
    save_chunks(db_path, chunks, strategy="fixed")
    
    # 更新 meta.json
    import json, time
    meta_file = novel_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        meta["total_chars"] = len(cleaned)
        meta["chapter_count"] = len(result.chapters)
        meta["chunk_count"] = len(chunks)
        meta["cleaned"] = True
        meta["junk_removed"] = removed_count
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    
    print(f"  完成")

print("\n全部处理完毕。")
