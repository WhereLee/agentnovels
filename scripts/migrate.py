"""迁移旧目录结构到新结构"""
import sys
sys.path.insert(0, 'src')

import json
import shutil
import time
from pathlib import Path

from encoding_detector import read_file_auto_encoding
from chapter_detector import detect_chapters
from chunker import chunk_chapters, save_chunks, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP

PROJECT_ROOT = Path(__file__).resolve().parent
NOVELS_DIR = PROJECT_ROOT / "novels"
OLD_TXT_DIR = NOVELS_DIR / "txt"
OLD_RAW_DIR = NOVELS_DIR / "raw"

def migrate():
    if not OLD_TXT_DIR.exists():
        print("无旧数据需要迁移")
        return

    for txt_file in OLD_TXT_DIR.glob("*.txt"):
        novel_name = txt_file.stem
        new_dir = NOVELS_DIR / novel_name

        if new_dir.exists():
            print(f"  跳过（已存在）: {novel_name}")
            continue

        print(f"  迁移: {novel_name}")
        new_dir.mkdir(parents=True)

        # 1. 复制原文
        shutil.copy2(txt_file, new_dir / "original.txt")

        # 2. 读取 + 检测
        text, encoding, confidence = read_file_auto_encoding(str(new_dir / "original.txt"))
        result = detect_chapters(text)

        # 3. 保存章节
        chapters_dir = new_dir / "chapters"
        chapters_dir.mkdir()
        for ch in result.chapters:
            import re
            safe_title = re.sub(r'[<>:"/\\|?*]', '', ch["title"])[:30]
            with open(chapters_dir / f"{ch['index']:03d}_{safe_title}.json", 'w', encoding='utf-8') as f:
                json.dump({"title": ch["title"], "content": ch["content"]}, f, ensure_ascii=False, indent=2)

        # 4. 切块
        chunks = chunk_chapters(result.chapters)
        save_chunks(chunks, str(new_dir / "chunks" / "fixed"))

        # 5. meta.json
        meta = {
            "name": novel_name,
            "encoding": encoding,
            "encoding_confidence": round(confidence, 3),
            "total_chars": len(text),
            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chapter_detected": result.detected,
            "chapter_pattern": result.pattern_name,
            "chapter_count": len(result.chapters),
            "chunk_count": len(chunks),
            "chunk_strategy": "fixed",
            "chunk_params": {"size": DEFAULT_CHUNK_SIZE, "overlap": DEFAULT_OVERLAP},
            "chapters": [{"index": ch["index"], "title": ch["title"], "chars": ch["chars"]} for ch in result.chapters],
            "status": "ready",
            "message": result.message,
        }
        with open(new_dir / "meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"    完成: {len(result.chapters)} 章, {len(chunks)} 块")

    print("\n迁移完成。旧目录 novels/txt 和 novels/raw 可手动删除。")

if __name__ == "__main__":
    print("=== 数据迁移 ===")
    migrate()
