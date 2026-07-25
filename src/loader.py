"""文本加载与切块模块：从 raw JSON 读取章节，按段落聚合切块"""
import json
import re
from pathlib import Path
from typing import List, Dict

from config import NOVELS_RAW_DIR, CHUNK_TARGET_SIZE, CHUNK_MAX_SIZE, CHUNK_OVERLAP


def load_chapters(novel_name: str) -> List[Dict]:
    """加载某本小说的所有章节 JSON"""
    novel_dir = NOVELS_RAW_DIR / novel_name
    if not novel_dir.exists():
        raise FileNotFoundError(f"未找到小说目录: {novel_dir}")

    chapters = []
    for f in sorted(novel_dir.glob("*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            data = json.load(fp)
            chapters.append(data)
    return chapters


def split_into_paragraphs(text: str) -> List[str]:
    """将文本按段落拆分（以换行+制表符或连续换行为分隔）"""
    # 小说段落通常以 \n\t 或 \n\n 分隔
    paragraphs = re.split(r"\n[\t]*", text)
    # 过滤空段落
    return [p.strip() for p in paragraphs if p.strip()]


def force_split_long_paragraph(para: str, max_size: int) -> List[str]:
    """对超长段落按句号/感叹号/问号强制断句"""
    if len(para) <= max_size:
        return [para]

    segments = []
    current = ""
    # 按句末标点切分
    sentences = re.split(r'(?<=[。！？…」』\u201d\)])', para)

    for sent in sentences:
        if not sent:
            continue
        if len(current) + len(sent) > max_size and current:
            segments.append(current)
            current = sent
        else:
            current += sent

    if current:
        segments.append(current)

    return segments


def chunk_chapter(title: str, content: str, chapter_idx: int) -> List[Dict]:
    """
    对单章内容进行段落聚合切块。
    返回 chunk 列表，每个 chunk 包含 text, chapter_title, chapter_idx, chunk_idx
    """
    paragraphs = split_into_paragraphs(content)
    chunks = []
    current_chunk = ""
    chunk_idx = 0

    for para in paragraphs:
        # 如果单段超长，先强制断句
        if len(para) > CHUNK_MAX_SIZE:
            # 先把当前累积的存下来
            if current_chunk:
                chunks.append(_make_chunk(current_chunk, title, chapter_idx, chunk_idx))
                chunk_idx += 1
                current_chunk = ""
            # 断句后逐段加入
            sub_parts = force_split_long_paragraph(para, CHUNK_MAX_SIZE)
            for part in sub_parts:
                if len(current_chunk) + len(part) > CHUNK_TARGET_SIZE and current_chunk:
                    chunks.append(_make_chunk(current_chunk, title, chapter_idx, chunk_idx))
                    chunk_idx += 1
                    # 重叠：取当前块末尾作为下一块开头
                    current_chunk = current_chunk[-CHUNK_OVERLAP:] + part
                else:
                    current_chunk += part
            continue

        # 正常段落：累积到目标大小
        if len(current_chunk) + len(para) > CHUNK_TARGET_SIZE and current_chunk:
            chunks.append(_make_chunk(current_chunk, title, chapter_idx, chunk_idx))
            chunk_idx += 1
            # 重叠
            current_chunk = current_chunk[-CHUNK_OVERLAP:] + para
        else:
            current_chunk += ("\n" + para if current_chunk else para)

    # 最后一块
    if current_chunk:
        chunks.append(_make_chunk(current_chunk, title, chapter_idx, chunk_idx))

    return chunks


def _make_chunk(text: str, chapter_title: str, chapter_idx: int, chunk_idx: int) -> Dict:
    return {
        "text": text,
        "chapter_title": chapter_title,
        "chapter_idx": chapter_idx,
        "chunk_idx": chunk_idx,
        "id": f"ch{chapter_idx:02d}_ck{chunk_idx:03d}",
    }


def build_all_chunks(novel_name: str) -> List[Dict]:
    """加载小说所有章节并切块，返回全部 chunk 列表"""
    chapters = load_chapters(novel_name)
    all_chunks = []
    for idx, ch in enumerate(chapters):
        chunks = chunk_chapter(ch["title"], ch["content"], idx)
        all_chunks.extend(chunks)
    return all_chunks


def list_novels() -> List[str]:
    """列出 raw 目录下所有可用小说"""
    if not NOVELS_RAW_DIR.exists():
        return []
    return [d.name for d in NOVELS_RAW_DIR.iterdir() if d.is_dir()]
