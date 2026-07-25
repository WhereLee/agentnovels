"""
文本切块模块：固定字数切块（段落边界优先 + 重叠）

策略：
- 目标块大小 500 字，段落边界切断
- 超长段落按句号/感叹号/问号强制断句
- 相邻块 100 字重叠，保证上下文连贯
- 不跨章节
"""
import json
import re
from typing import List, Dict
from pathlib import Path

# 默认参数
DEFAULT_CHUNK_SIZE = 500
DEFAULT_OVERLAP = 100
DEFAULT_MAX_CHUNK_SIZE = 800

# 断句标点
SENTENCE_ENDINGS = re.compile(r'(?<=[。！？…\u201d」』】])')


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> List[str]:
    """
    对纯文本执行固定字数切块。
    
    参数:
        text: 输入文本
        chunk_size: 目标块大小（字）
        overlap: 相邻块重叠字数
        max_chunk_size: 单块最大字数（超过则强制断句）
    
    返回:
        切块列表
    """
    if not text or not text.strip():
        return []

    # 按段落分割（\n 或 \n\t 或连续空行）
    paragraphs = _split_paragraphs(text)

    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # 当前块 + 新段落 是否超限
        if len(current_chunk) + len(para) <= chunk_size:
            # 可以加入当前块
            current_chunk += ("\n" + para) if current_chunk else para
        else:
            # 当前块已满，保存
            if current_chunk:
                chunks.append(current_chunk)

            # 新段落本身是否超长
            if len(para) > max_chunk_size:
                # 强制断句切分
                sub_chunks = _split_long_paragraph(para, chunk_size, max_chunk_size)
                chunks.extend(sub_chunks[:-1])  # 前面的直接加入
                current_chunk = sub_chunks[-1] if sub_chunks else ""
            else:
                current_chunk = para

    # 最后一块
    if current_chunk.strip():
        chunks.append(current_chunk)

    # 添加重叠
    if overlap > 0 and len(chunks) > 1:
        chunks = _add_overlap(chunks, overlap)

    return chunks


def chunk_chapters(
    chapters: List[Dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
    max_chunk_size: int = DEFAULT_MAX_CHUNK_SIZE,
) -> List[Dict]:
    """
    对章节列表执行切块（不跨章）。
    
    参数:
        chapters: [{"index": 0, "title": "...", "content": "..."}, ...]
    
    返回:
        [{"chunk_id": 0, "chapter_index": 0, "chapter_title": "...", "content": "...", "chars": 123}, ...]
    """
    all_chunks = []
    chunk_id = 0

    for chapter in chapters:
        content = chapter.get("content", "")
        chapter_idx = chapter.get("index", 0)
        chapter_title = chapter.get("title", "")

        text_chunks = chunk_text(content, chunk_size, overlap, max_chunk_size)

        for text in text_chunks:
            all_chunks.append({
                "chunk_id": chunk_id,
                "chapter_index": chapter_idx,
                "chapter_title": chapter_title,
                "content": text,
                "chars": len(text),
            })
            chunk_id += 1

    return all_chunks


def save_chunks(
    chunks: List[Dict],
    output_dir: str,
    params: Dict = None,
):
    """
    保存切块结果到指定目录。
    
    生成:
        output_dir/result.json  - 切块数据
        output_dir/params.json  - 参数记录
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 保存切块结果
    result_file = output_path / "result.json"
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # 保存参数
    if params is None:
        params = {
            "strategy": "fixed",
            "chunk_size": DEFAULT_CHUNK_SIZE,
            "overlap": DEFAULT_OVERLAP,
            "max_chunk_size": DEFAULT_MAX_CHUNK_SIZE,
        }
    params_file = output_path / "params.json"
    with open(params_file, 'w', encoding='utf-8') as f:
        json.dump(params, f, ensure_ascii=False, indent=2)


def _split_paragraphs(text: str) -> List[str]:
    """按段落分割文本"""
    # 支持多种段落分隔：\n\t、\n\n、\n（后跟非空白）
    # 先统一换行符
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # 按单个换行分段（保留缩进信息）
    raw_paras = text.split('\n')

    # 合并空行（连续空行视为一个分隔）
    paragraphs = []
    for line in raw_paras:
        stripped = line.strip()
        if stripped:
            paragraphs.append(stripped)
        elif paragraphs and paragraphs[-1] != "":
            paragraphs.append("")  # 标记段落边界

    return [p for p in paragraphs if p]  # 移除空标记


def _split_long_paragraph(
    para: str,
    chunk_size: int,
    max_size: int,
) -> List[str]:
    """将超长段落按句子边界切分"""
    sentences = SENTENCE_ENDINGS.split(para)
    sentences = [s for s in sentences if s.strip()]

    chunks = []
    current = ""

    for sent in sentences:
        if len(current) + len(sent) <= chunk_size:
            current += sent
        else:
            if current:
                chunks.append(current)
            # 单句超长 → 硬切
            if len(sent) > max_size:
                for i in range(0, len(sent), chunk_size):
                    chunks.append(sent[i:i+chunk_size])
                current = ""
            else:
                current = sent

    if current:
        chunks.append(current)

    return chunks


def _add_overlap(chunks: List[str], overlap: int) -> List[str]:
    """为相邻块添加重叠"""
    if len(chunks) <= 1:
        return chunks

    result = [chunks[0]]
    for i in range(1, len(chunks)):
        # 从前一块末尾取 overlap 字
        prev_tail = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
        # 在句子边界处截断重叠（避免半句话）
        cut_point = _find_sentence_start(prev_tail)
        overlap_text = prev_tail[cut_point:]
        result.append(overlap_text + "\n" + chunks[i])

    return result


def _find_sentence_start(text: str) -> int:
    """找到文本中第一个完整句子的起始位置"""
    # 找第一个句末标点后的位置
    for i, ch in enumerate(text):
        if ch in '。！？…\u201d」』】':
            return i + 1
    # 找不到就返回 0（全部作为重叠）
    return 0
