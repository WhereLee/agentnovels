"""编码自动检测模块：BOM优先 → chardet → 多编码尝试 → 中文有效性验证"""
import re
from pathlib import Path
from typing import Optional, Tuple

# BOM 标记 → 编码名（按长度降序，避免短 BOM 误匹配长 BOM）
BOM_MARKERS = [
    (b'\xef\xbb\xbf', 'utf-8-sig'),
    (b'\xff\xfe\x00\x00', 'utf-32-le'),
    (b'\x00\x00\xfe\xff', 'utf-32-be'),
    (b'\xff\xfe', 'utf-16-le'),
    (b'\xfe\xff', 'utf-16-be'),
]

# 候选编码（按中文场景使用率排序）
CANDIDATE_ENCODINGS = ['utf-8', 'gb18030', 'gbk', 'big5', 'utf-16']

# 有效中文字符正则
CHINESE_CHAR_PATTERN = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]')


def detect_encoding(file_path: str, sample_size: int = 8192) -> Tuple[str, float]:
    """
    检测文件编码。
    
    返回: (编码名, 置信度)
    置信度说明:
      1.0  = BOM 精确匹配
      0.9+ = chardet 高置信
      0.7+ = 多编码尝试 + 中文验证通过
      <0.7 = 不确定，使用 fallback
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在: {file_path}")
    if path.stat().st_size == 0:
        raise ValueError(f"文件为空: {file_path}")

    with open(file_path, 'rb') as f:
        raw = f.read(sample_size)

    # 第一层：BOM 检测（100% 准确）
    bom_encoding = _check_bom(raw)
    if bom_encoding:
        return bom_encoding, 1.0

    # 第二层：chardet 检测
    chardet_result = _try_chardet(raw)
    if chardet_result:
        encoding, confidence = chardet_result
        if confidence >= 0.7:
            # 验证解码有效性
            if _validate_decode(raw, encoding):
                return encoding, confidence

    # 第三层：逐编码尝试 + 中文有效性验证
    best_encoding = None
    best_ratio = 0.0

    for encoding in CANDIDATE_ENCODINGS:
        ratio = _chinese_valid_ratio(raw, encoding)
        if ratio > best_ratio:
            best_ratio = ratio
            best_encoding = encoding

    if best_encoding and best_ratio >= 0.5:
        return best_encoding, min(best_ratio, 0.85)

    # 兜底：gb18030（兼容 GBK，覆盖最广）
    return 'gb18030', 0.5


def read_file_auto_encoding(file_path: str) -> Tuple[str, str, float]:
    """
    自动检测编码并读取文件全文。
    
    返回: (文本内容, 编码名, 置信度)
    """
    encoding, confidence = detect_encoding(file_path)

    with open(file_path, 'r', encoding=encoding, errors='replace') as f:
        content = f.read()

    return content, encoding, confidence


def _check_bom(raw: bytes) -> Optional[str]:
    """检查 BOM 标记"""
    for bom, encoding in BOM_MARKERS:
        if raw.startswith(bom):
            return encoding
    return None


def _try_chardet(raw: bytes) -> Optional[Tuple[str, float]]:
    """尝试使用 chardet 检测"""
    try:
        import chardet
        result = chardet.detect(raw)
        if result and result.get('encoding'):
            return result['encoding'].lower(), result.get('confidence', 0)
    except ImportError:
        pass
    except Exception:
        pass
    return None


def _validate_decode(raw: bytes, encoding: str) -> bool:
    """验证解码后是否为有效文本（非乱码）"""
    try:
        text = raw.decode(encoding, errors='strict')
        # 可打印字符占比 > 90%
        printable = sum(1 for c in text if c.isprintable() or c in '\n\r\t')
        return printable / max(len(text), 1) > 0.9
    except (UnicodeDecodeError, LookupError):
        return False


def _chinese_valid_ratio(raw: bytes, encoding: str) -> float:
    """
    尝试用指定编码解码，计算有效中文字符占比。
    占比越高说明编码越可能是对的。
    """
    try:
        text = raw.decode(encoding, errors='strict')
    except (UnicodeDecodeError, LookupError):
        return 0.0

    if not text:
        return 0.0

    # 统计中文字符数 / 总非空白字符数
    non_space = re.sub(r'\s', '', text)
    if not non_space:
        return 0.0

    chinese_count = len(CHINESE_CHAR_PATTERN.findall(non_space))
    return chinese_count / len(non_space)
