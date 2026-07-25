"""
章节自动检测管线：模式匹配 → 上下文验证 → 密度/均匀性决策 → 降级

设计原则：
- 宽进严出：第一层尽量多收集候选，后续层层过滤
- 容错降级：检测失败不阻断流程，整本作为单章处理
- 可扩展：PATTERNS 列表可随时追加新模式
"""
import re
import statistics
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class ChapterCandidate:
    """章节候选"""
    line_number: int        # 在全文中的行号（0-based）
    char_offset: int        # 在全文中的字符偏移
    title: str              # 章节标题（含标记）
    pattern_name: str       # 命中的模式名


@dataclass
class DetectionResult:
    """检测结果"""
    detected: bool = False
    pattern_name: str = ""
    chapters: List[Dict] = field(default_factory=list)
    # chapters: [{"index": 0, "title": "...", "content": "...", "chars": 123}, ...]
    message: str = ""


# ============================================================
# 第一层：模式库（宽进）
# ============================================================

# 每个模式: (名称, 编译后的正则, 说明)
PATTERNS: List[Tuple[str, re.Pattern, str]] = [
    # 标准中文章节（最全面）
    (
        "standard_chinese",
        re.compile(
            r'^[\s\u3000]*(?:正文[\s\u3000]+)?'           # 可选"正文"前缀
            r'第[零〇一二三四五六七八九十百千万\d]+'       # 第X
            r'[章节幕回卷集部篇]'                          # 章/节/幕/回/卷/集/部/篇
            r'[\s\u3000:：.．、\-—·]*'                    # 分隔符（含间隔号·）
            r'.{0,40}$'                                   # 标题（≤40字）
        ),
        "第X章/节/幕/回/卷/集/部/篇"
    ),
    # 英文章节
    (
        "english_chapter",
        re.compile(
            r'^[\s]*(?:Chapter|CHAPTER|chapter)\s+\d+'
            r'[\s:：.\-—]*.{0,40}$'
        ),
        "Chapter X"
    ),
    # 特殊标记（序幕/终章等）
    (
        "special_marker",
        re.compile(
            r'^[\s\u3000]*(?:序幕|引子|楔子|序章|终章|尾声|番外|后记|结语|终曲)'
            r'[\s\u3000:：.．、\-—·]*.{0,30}$'
        ),
        "序幕/楔子/终章/尾声等特殊标记"
    ),
    # 中文数字编号 + 标题（金庸式："一 风雪惊变"）
    (
        "chinese_numbered",
        re.compile(
            r'^[\s\u3000]*[零一二三四五六七八九十百千]+'
            r'[\s\u3000、.．]+.{1,30}$'
        ),
        "中文数字+标题（如：一、xxx）"
    ),
    # 阿拉伯数字编号 + 标题（"1 重生归来" / "001. 开场"）
    (
        "arabic_numbered",
        re.compile(
            r'^[\s\u3000]*\d{1,4}'
            r'[\s\u3000、.．]+.{1,30}$'
        ),
        "阿拉伯数字+标题（如：1. xxx）"
    ),
    # 方括号标题（【章节名】）
    (
        "bracket_title",
        re.compile(
            r'^[\s\u3000]*【[^】]{1,30}】[\s\u3000]*$'
        ),
        "【标题】独占一行"
    ),
    # 分隔符型（*** / === / ---）
    (
        "separator",
        re.compile(
            r'^[\s\u3000]*(?:[*＊]{3,}|[=＝]{3,}|[\-—]{3,}|[★■◆●○]{3,})[\s\u3000]*$'
        ),
        "分隔符独占一行（***/===/---）"
    ),
]


# ============================================================
# 第二层：上下文验证（严出）
# ============================================================

def _validate_candidate(
    candidate: ChapterCandidate,
    lines: List[str],
    full_text: str,
    total_chars: int,
) -> bool:
    """
    验证单个候选是否为真实章节头。
    返回 True = 通过验证。
    """
    line = lines[candidate.line_number]
    stripped = line.strip()

    # 规则1：行长度 ≤ 50 字（排除正文中偶然匹配）
    if len(stripped) > 50:
        return False

    # 规则2：不在引号/对话内
    # 检查该行前面是否有未闭合的引号
    if _is_inside_dialogue(lines, candidate.line_number):
        return False

    # 规则3：后方有实质内容（到文末或到下一候选之间 ≥ 1000 字）
    remaining = total_chars - candidate.char_offset
    if remaining < 1000 and candidate.line_number > 0:
        # 文末附近的匹配大概率不是章节头（除非是最后一个章节）
        pass  # 放宽：允许最后一段短

    # 规则4：前方是自然收束（空行、句号、感叹号、问号、引号结尾）
    if candidate.line_number > 0:
        prev_line = lines[candidate.line_number - 1].strip()
        if prev_line and not _is_natural_ending(prev_line):
            # 前一行既非空行也非句末 → 可能是段中误命中
            # 但对 special_marker 和 separator 放宽
            if candidate.pattern_name not in ("special_marker", "separator"):
                return False

    return True


def _is_inside_dialogue(lines: List[str], line_idx: int) -> bool:
    """
    简单判断该行是否处于未闭合引号内。
    策略：向上扫描最近 5 行，统计引号开闭。
    """
    open_quotes = 0
    # 从当前行向上扫描
    for i in range(max(0, line_idx - 5), line_idx):
        for ch in lines[i]:
            if ch in '\u201c「':  # 左引号
                open_quotes += 1
            elif ch in '\u201d」':  # 右引号
                open_quotes -= 1
    # 当前行本身
    current = lines[line_idx]
    # 如果行首就是引号内的延续（前面有未闭合引号）
    if open_quotes > 0:
        # 当前行不以引号开头 → 大概率是对话内容
        if not current.strip().startswith(('\u201c', '「')):
            return True
    return False


def _is_natural_ending(line: str) -> bool:
    """判断一行是否为自然收束（段落结尾）"""
    if not line:
        return True  # 空行 = 自然分隔
    # 以句末标点结尾
    if line[-1] in '。！？…\u201d」』】\n':
        return True
    # 以省略号结尾
    if line.endswith('……') or line.endswith('...'):
        return True
    return False


# ============================================================
# 第三层：密度 + 均匀性验证（决策）
# ============================================================

def _evaluate_pattern(
    candidates: List[ChapterCandidate],
    total_chars: int,
) -> Tuple[bool, float]:
    """
    评估一组候选是否构成真实章节结构。
    
    返回: (是否采纳, 得分)
    条件:
      - 候选数 ≥ 3
      - 间隔均匀性: std/mean < 0.6
    """
    if len(candidates) < 3:
        return False, 0.0

    # 计算相邻候选的字符间距
    offsets = sorted(c.char_offset for c in candidates)
    intervals = [offsets[i+1] - offsets[i] for i in range(len(offsets)-1)]

    if not intervals:
        return False, 0.0

    mean_interval = statistics.mean(intervals)
    if mean_interval == 0:
        return False, 0.0

    std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
    cv = std_interval / mean_interval  # 变异系数

    # 均匀性阈值：CV < 0.6 认为大致均匀
    if cv > 0.6:
        return False, 0.0

    # 额外检查：平均间隔 ≥ 2000 字（排除目录页的密集命中）
    if mean_interval < 2000:
        return False, 0.0

    # 得分：候选数越多 + 越均匀 → 分越高
    score = len(candidates) * (1 - cv)
    return True, score


# ============================================================
# 目录页排除
# ============================================================

def _exclude_toc_region(
    candidates: List[ChapterCandidate],
    lines: List[str],
    total_lines: int,
) -> List[ChapterCandidate]:
    """
    排除目录页区域的候选。
    策略：如果前 10% 的行内候选密度是全文平均的 5 倍以上，
    判定为目录区，移除该区域的候选。
    """
    if not candidates or total_lines < 50:
        return candidates

    toc_boundary = int(total_lines * 0.1)  # 前 10% 行
    toc_candidates = [c for c in candidates if c.line_number < toc_boundary]
    body_candidates = [c for c in candidates if c.line_number >= toc_boundary]

    if not toc_candidates:
        return candidates

    # 密度比较
    toc_density = len(toc_candidates) / max(toc_boundary, 1)
    body_density = len(body_candidates) / max(total_lines - toc_boundary, 1)

    if body_density == 0:
        # 全部候选都在前 10% → 大概率是目录页，全部丢弃
        return []

    if toc_density / max(body_density, 1e-9) > 5:
        # 前部密度远高于正文 → 排除前部
        return body_candidates

    return candidates


# ============================================================
# 编号连续性检查
# ============================================================

def _check_number_continuity(candidates: List[ChapterCandidate]) -> List[ChapterCandidate]:
    """
    对 standard_chinese 和 arabic_numbered 模式，检查编号连续性。
    移除明显不连续的候选（可能是正文中误命中）。
    
    策略宽松：只移除"孤立跳跃"（前后都不连续），保留偶尔跳号。
    """
    if len(candidates) <= 3:
        return candidates

    # 提取编号
    numbers = []
    for c in candidates:
        num = _extract_number(c.title)
        numbers.append(num)

    # 如果大部分有编号，检查连续性
    valid_nums = [n for n in numbers if n is not None]
    if len(valid_nums) < len(candidates) * 0.5:
        return candidates  # 编号提取率太低，跳过检查

    # 标记孤立跳跃：与前一个和后一个都不连续
    result = []
    for i, c in enumerate(candidates):
        if numbers[i] is None:
            result.append(c)
            continue

        prev_ok = (i == 0) or (numbers[i-1] is None) or (numbers[i] - numbers[i-1] in (1, 2))
        next_ok = (i == len(candidates)-1) or (numbers[i+1] is None) or (numbers[i+1] - numbers[i] in (1, 2))

        if prev_ok or next_ok:
            result.append(c)
        # else: 孤立跳跃，移除

    return result


def _extract_number(title: str) -> Optional[int]:
    """从章节标题中提取编号数字"""
    # 阿拉伯数字
    m = re.search(r'(\d+)', title)
    if m:
        return int(m.group(1))

    # 中文数字（简化处理）
    cn_map = {'零': 0, '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4,
              '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10}
    m = re.search(r'第([零〇一二三四五六七八九十百千万]+)', title)
    if m:
        return _chinese_to_int(m.group(1))

    # 纯中文数字开头
    m = re.match(r'^([零〇一二三四五六七八九十百千]+)', title.strip())
    if m:
        return _chinese_to_int(m.group(1))

    return None


def _chinese_to_int(cn: str) -> Optional[int]:
    """中文数字转整数（支持到千位）"""
    digit_map = {'零': 0, '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4,
                 '五': 5, '六': 6, '七': 7, '八': 8, '九': 9}
    unit_map = {'十': 10, '百': 100, '千': 1000, '万': 10000}

    if not cn:
        return None

    # 纯数字字符
    if cn.isdigit():
        return int(cn)

    result = 0
    current = 0

    for ch in cn:
        if ch in digit_map:
            current = digit_map[ch]
        elif ch in unit_map:
            unit = unit_map[ch]
            if current == 0:
                current = 1  # "十" = 10, 不是 0*10
            result += current * unit
            current = 0
        else:
            return None

    result += current
    return result if result > 0 else None


# ============================================================
# 主入口
# ============================================================

def detect_chapters(text: str) -> DetectionResult:
    """
    对全文执行章节检测管线。
    
    流程:
    1. 按行扫描，收集所有模式候选
    2. 排除目录页区域
    3. 上下文验证
    4. 编号连续性检查
    5. 密度/均匀性决策
    6. 选择最佳模式，执行切分
    7. 降级：全部失败 → 单章处理
    """
    lines = text.split('\n')
    total_lines = len(lines)
    total_chars = len(text)

    # 预计算每行的字符偏移
    line_offsets = []
    offset = 0
    for line in lines:
        line_offsets.append(offset)
        offset += len(line) + 1  # +1 for '\n'

    # === 第一层：模式匹配 ===
    pattern_candidates: Dict[str, List[ChapterCandidate]] = {}

    for pattern_name, pattern_regex, _ in PATTERNS:
        candidates = []
        for line_idx, line in enumerate(lines):
            if pattern_regex.match(line):
                title = line.strip()
                candidates.append(ChapterCandidate(
                    line_number=line_idx,
                    char_offset=line_offsets[line_idx],
                    title=title,
                    pattern_name=pattern_name,
                ))
        if candidates:
            pattern_candidates[pattern_name] = candidates

    if not pattern_candidates:
        return _fallback_single_chapter(text, "未检测到任何章节模式")

    # === 对每种模式独立评估 ===
    best_pattern = None
    best_score = 0.0
    best_valid_candidates = []

    for pattern_name, candidates in pattern_candidates.items():
        # 排除目录页
        candidates = _exclude_toc_region(candidates, lines, total_lines)
        if not candidates:
            continue

        # 上下文验证
        valid = [c for c in candidates
                 if _validate_candidate(c, lines, text, total_chars)]
        if not valid:
            continue

        # 编号连续性
        valid = _check_number_continuity(valid)
        if not valid:
            continue

        # 密度/均匀性决策
        accepted, score = _evaluate_pattern(valid, total_chars)
        if accepted and score > best_score:
            best_score = score
            best_pattern = pattern_name
            best_valid_candidates = valid

    # === 降级 ===
    if not best_pattern:
        return _fallback_single_chapter(text, "候选未通过密度/均匀性验证")

    # === 合并特殊标记（序幕/尾声等）到主模式 ===
    if "special_marker" in pattern_candidates and best_pattern != "special_marker":
        special_candidates = pattern_candidates["special_marker"]
        special_candidates = _exclude_toc_region(special_candidates, lines, total_lines)
        for sc in special_candidates:
            if _validate_candidate(sc, lines, text, total_chars):
                # 检查是否与已有候选位置冲突
                if not any(abs(sc.char_offset - bc.char_offset) < 100 for bc in best_valid_candidates):
                    best_valid_candidates.append(sc)
        # 重新按位置排序
        best_valid_candidates.sort(key=lambda c: c.char_offset)

    # === 执行切分 ===
    chapters = _split_by_candidates(text, best_valid_candidates, lines, line_offsets)

    return DetectionResult(
        detected=True,
        pattern_name=best_pattern,
        chapters=chapters,
        message=f"检测到 {len(chapters)} 个章节（模式: {best_pattern}）",
    )


def _split_by_candidates(
    text: str,
    candidates: List[ChapterCandidate],
    lines: List[str],
    line_offsets: List[int],
) -> List[Dict]:
    """根据候选位置切分全文为章节"""
    chapters = []
    sorted_candidates = sorted(candidates, key=lambda c: c.char_offset)

    for i, cand in enumerate(sorted_candidates):
        start = cand.char_offset
        end = sorted_candidates[i+1].char_offset if i+1 < len(sorted_candidates) else len(text)

        content = text[start:end].strip()
        # 标题是候选行本身
        title = cand.title

        chapters.append({
            "index": i,
            "title": title,
            "content": content,
            "chars": len(content),
        })

    # 如果第一个候选之前有内容（如"前言"），作为第 0 章
    if sorted_candidates and sorted_candidates[0].char_offset > 500:
        preface_content = text[:sorted_candidates[0].char_offset].strip()
        if len(preface_content) > 200:  # 有实质内容才保留
            chapters.insert(0, {
                "index": 0,
                "title": "前言/引子",
                "content": preface_content,
                "chars": len(preface_content),
            })
            # 重新编号
            for i, ch in enumerate(chapters):
                ch["index"] = i

    return chapters


def _fallback_single_chapter(text: str, reason: str) -> DetectionResult:
    """降级：整本作为单章"""
    return DetectionResult(
        detected=False,
        pattern_name="none",
        chapters=[{
            "index": 0,
            "title": "全文",
            "content": text.strip(),
            "chars": len(text.strip()),
        }],
        message=f"未检测到章节结构（{reason}），整本作为单章处理",
    )
