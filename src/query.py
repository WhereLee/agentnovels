"""查询预处理：指代消解 + 查询增强（纯规则，不调 LLM）"""
import re
from typing import List, Dict

# 中文代词映射
PRONOUNS_MALE = {"他", "他的", "他们", "那人", "那个人", "那个男人", "那个男的"}
PRONOUNS_FEMALE = {"她", "她的", "她们", "那人", "那个人", "那个女人", "那个女的"}
PRONOUNS_NEUTRAL = {"它", "它的", "这", "那", "这个", "那个"}

# 常见指代表达 → 消解策略
COREFERENCE_PATTERNS = [
    (re.compile(r"(他|他的|那个人|那个男人|那个男的|那人)(?!们)"), "male"),
    (re.compile(r"(她|她的|那个人|那个女人|那个女的|那人)(?!们)"), "female"),
    (re.compile(r"刚才说的那个|之前聊的|你刚说的那个"), "recent_topic"),
]


def resolve_coreference(query: str, memory) -> str:
    """
    基于记忆锚点消解查询中的代词/指代。
    
    策略：
    1. 检查最近几轮对话中提到的角色
    2. 用锚点中的角色列表辅助判断
    3. 简单替换，不引入额外复杂度
    """
    if not memory or not memory.recent:
        return query

    # 从最近对话中提取最后提到的角色名
    last_character = _find_last_character(memory)

    if not last_character:
        return query

    # 检查是否包含代词
    resolved = query
    has_pronoun = False

    for pronoun in PRONOUNS_MALE | PRONOUNS_FEMALE:
        if pronoun in query:
            has_pronoun = True
            break

    if has_pronoun and last_character:
        # 在查询前注入角色上下文（不直接替换，而是追加说明）
        resolved = f"{query}（{query}中的代词指的是{last_character}）"

    return resolved


def _find_last_character(memory) -> str:
    """从最近对话中找到最后提到的角色名"""
    known_characters = memory.anchors.get("discussed_characters", [])
    if not known_characters:
        return ""

    # 从最近的对话倒序扫描，找最先出现的已知角色
    for msg in reversed(memory.recent[-10:]):  # 最近5轮
        content = msg.get("content", "")
        for char_name in known_characters:
            if char_name in content:
                return char_name

    # 兜底：返回锚点中第一个角色
    return known_characters[0] if known_characters else ""


def enhance_query(query: str, memory) -> str:
    """
    查询增强入口：指代消解 + 后续可扩展的增强步骤。
    返回增强后的查询。
    """
    # 1. 指代消解
    query = resolve_coreference(query, memory)

    # 2. 未来可扩展：别名展开、关键词补充等

    return query
