"""
查询改写模块：用 LLM 将用户的口语化/间接性问题改写为多个检索友好版本

解决语义鸿沟问题：
- 用户说"那个冷脸男人" → 改写为"楚子航 外貌 穿着 黑衣"
- 用户说"主角第一次无力感" → 改写为"路明非 第一次 危险 恐惧 绝望"

策略：
- 原始查询保留（不丢信息）
- LLM 生成 2-3 个改写版本（用原文可能出现的措辞）
- 每个版本独立检索，结果合并去重后 RRF 融合
"""
from typing import List

from openai import OpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, logger

# 改写用的轻量 prompt（极短输出，低 temperature）
REWRITE_PROMPT = """你是一个小说检索助手。将读者的问题改写为2个适合在小说原文中搜索的版本。

要求：
- 用原文中可能实际出现的措辞（角色全名、具体场景词、动作描写词）
- 如果问题中有间接指代（如"那个男人""主角"），推测最可能的角色名
- 每个版本一行，不要编号，不要解释

读者问题：{query}"""


def rewrite_query(query: str) -> List[str]:
    """
    将用户查询改写为多个检索版本。
    
    返回: [原始查询, 改写1, 改写2]（最多 3 个）
    如果 LLM 不可用，降级返回仅原始查询。
    """
    queries = [query]

    try:
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=10)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
            max_tokens=100,
            temperature=0.3,
            stream=False,
        )
        content = response.choices[0].message.content.strip()
        rewrites = [line.strip() for line in content.split("\n") if line.strip()]
        # 最多取 2 个改写
        for rw in rewrites[:2]:
            if rw and rw != query and len(rw) > 3:
                queries.append(rw)

        logger.debug(f"查询改写: '{query}' → {queries}")

    except Exception as e:
        # LLM 不可用时降级：仅用原始查询
        logger.warning(f"查询改写失败（降级为原始查询）: {e}")

    return queries
