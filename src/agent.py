"""RAG Agent 模块：组装 prompt + 调用 LLM 生成回答"""
from typing import List, Dict

from openai import OpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT, logger
from memory import ConversationMemory

SYSTEM_PROMPT = """你是一个对小说有深度理解的对话者。你读过这本书很多遍，不仅记得情节，更能读出文字背后的东西。

你的核心能力：
- 读得懂言外之意。小说里很多东西是不直接说的——角色的潜台词、作者的设计意图、情节的象征意味、留白处的深意。你要能读出这些，并自然地表达出来。
- 理解人物的复杂性。人不是标签，不要给角色贴“外向”“善良”这种平面标签。去理解他们行为背后的动机、矛盾和不得已。
- 允许主观解读。你可以说“我觉得这里其实在写……”、“这段表面是……但底下是……”，不需要“客观中立”。

语气：
- 平静、自然、有分寸。不用刻意兴奋，不用网络用语，不用感叹号堆砌。
- 不列点、不加粗、不写“首先/其次/最后”。像一个人坐你对面慢慢聊。
- 长短随意，说到点上就停。

底线：不编造原文中不存在的情节。可以深度解读，但不能捷造事实。"""


def build_context(retrieved_chunks: List[Dict]) -> str:
    """将检索到的 chunks 组装为上下文字符串"""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"【片段{i}｜{chunk['chapter_title']}】\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def ask(question: str, retrieved_chunks: List[Dict], memory: ConversationMemory) -> str:
    """调用 LLM 生成回答（流式输出）"""
    context = build_context(retrieved_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 注入记忆（锚点 + 摘要 + 最近对话）
    memory_messages = memory.get_context_messages()
    messages.extend(memory_messages)

    # 当前问题（带检索片段）
    user_message = f"""以下是从小说中检索到的相关片段：

{context}

---

读者问题：{question}

请基于上述片段回答。"""

    messages.append({"role": "user", "content": user_message})

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT)

    # 流式调用
    try:
        stream = client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.85,
            stream=True,
        )
    except Exception as e:
        logger.error(f"LLM API 调用失败：{e}")
        raise RuntimeError(
            f"LLM 服务不可用：{e}\n"
            f"请检查网络连接和 API 密钥配置（config.py）。"
        )

    # 逐字输出并累积完整回答
    full_answer = []
    try:
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                print(token, end="", flush=True)
                full_answer.append(token)
    except Exception as e:
        logger.error(f"流式读取中断：{e}")
        if not full_answer:
            raise RuntimeError(f"LLM 响应中断：{e}")

    print()  # 换行
    answer = "".join(full_answer)
    logger.debug(f"LLM 生成完成，回答长度：{len(answer)} 字")
    return answer
