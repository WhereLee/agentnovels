"""RAG Agent 模块：组装 prompt + 调用 LLM 生成回答"""
from typing import List, Dict

from openai import OpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL

SYSTEM_PROMPT = """你是一个小说问答助手。你的任务是基于提供的小说原文片段，准确回答读者关于小说内容的问题。

规则：
1. 只基于提供的原文片段回答，不要编造原文中没有的内容。
2. 如果提供的片段不足以回答问题，诚实说明"根据现有片段无法确定"。
3. 回答时引用具体情节，让读者能追溯到原文。
4. 语气自然，像一个读过这本书的朋友在聊天。
5. 如果问题涉及角色评价，综合多个片段给出立体分析。"""


def build_context(retrieved_chunks: List[Dict]) -> str:
    """将检索到的 chunks 组装为上下文字符串"""
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"【片段{i}｜{chunk['chapter_title']}】\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(context_parts)


def ask(question: str, retrieved_chunks: List[Dict], history: List[Dict] = None) -> str:
    """调用 LLM 生成回答"""
    context = build_context(retrieved_chunks)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # 加入对话历史（如果有）
    if history:
        messages.extend(history[-6:])  # 最近3轮对话

    user_message = f"""以下是从小说中检索到的相关片段：

{context}

---

读者问题：{question}

请基于上述片段回答。"""

    messages.append({"role": "user", "content": user_message})

    client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=0.7,
        max_tokens=2000,
    )

    return response.choices[0].message.content
