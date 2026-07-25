"""对话记忆管理：原始区 + 摘要区 + 锚点区 + 持久化"""
import json
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, INDEX_DIR

# 记忆配置
MAX_RECENT_TURNS = 40       # 原始区保留最近 N 轮对话
COMPRESS_BATCH = 10         # 每次压缩最老的 N 轮


class ConversationMemory:
    """
    单书单对话的记忆管理器。
    
    结构：
    - anchors: 锚点区（结构化关键信息，置于 prompt 最前端）
    - summary: 摘要区（早期对话的压缩）
    - recent: 原始区（最近 N 轮完整对话）
    """

    def __init__(self, novel_name: str):
        self.novel_name = novel_name
        self.session_file = INDEX_DIR / novel_name / "session.json"
        self.anchors: Dict[str, List[str]] = {
            "discussed_characters": [],   # 讨论过的角色
            "user_focus": [],             # 用户关注点
            "consensus": [],              # 已达成的共识/结论
            "open_questions": [],         # 未解决的疑问
        }
        self.summary: str = ""
        self.recent: List[Dict] = []     # [{"role": "user"/"assistant", "content": "..."}]
        self.total_turns: int = 0

        # 尝试加载已有会话
        self._load()

    def add_turn(self, user_msg: str, assistant_msg: str):
        """添加一轮对话，必要时触发压缩"""
        self.recent.append({"role": "user", "content": user_msg})
        self.recent.append({"role": "assistant", "content": assistant_msg})
        self.total_turns += 1

        # 超出窗口时压缩最老的一批
        if len(self.recent) > MAX_RECENT_TURNS * 2:
            self._compress()

        # 每轮结束自动持久化
        self._save()

    def _compress(self):
        """将最老的 N 轮对话压缩进摘要区，并更新锚点"""
        # 取出最老的 N 轮（2N 条消息）
        batch = self.recent[:COMPRESS_BATCH * 2]
        self.recent = self.recent[COMPRESS_BATCH * 2:]

        # 构造压缩 prompt
        batch_text = "\n".join(
            f"{'读者' if m['role'] == 'user' else 'AI'}：{m['content']}"
            for m in batch
        )

        compress_prompt = f"""请将以下对话历史压缩为简洁摘要，保留关键信息（讨论了哪些角色、什么观点、什么结论）。
同时从中提取结构化锚点信息。

已有摘要：
{self.summary if self.summary else '（无）'}

已有锚点：
- 讨论过的角色：{', '.join(self.anchors['discussed_characters']) or '无'}
- 用户关注点：{', '.join(self.anchors['user_focus']) or '无'}
- 已达成共识：{', '.join(self.anchors['consensus']) or '无'}
- 未解决疑问：{', '.join(self.anchors['open_questions']) or '无'}

需要压缩的新对话：
{batch_text}

请输出 JSON 格式：
{{
  "summary": "更新后的完整摘要（包含已有摘要+新内容的融合，200字以内）",
  "discussed_characters": ["更新后的角色列表"],
  "user_focus": ["更新后的用户关注点"],
  "consensus": ["更新后的共识"],
  "open_questions": ["更新后的未解决疑问"]
}}"""

        try:
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": compress_prompt}],
                temperature=0.3,
            )
            result_text = response.choices[0].message.content

            # 解析 JSON（容错处理）
            result_text = result_text.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(result_text)

            self.summary = result.get("summary", self.summary)
            for key in self.anchors:
                if key in result and isinstance(result[key], list):
                    self.anchors[key] = result[key]

        except Exception as e:
            # 压缩失败时简单拼接，不丢数据
            fallback = "\n".join(
                f"{'读者' if m['role'] == 'user' else 'AI'}：{m['content'][:100]}"
                for m in batch
            )
            self.summary += f"\n[第{self.total_turns}轮前后] {fallback[:500]}"

    def get_context_messages(self) -> List[Dict]:
        """
        构造发给 LLM 的消息列表（不含 system prompt 和检索片段）。
        结构：锚点注入 → 摘要 → 最近对话
        """
        messages = []

        # 锚点 + 摘要作为一条"背景信息"注入
        context_parts = []
        if any(self.anchors.values()):
            context_parts.append("【对话记忆锚点】")
            if self.anchors["discussed_characters"]:
                context_parts.append(f"讨论过的角色：{'、'.join(self.anchors['discussed_characters'])}")
            if self.anchors["user_focus"]:
                context_parts.append(f"读者关注：{'、'.join(self.anchors['user_focus'])}")
            if self.anchors["consensus"]:
                context_parts.append(f"已有共识：{'；'.join(self.anchors['consensus'])}")
            if self.anchors["open_questions"]:
                context_parts.append(f"待探讨：{'；'.join(self.anchors['open_questions'])}")

        if self.summary:
            context_parts.append(f"\n【早期对话摘要】\n{self.summary}")

        if context_parts:
            messages.append({
                "role": "user",
                "content": "\n".join(context_parts) + "\n\n（以上是我们之前对话的记忆，请基于此继续。）"
            })
            messages.append({
                "role": "assistant",
                "content": "好的，我记得我们之前聊的内容。继续吧。"
            })

        # 最近对话原文
        messages.extend(self.recent)

        return messages

    def _save(self):
        """持久化到本地 JSON"""
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "novel_name": self.novel_name,
            "anchors": self.anchors,
            "summary": self.summary,
            "recent": self.recent,
            "total_turns": self.total_turns,
        }
        with open(self.session_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load(self):
        """从本地 JSON 恢复会话"""
        if not self.session_file.exists():
            return
        try:
            with open(self.session_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.anchors = data.get("anchors", self.anchors)
            self.summary = data.get("summary", "")
            self.recent = data.get("recent", [])
            self.total_turns = data.get("total_turns", 0)
            if self.total_turns > 0:
                print(f"  恢复会话：已聊 {self.total_turns} 轮")
        except Exception:
            pass

    def reset(self):
        """重置会话（开新书或用户主动清空）"""
        self.anchors = {
            "discussed_characters": [],
            "user_focus": [],
            "consensus": [],
            "open_questions": [],
        }
        self.summary = ""
        self.recent = []
        self.total_turns = 0
        if self.session_file.exists():
            self.session_file.unlink()
