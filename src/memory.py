"""对话记忆管理：原始区 + 摘要区 + 锚点区 + 里程碑 + 持久化"""
import json
from pathlib import Path
from typing import List, Dict, Optional

from openai import OpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, INDEX_DIR, logger

# 记忆配置
MAX_RECENT_TURNS = 40       # 原始区保留最近 N 轮对话
COMPRESS_BATCH = 10         # 每次压缩最老的 N 轮
MILESTONE_INTERVAL = 20     # 每 N 轮生成一个里程碑

# 锚点容量上限
ANCHOR_LIMITS = {
    "discussed_characters": 15,
    "user_focus": 20,
    "consensus": 20,
    "open_questions": 5,
}


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
            "discussed_characters": [],
            "user_focus": [],
            "consensus": [],
            "open_questions": [],
        }
        self.summary: str = ""
        self.recent: List[Dict] = []
        self.total_turns: int = 0
        self.milestones: List[str] = []   # 里程碑记忆（不可压缩）
        self.first_turn: Optional[Dict] = None  # 首轮钉住

        self._load()

    def add_turn(self, user_msg: str, assistant_msg: str):
        """添加一轮对话，必要时触发压缩/里程碑"""
        self.recent.append({"role": "user", "content": user_msg})
        self.recent.append({"role": "assistant", "content": assistant_msg})
        self.total_turns += 1

        # 首轮钉住
        if self.total_turns == 1:
            self.first_turn = {"user": user_msg, "assistant": assistant_msg}

        # 超出窗口时压缩最老的一批
        if len(self.recent) > MAX_RECENT_TURNS * 2:
            self._compress()

        # 每 N 轮生成里程碑
        if self.total_turns % MILESTONE_INTERVAL == 0:
            self._generate_milestone()

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

            # 锚点剪枝：超出容量时截断最老的
            self._prune_anchors()

        except Exception as e:
            logger.warning(f"记忆压缩失败，回退到简单拼接：{e}")
            fallback = "\n".join(
                f"{'读者' if m['role'] == 'user' else 'AI'}：{m['content'][:100]}"
                for m in batch
            )
            self.summary += f"\n[第{self.total_turns}轮前后] {fallback[:500]}"

    def _prune_anchors(self):
        """锚点剪枝：每个类别超出上限时保留最近的"""
        for key, limit in ANCHOR_LIMITS.items():
            if len(self.anchors.get(key, [])) > limit:
                self.anchors[key] = self.anchors[key][-limit:]
                logger.debug(f"锚点剪枝：{key} 截断为 {limit} 条")

    def _generate_milestone(self):
        """每 N 轮生成一个不可压缩的里程碑摘要"""
        # 取最近 N 轮对话
        recent_batch = self.recent[-MILESTONE_INTERVAL * 2:]
        batch_text = "\n".join(
            f"{'读者' if m['role'] == 'user' else 'AI'}：{m['content'][:150]}"
            for m in recent_batch
        )

        prompt = f"""请用一句话概括以下对话阶段的核心内容（30字以内）：

{batch_text}

只输出一句概括，不要其他内容。"""

        try:
            client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=100,
            )
            milestone_text = response.choices[0].message.content.strip()
            start = self.total_turns - MILESTONE_INTERVAL + 1
            self.milestones.append(f"第{start}-{self.total_turns}轮：{milestone_text}")
            logger.info(f"里程碑生成：{self.milestones[-1]}")
        except Exception as e:
            logger.warning(f"里程碑生成失败：{e}")
            self.milestones.append(f"第{self.total_turns - MILESTONE_INTERVAL + 1}-{self.total_turns}轮：（生成失败）")

    def get_context_messages(self) -> List[Dict]:
        """
        构造发给 LLM 的消息列表。
        结构：里程碑 → 锚点 → 摘要 → 首轮 → 最近对话
        """
        messages = []
    
        context_parts = []
    
        # 里程碑（不可压缩的阶段记忆）
        if self.milestones:
            context_parts.append("【对话里程碑】")
            context_parts.extend(self.milestones)
    
        # 锚点
        if any(self.anchors.values()):
            context_parts.append("\n【对话记忆锚点】")
            if self.anchors["discussed_characters"]:
                context_parts.append(f"讨论过的角色：{'、'.join(self.anchors['discussed_characters'])}")
            if self.anchors["user_focus"]:
                context_parts.append(f"读者关注：{'、'.join(self.anchors['user_focus'])}")
            if self.anchors["consensus"]:
                context_parts.append(f"已有共识：{'；'.join(self.anchors['consensus'][-10:])}")  # 最多取最近10条
            if self.anchors["open_questions"]:
                context_parts.append(f"待探讨：{'；'.join(self.anchors['open_questions'])}")
    
        # 摘要
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
    
        # 首轮钉住（如果不在 recent 中）
        if self.first_turn and self.total_turns > MAX_RECENT_TURNS:
            messages.append({"role": "user", "content": f"[我们的第一轮对话] 读者：{self.first_turn['user']}"})
            messages.append({"role": "assistant", "content": self.first_turn['assistant'][:300]})
    
        # 最近对话原文
        messages.extend(self.recent)
    
        return messages

    def _save(self):
        """原子写入 + 备份"""
        self.session_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "novel_name": self.novel_name,
            "anchors": self.anchors,
            "summary": self.summary,
            "recent": self.recent,
            "total_turns": self.total_turns,
            "milestones": self.milestones,
            "first_turn": self.first_turn,
        }

        tmp_file = self.session_file.with_suffix(".tmp")
        bak_file = self.session_file.with_suffix(".bak")

        try:
            # 第1步：写入临时文件
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            # 第2步：当前文件备份
            if self.session_file.exists():
                self.session_file.replace(bak_file)
            # 第3步：临时文件原子替换
            tmp_file.replace(self.session_file)
        except Exception as e:
            logger.error(f"会话保存失败：{e}")
            # 回退：如果 tmp 存在但替换失败，尝试直接写入
            try:
                with open(self.session_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
            except Exception:
                pass

    def _load(self):
        """从本地 JSON 恢复会话（失败时尝试备份）"""
        if not self.session_file.exists():
            # 尝试从备份恢复
            bak_file = self.session_file.with_suffix(".bak")
            if bak_file.exists():
                logger.warning("主会话文件不存在，从备份恢复")
                self._load_from(bak_file)
            return
        self._load_from(self.session_file)

    def _load_from(self, filepath: Path):
        """从指定文件加载会话"""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.anchors = data.get("anchors", self.anchors)
            self.summary = data.get("summary", "")
            self.recent = data.get("recent", [])
            self.total_turns = data.get("total_turns", 0)
            self.milestones = data.get("milestones", [])
            self.first_turn = data.get("first_turn", None)
            if self.total_turns > 0:
                print(f"  恢复会话：已聊 {self.total_turns} 轮")
        except json.JSONDecodeError as e:
            logger.error(f"会话文件损坏（JSON 解析失败）：{filepath}\n{e}")
            # 尝试备份
            bak_file = self.session_file.with_suffix(".bak")
            if filepath != bak_file and bak_file.exists():
                logger.info("尝试从备份恢复...")
                self._load_from(bak_file)
        except Exception as e:
            logger.error(f"会话加载失败：{e}")

    def reset(self):
        """重置会话"""
        self.anchors = {
            "discussed_characters": [],
            "user_focus": [],
            "consensus": [],
            "open_questions": [],
        }
        self.summary = ""
        self.recent = []
        self.total_turns = 0
        self.milestones = []
        self.first_turn = None
        if self.session_file.exists():
            self.session_file.unlink()
        bak_file = self.session_file.with_suffix(".bak")
        if bak_file.exists():
            bak_file.unlink()
