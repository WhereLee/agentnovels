"""
自定义词典构建模块

从小说数据中自动提取高频词（角色名、地名等），
生成 jieba 自定义词典，提升 BM25 分词质量。

策略：
1. 从章节标题中提取关键词（标题通常含角色名/地名）
2. 从 chunks 正文中统计高频 2-4 字组合（出现 > 20 次）
3. 合并去重 → 写入 custom_dict.txt
"""
import re
import json
import sqlite3
from pathlib import Path
from collections import Counter
from typing import List

from config import NOVELS_RAW_DIR, logger

# 停用词（不应作为专有名词的高频词）
STOP_WORDS = {
    "的", "了", "是", "在", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着",
    "没有", "看", "好", "自己", "这", "他", "她", "它", "们", "那", "些",
    "什么", "这个", "那个", "但是", "如果", "因为", "所以", "可以", "已经",
    "还是", "或者", "虽然", "然后", "这样", "那样", "知道", "觉得", "时候",
    "现在", "真的", "其实", "应该", "可能", "比较", "非常", "特别", "终于",
    "突然", "忽然", "竟然", "居然", "果然", "当然", "不过", "只是", "就是",
    "而且", "并且", "于是", "之后", "以前", "以后", "左右", "大概", "大约",
    "起来", "出来", "下来", "过来", "回去", "回来", "过来", "过去",
    "一声", "一下", "一点", "一些", "一种", "一边", "一面",
    "不是", "没有", "不会", "不能", "不要", "不行", "不对",
    "怎么", "怎样", "为什么", "多少", "几个", "哪里",
    "他们", "她们", "我们", "你们", "大家", "自己",
    "东西", "事情", "问题", "地方", "时间", "办法",
    "开始", "结束", "继续", "发现", "感觉", "希望",
}

# 中文字符匹配
RE_CHINESE = re.compile(r'[\u4e00-\u9fff]{2,4}')


def _extract_from_titles(novel_dir: Path) -> List[str]:
    """从 meta.json 的章节标题中提取关键词"""
    meta_file = novel_dir / "meta.json"
    if not meta_file.exists():
        return []

    with open(meta_file, 'r', encoding='utf-8') as f:
        meta = json.load(f)

    words = []
    chapters = meta.get("chapters", [])
    for ch in chapters:
        title = ch.get("title", "")
        # 提取标题中的中文词（2-4字）
        found = RE_CHINESE.findall(title)
        words.extend(found)

    return words


def _extract_from_chunks(novel_dir: Path, strategy: str = "fixed", min_freq: int = 20) -> List[str]:
    """
    从 chunks 正文中统计高频 2-4 字组合。
    
    使用滑动窗口提取所有 2/3/4 字组合，
    保留出现次数 > min_freq 的作为候选专有名词。
    """
    db_path = novel_dir / "data.db"
    if not db_path.exists():
        return []

    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT content FROM chunks WHERE strategy = ?", (strategy,)
    ).fetchall()
    conn.close()

    if not rows:
        return []

    # 统计 n-gram 频率
    counter_2 = Counter()
    counter_3 = Counter()
    counter_4 = Counter()

    for (content,) in rows:
        # 只取纯中文段落（去掉标点和非中文字符）
        chinese_only = re.sub(r'[^\u4e00-\u9fff]', '', content)
        n = len(chinese_only)

        for i in range(n - 1):
            counter_2[chinese_only[i:i+2]] += 1
        for i in range(n - 2):
            counter_3[chinese_only[i:i+3]] += 1
        for i in range(n - 3):
            counter_4[chinese_only[i:i+4]] += 1

    # 筛选高频词
    candidates = []
    for word, freq in counter_4.items():
        if freq >= min_freq and word not in STOP_WORDS:
            candidates.append(word)
    for word, freq in counter_3.items():
        if freq >= min_freq and word not in STOP_WORDS:
            # 排除已被4字词覆盖的
            candidates.append(word)
    for word, freq in counter_2.items():
        if freq >= min_freq * 2 and word not in STOP_WORDS:
            # 2字词需要更高频率（避免噪音）
            candidates.append(word)

    return candidates


def build_custom_dict(novel_name: str) -> str:
    """
    为指定小说构建自定义词典。
    
    返回词典文件路径。
    """
    novel_dir = NOVELS_RAW_DIR / novel_name
    dict_path = novel_dir / "custom_dict.txt"

    # 1. 从标题提取
    title_words = _extract_from_titles(novel_dir)

    # 2. 从 chunks 提取高频词
    chunk_words = _extract_from_chunks(novel_dir, strategy="fixed", min_freq=20)

    # 3. 合并去重
    all_words = set(title_words) | set(chunk_words)
    # 移除停用词
    all_words -= STOP_WORDS
    # 移除纯数字或太短的
    all_words = {w for w in all_words if len(w) >= 2}

    # 4. 写入词典文件（jieba 格式：词语 频率 词性）
    sorted_words = sorted(all_words, key=lambda x: (-len(x), x))
    with open(dict_path, 'w', encoding='utf-8') as f:
        for word in sorted_words:
            # 词频设为较高值确保被优先切分，词性标记为 n（名词）
            f.write(f"{word} 5000 n\n")

    logger.info(f"自定义词典构建完成: {novel_name}, {len(sorted_words)} 词 → {dict_path}")
    return str(dict_path)


def load_dict(novel_name: str) -> bool:
    """
    加载小说的自定义词典到 jieba。
    
    如果词典不存在，先构建。
    返回是否加载成功。
    """
    import jieba

    novel_dir = NOVELS_RAW_DIR / novel_name
    dict_path = novel_dir / "custom_dict.txt"

    if not dict_path.exists():
        build_custom_dict(novel_name)

    if dict_path.exists():
        jieba.load_userdict(str(dict_path))
        logger.debug(f"已加载自定义词典: {dict_path}")
        return True
    return False
