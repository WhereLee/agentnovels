"""
文本清洗模块：通用小说文本预处理

策略（按优先级）：
1. 剥离所有 HTML 标签（小说正文不应含标签）
2. 移除 URL 模式
3. 移除已知站点裸文字水印
4. 清理残留空行
"""
import re
from typing import Tuple, List

# === 通用规则 ===

# HTML 标签（含自闭合）
RE_HTML_TAG = re.compile(r'<[^>]{1,50}>')

# URL 模式
RE_URL = re.compile(r'(?:https?://|www\.)[a-zA-Z0-9.\-]+\.[a-z]{2,6}(?:/\S*)?', re.IGNORECASE)

# 裸域名（如 tianyabook.com、99lib.net）
RE_DOMAIN = re.compile(r'[a-zA-Z0-9\-]{2,20}\.(?:com|net|org|cc|me|io|top|xyz)(?![a-z])', re.IGNORECASE)

# === 已知站点关键词（可扩展）===
KNOWN_SITES = [
    '99lib', 'tianyabook', 'biquge', 'xbiquge', '69shu', 'bqg',
    'uukanshu', 'kanshu', 'zanghaihua', 'txt99', 'qidian',
]
RE_KNOWN_SITES = re.compile('|'.join(KNOWN_SITES), re.IGNORECASE)

# === 清理后残留 ===
RE_MULTI_BLANK = re.compile(r'\n{3,}')
RE_TRAILING_SPACES = re.compile(r'[ \t]+\n')


def clean_text(text: str) -> Tuple[str, dict]:
    """
    清洗小说文本。
    
    返回: (清理后文本, 统计信息)
    统计信息: {"html_tags": N, "urls": N, "domains": N, "known_sites": N, "total": N}
    """
    stats = {"html_tags": 0, "urls": 0, "domains": 0, "known_sites": 0}

    # 1. 剥离 HTML 标签
    text, n = RE_HTML_TAG.subn('', text)
    stats["html_tags"] = n

    # 2. 移除 URL
    text, n = RE_URL.subn('', text)
    stats["urls"] = n

    # 3. 移除裸域名
    text, n = RE_DOMAIN.subn('', text)
    stats["domains"] = n

    # 4. 移除已知站点关键词残留
    text, n = RE_KNOWN_SITES.subn('', text)
    stats["known_sites"] = n

    # 5. 清理格式
    text = RE_TRAILING_SPACES.sub('\n', text)
    text = RE_MULTI_BLANK.sub('\n\n', text)

    stats["total"] = sum(stats.values())
    return text, stats
