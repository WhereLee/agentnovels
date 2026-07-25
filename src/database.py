"""
SQLite 数据访问层：每本小说一个 data.db

Schema:
  chapters: 章节信息
  chunks:   切块数据（按策略区分）
"""
import sqlite3
import json
from pathlib import Path
from typing import List, Dict, Optional


def get_db_path(novel_dir: str) -> str:
    """获取小说数据库路径"""
    return str(Path(novel_dir) / "data.db")


def init_db(db_path: str):
    """初始化数据库表结构"""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS chapters (
            chapter_index INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            chars INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id INTEGER PRIMARY KEY AUTOINCREMENT,
            chapter_index INTEGER NOT NULL,
            chapter_title TEXT NOT NULL,
            content TEXT NOT NULL,
            chars INTEGER NOT NULL,
            strategy TEXT NOT NULL DEFAULT 'fixed',
            FOREIGN KEY (chapter_index) REFERENCES chapters(chapter_index)
        );

        CREATE INDEX IF NOT EXISTS idx_chunks_strategy ON chunks(strategy);
        CREATE INDEX IF NOT EXISTS idx_chunks_chapter ON chunks(chapter_index);
    """)
    conn.commit()
    conn.close()


def save_chapters(db_path: str, chapters: List[Dict]):
    """
    保存章节数据。
    
    chapters: [{"index": 0, "title": "...", "content": "...", "chars": 123}, ...]
    """
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM chapters")  # 全量替换
    conn.executemany(
        "INSERT INTO chapters (chapter_index, title, content, chars) VALUES (?, ?, ?, ?)",
        [(ch["index"], ch["title"], ch["content"], ch["chars"]) for ch in chapters]
    )
    conn.commit()
    conn.close()


def save_chunks(db_path: str, chunks: List[Dict], strategy: str = "fixed"):
    """
    保存切块数据。
    
    chunks: [{"chunk_id": 0, "chapter_index": 0, "chapter_title": "...", "content": "...", "chars": 123}, ...]
    strategy: 切块策略名（fixed / paragraph / ...）
    """
    conn = sqlite3.connect(db_path)
    # 清除该策略的旧数据
    conn.execute("DELETE FROM chunks WHERE strategy = ?", (strategy,))
    conn.executemany(
        "INSERT INTO chunks (chapter_index, chapter_title, content, chars, strategy) VALUES (?, ?, ?, ?, ?)",
        [(c["chapter_index"], c["chapter_title"], c["content"], c["chars"], strategy) for c in chunks]
    )
    conn.commit()
    conn.close()


def load_chunks(db_path: str, strategy: str = "fixed") -> List[Dict]:
    """加载指定策略的全部切块"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT chunk_id, chapter_index, chapter_title, content, chars FROM chunks WHERE strategy = ? ORDER BY chunk_id",
        (strategy,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def load_chapters(db_path: str) -> List[Dict]:
    """加载全部章节"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT chapter_index, title, content, chars FROM chapters ORDER BY chapter_index"
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def load_chunks_by_chapter(db_path: str, chapter_index: int, strategy: str = "fixed") -> List[Dict]:
    """加载指定章节的切块"""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT chunk_id, chapter_index, chapter_title, content, chars FROM chunks WHERE chapter_index = ? AND strategy = ? ORDER BY chunk_id",
        (chapter_index, strategy)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_chunk_count(db_path: str, strategy: str = "fixed") -> int:
    """获取切块总数"""
    conn = sqlite3.connect(db_path)
    count = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE strategy = ?", (strategy,)
    ).fetchone()[0]
    conn.close()
    return count


def get_strategies(db_path: str) -> List[str]:
    """获取已有的切块策略列表"""
    conn = sqlite3.connect(db_path)
    rows = conn.execute("SELECT DISTINCT strategy FROM chunks").fetchall()
    conn.close()
    return [r[0] for r in rows]
