"""诊断：搜索原文中网吧/混混相关chunk"""
import sqlite3

conn = sqlite3.connect('novels/龙族2·悼亡者之瞳/data.db')

# 搜索多个关键词
keywords = ['网吧', '混混', '打架', '揍', '路明非']
for kw in keywords:
    rows = conn.execute(
        "SELECT chunk_id, chapter_title, substr(content, 1, 80) FROM chunks WHERE strategy='fixed' AND content LIKE ?",
        (f'%{kw}%',)
    ).fetchall()
    print(f'"{kw}": {len(rows)} chunks')
    for r in rows[:2]:
        print(f'  chunk_{r[0]} [{r[1]}]: {r[2]}...')
    print()

conn.close()
