"""诊断：搜索夏弥相关chunk"""
import sqlite3

conn = sqlite3.connect('novels/龙族2·悼亡者之瞳/data.db')

keywords = ['夏弥', '夏弥家', '夏弥的房', '楚子航去']
for kw in keywords:
    rows = conn.execute(
        "SELECT chunk_id, chapter_index, chapter_title, substr(content, 1, 100) FROM chunks WHERE strategy='fixed' AND content LIKE ?",
        (f'%{kw}%',)
    ).fetchall()
    print(f'"{kw}": {len(rows)} chunks')
    for r in rows[:5]:
        print(f'  chunk_{r[0]} [第{r[1]}章 {r[2]}]: {r[3]}...')
    print()

conn.close()
