"""按章节切分小说文本为独立 JSON 文件"""
import os
import re
import json

# 路径
txt_dir = r"c:\Users\lrs\Desktop\pytxt\novels\txt"
files = os.listdir(txt_dir)
src = os.path.join(txt_dir, files[0])
novel_name = os.path.splitext(files[0])[0]

out_dir = os.path.join(r"c:\Users\lrs\Desktop\pytxt\novels\raw", novel_name)
os.makedirs(out_dir, exist_ok=True)

# 读取（GBK 编码）
with open(src, "r", encoding="gbk") as f:
    lines = f.readlines()

# 识别章节标记行
chapter_pattern = re.compile(r"^(序幕|第.+?幕|尾声)\s+(.+)")
chapters = []

for i, line in enumerate(lines):
    stripped = line.strip()
    m = chapter_pattern.match(stripped)
    if m:
        chapters.append((i, stripped))

# 切分
results = []
for idx, (start_line, title) in enumerate(chapters):
    end_line = chapters[idx + 1][0] if idx + 1 < len(chapters) else len(lines)
    content_lines = lines[start_line + 1 : end_line]
    content = "".join(content_lines).strip()
    results.append({"title": title, "content": content})

# 写入 JSON
for idx, ch in enumerate(results):
    safe_title = ch["title"].replace(" ", "_").replace("/", "_").replace("\\", "_")
    if len(safe_title) > 30:
        safe_title = safe_title[:30]
    fname = f"{idx:02d}_{safe_title}.json"
    fpath = os.path.join(out_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(ch, f, ensure_ascii=False, indent=2)

# 输出统计
summary_lines = [f"小说: {novel_name}", f"章节数: {len(results)}", f"输出目录: {out_dir}", ""]
for idx, ch in enumerate(results):
    title = ch["title"]
    clen = len(ch["content"])
    summary_lines.append(f"{idx:02d} | {title} | {clen}字")

with open(os.path.join(out_dir, "_summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(summary_lines))

print("done")
