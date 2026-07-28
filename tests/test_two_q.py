"""逐个测试问题并显示日志"""
import urllib.request
import json
import time

BASE = "http://127.0.0.1:8000"
NOVEL = "龙族2·悼亡者之瞳"

questions = [
    "路明非吃的什么早餐？",
    "那他在和陈雯雯吃饭的时候吃的是什么呢？",
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"Q: {q}")
    print(f"{'='*50}")

    url = f"{BASE}/api/novels/{urllib.request.quote(NOVEL)}/chat"
    data = json.dumps({"message": q, "strategy": "fixed"}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")

    t0 = time.perf_counter()
    resp = urllib.request.urlopen(req, timeout=120)
    content = resp.read().decode("utf-8")
    elapsed = time.perf_counter() - t0

    tokens = []
    sources = []
    for line in content.split("\n"):
        if not line.startswith("data: "):
            continue
        d = json.loads(line[6:])
        if d["type"] == "token":
            tokens.append(d["content"])
        elif d["type"] == "sources":
            sources = d["chunks"]

    answer = "".join(tokens)
    print(f"耗时: {elapsed:.1f}s")
    print(f"回答 ({len(answer)}字): {answer[:150]}...")
    print(f"\n来源片段 ({len(sources)}个):")
    for i, s in enumerate(sources, 1):
        print(f"  #{i:2d} score={s['score']:7.4f} | {s['chapter_title']} | chunk_{s['chunk_id']} | {s['content_preview'][:40]}")

# 显示最新日志
print(f"\n{'='*50}")
print("最新日志:")
print(f"{'='*50}")
with open("logs/retrieval.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()
for line in lines[-2:]:
    d = json.loads(line)
    print(json.dumps(d, ensure_ascii=False, indent=2))
    print()
