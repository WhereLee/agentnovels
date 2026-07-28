"""检索质量评测：多类型问题测试"""
import sqlite3
import urllib.request
import json
import time

NOVEL = "龙族2·悼亡者之瞳"
BASE = "http://127.0.0.1:8000"
DB_PATH = f"novels/{NOVEL}/data.db"

# === Step 1: 确认关键场景在原文中存在 ===
def check_data():
    conn = sqlite3.connect(DB_PATH)
    keywords = ['早餐', '网吧', '夏弥家', '高架', '陈雯雯', '奥丁', '村雨', '钢琴', '迈巴赫', '雨', '楚子航', '路明非']
    print("=== 数据确认 ===")
    for kw in keywords:
        count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE strategy='sentence' AND content LIKE ?",
            (f'%{kw}%',)
        ).fetchone()[0]
        print(f"  '{kw}': {count} chunks")
    total = conn.execute("SELECT COUNT(*) FROM chunks WHERE strategy='sentence'").fetchone()[0]
    print(f"  总 sentence chunks: {total}")
    conn.close()

# === Step 2: 测试问题集 ===
QUESTIONS = [
    # A类: 明确实体名查询
    {"id": "A1", "type": "实体+事件", "query": "路明非吃的什么早餐？", "expect_keywords": ["早餐"]},
    {"id": "A2", "type": "实体+事件", "query": "路明非在网吧门口做了什么？", "expect_keywords": ["网吧"]},
    {"id": "A3", "type": "实体+事件", "query": "楚子航在高架路上遇到了什么？", "expect_keywords": ["高架"]},
    {"id": "A4", "type": "实体+物品", "query": "村雨是什么？", "expect_keywords": ["村雨"]},
    {"id": "A5", "type": "实体+事件", "query": "路明非和陈雯雯一起吃饭的场景", "expect_keywords": ["陈雯雯"]},
    # B类: 场景/描述查询（词汇不匹配风险）
    {"id": "B1", "type": "场景描述", "query": "夏弥的房间里有什么？", "expect_keywords": ["夏弥"]},
    {"id": "B2", "type": "场景描述", "query": "小说里描写下雨的场景", "expect_keywords": ["雨"]},
    {"id": "B3", "type": "场景描述", "query": "路明非第一次展现能力的片段", "expect_keywords": ["路明非"]},
    # C类: 关系查询
    {"id": "C1", "type": "关系", "query": "楚子航和夏弥之间发生了什么？", "expect_keywords": ["楚子航", "夏弥"]},
    {"id": "C2", "type": "关系", "query": "路明非和诺诺的互动", "expect_keywords": ["路明非", "诺诺"]},
    # D类: 模糊/感受查询
    {"id": "D1", "type": "模糊", "query": "最紧张的一段战斗描写", "expect_keywords": []},
    {"id": "D2", "type": "模糊", "query": "楚子航最冷漠的一个场景", "expect_keywords": ["楚子航"]},
]

# === Step 3: 逐题测试 ===
def run_test(q):
    url = f"{BASE}/api/novels/{urllib.request.quote(NOVEL)}/chat"
    data = json.dumps({"message": q["query"]}).encode()
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    
    t0 = time.perf_counter()
    try:
        resp = urllib.request.urlopen(req, timeout=120)
        content = resp.read().decode("utf-8")
    except Exception as e:
        return {"error": str(e), "elapsed": time.perf_counter() - t0}
    
    total_time = time.perf_counter() - t0
    
    # 解析 SSE
    meta = None
    tokens = []
    sources = []
    for line in content.split("\n"):
        if not line.startswith("data: "):
            continue
        try:
            d = json.loads(line[6:])
        except:
            continue
        if d["type"] == "meta":
            meta = d["data"]
        elif d["type"] == "token":
            tokens.append(d["content"])
        elif d["type"] == "sources":
            sources = d["chunks"]
    
    answer = "".join(tokens)
    
    # 判断命中：返回的 sources 中是否包含期望关键词
    hit = False
    hit_chunks = []
    if q["expect_keywords"]:
        for s in sources:
            preview = s.get("content_preview", "") + s.get("chapter_title", "")
            if any(kw in preview for kw in q["expect_keywords"]):
                hit = True
                hit_chunks.append(s["chunk_id"])
    
    return {
        "meta": meta,
        "answer_len": len(answer),
        "answer_preview": answer,
        "sources": sources,
        "total_time": round(total_time, 2),
        "keyword_hit": hit,
        "hit_chunks": hit_chunks,
    }

# === Step 4: 生成报告 ===
def generate_report(results):
    lines = []
    lines.append("# 检索质量评测报告")
    lines.append(f"\n- 小说: {NOVEL}")
    lines.append(f"- 时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- 策略: sentence")
    lines.append(f"- 管线: 向量+BM25 → RRF → 实体加分 → Rerank")
    lines.append(f"- 测试题数: {len(QUESTIONS)}")
    lines.append("")
    
    # 汇总
    hit_count = sum(1 for r in results if r.get("keyword_hit"))
    error_count = sum(1 for r in results if "error" in r)
    lines.append("## 汇总")
    lines.append(f"\n| 指标 | 值 |")
    lines.append(f"|------|---|")
    lines.append(f"| 关键词命中率 | {hit_count}/{len(QUESTIONS)} ({hit_count/len(QUESTIONS)*100:.0f}%) |")
    lines.append(f"| 错误数 | {error_count} |")
    avg_time = sum(r.get("total_time", 0) for r in results) / max(len(results), 1)
    lines.append(f"| 平均响应时间 | {avg_time:.1f}s |")
    lines.append("")
    
    # 逐题详情
    lines.append("## 逐题详情")
    lines.append("")
    
    for q, r in zip(QUESTIONS, results):
        hit_mark = "✓" if r.get("keyword_hit") else ("✗" if "error" not in r else "ERR")
        lines.append(f"### [{q['id']}] {q['query']}")
        lines.append(f"- 类型: {q['type']}")
        lines.append(f"- 期望关键词: {q['expect_keywords']}")
        lines.append(f"- 命中: {hit_mark}")
        
        if "error" in r:
            lines.append(f"- 错误: {r['error']}")
            lines.append("")
            continue
        
        meta = r.get("meta", {})
        lines.append(f"- top_score: {meta.get('top_score', 'N/A')}")
        lines.append(f"- avg_score: {meta.get('avg_score', 'N/A')}")
        lines.append(f"- 检索耗时: {meta.get('elapsed', 'N/A')}s")
        lines.append(f"- 总耗时: {r['total_time']}s")
        lines.append(f"- 管线: {meta.get('pipeline', 'N/A')}")
        lines.append(f"- 低置信度: {meta.get('low_confidence', 'N/A')}")
        lines.append(f"- 回答长度: {r['answer_len']}字")
        lines.append(f"")
        lines.append(f"**完整回答:**")
        lines.append(f"")
        lines.append(f"> {r['answer_preview']}")
        lines.append("")
        
        # 返回的 chunks
        lines.append(f"**检索结果 ({len(r['sources'])} 个):**")
        lines.append("")
        lines.append("| # | score | 章节 | chunk_id | 内容预览 |")
        lines.append("|---|-------|------|----------|---------|")
        for i, s in enumerate(r["sources"][:8], 1):
            preview = s.get("content_preview", "")[:50].replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {i} | {s.get('score', 'N/A')} | {s.get('chapter_title', '')} | {s.get('chunk_id', '')} | {preview} |")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    check_data()
    print("\n=== 开始测试 ===")
    results = []
    for q in QUESTIONS:
        print(f"  [{q['id']}] {q['query']}...", end=" ", flush=True)
        r = run_test(q)
        hit = "HIT" if r.get("keyword_hit") else ("ERR" if "error" in r else "MISS")
        elapsed = r.get("total_time", 0)
        print(f"{hit} ({elapsed:.1f}s)")
        results.append(r)
    
    # 生成报告
    report = generate_report(results)
    with open("eval_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n=== 完成，报告已写入 eval_report.md ===")
