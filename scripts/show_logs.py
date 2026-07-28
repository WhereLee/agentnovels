import json

with open('logs/retrieval.jsonl', 'r', encoding='utf-8') as f:
    lines = f.readlines()

print(f"total entries: {len(lines)}\n")
for i, line in enumerate(lines):
    d = json.loads(line)
    print(f"#{i+1} [{d['time']}] versions={d['query_versions']} top={d['top_score']} avg={d['avg_score']} low_conf={d['low_confidence']} elapsed={d['elapsed']}s")
    print(f"   query: {d['query']}")
    print()
