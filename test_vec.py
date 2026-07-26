"""测试向量化 API：基线 50 块"""
import urllib.request
import json
import time

BASE = "http://127.0.0.1:8000"
NOVEL = "龙族2·悼亡者之瞳"

# 1. 启动向量化
print("启动向量化: PyTorch, bs=48, 50块...")
data = json.dumps({"strategy": "fixed", "mode": "pytorch", "batch_size": 48, "count": 50}).encode()
req = urllib.request.Request(
    f"{BASE}/api/novels/{urllib.request.quote(NOVEL)}/vectorize",
    data=data,
    headers={"Content-Type": "application/json"},
    method="POST",
)
resp = urllib.request.urlopen(req)
print(json.loads(resp.read()))

# 2. 轮询进度
print("\n轮询进度...")
while True:
    time.sleep(3)
    resp = urllib.request.urlopen(
        f"{BASE}/api/novels/{urllib.request.quote(NOVEL)}/vectorize/progress?strategy=fixed"
    )
    p = json.loads(resp.read())
    print(f"  {p['embedded']}/{p['total']}  status={p['status']}  elapsed={p['elapsed']}s  eta={p['eta']}s")
    if p['status'] in ('done', 'error'):
        if p['status'] == 'error':
            print(f"  ERROR: {p['error']}")
        break

print("\n完成!")
