"""验证 ChromaDB PersistentClient 持久化是否正常"""
import chromadb
import numpy as np
import shutil
from pathlib import Path

TEST_DIR = Path("test_chroma_persist")

# 清理旧测试数据
if TEST_DIR.exists():
    shutil.rmtree(TEST_DIR)

print(f"ChromaDB 版本: {chromadb.__version__}")
print("=" * 50)

# === 第一次运行：写入数据 ===
print("\n[1] 创建 PersistentClient，写入数据...")
client = chromadb.PersistentClient(path=str(TEST_DIR))
collection = client.create_collection(
    name="test_novel",
    metadata={"hnsw:space": "cosine"},
)

# 写入 100 个随机向量
vectors = np.random.randn(100, 768).astype(np.float32).tolist()
ids = [f"chunk_{i}" for i in range(100)]
metadatas = [{"chapter_index": i % 10, "strategy": "fixed"} for i in range(100)]

collection.add(
    ids=ids,
    embeddings=vectors,
    metadatas=metadatas,
)
print(f"  写入 {collection.count()} 条")

# 查询测试
query = np.random.randn(1, 768).astype(np.float32).tolist()
results = collection.query(query_embeddings=query, n_results=5)
print(f"  查询返回 {len(results['ids'][0])} 条: {results['ids'][0]}")

# 关闭客户端
del client
print("  客户端已关闭")

# === 第二次运行：重新加载 ===
print("\n[2] 重新打开 PersistentClient，验证持久化...")
client2 = chromadb.PersistentClient(path=str(TEST_DIR))
collection2 = client2.get_collection("test_novel")
print(f"  加载成功! 数据量: {collection2.count()}")

# 再次查询
results2 = collection2.query(query_embeddings=query, n_results=5)
print(f"  查询返回: {results2['ids'][0]}")

# 验证一致性
assert results['ids'][0] == results2['ids'][0], "持久化前后查询结果不一致!"
print("\n  持久化前后查询结果一致")

print("\n" + "=" * 50)
print("  ChromaDB 持久化验证通过!")
print("=" * 50)

# 清理
shutil.rmtree(TEST_DIR)
