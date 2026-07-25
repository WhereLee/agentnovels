"""快速测试 RAG 问答"""
from vectorstore import HybridRetriever
from agent import ask

retriever = HybridRetriever("龙族2·悼亡者之瞳")
retriever.load_index()

question = "楚子航是个什么样的人？"
chunks = retriever.search(question)
answer = ask(question, chunks)

with open("test_output.txt", "w", encoding="utf-8") as f:
    f.write(f"问题：{question}\n\n")
    f.write(f"检索到 {len(chunks)} 个片段：\n")
    for c in chunks[:3]:
        f.write(f"  - {c['chapter_title']} (score: {c['score']:.4f})\n")
    f.write(f"\nAI 回答：\n{answer}\n")

print("done - see test_output.txt")
