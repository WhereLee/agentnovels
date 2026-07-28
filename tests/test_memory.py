"""测试记忆系统：模拟多轮对话，验证记忆积累、持久化、恢复"""
from memory import ConversationMemory
from vectorstore import HybridRetriever
from agent import ask

NOVEL = "龙族2·悼亡者之瞳"

# 清理旧会话
import os
session_file = f"index/{NOVEL}/session.json"
if os.path.exists(session_file):
    os.remove(session_file)

# 初始化
print("=== 初始化记忆 ===")
memory = ConversationMemory(NOVEL)
retriever = HybridRetriever(NOVEL)
retriever.load_index()

# 模拟 3 轮对话
questions = [
    "楚子航是个什么样的人？",
    "那他和他爸爸的关系呢？",
    "你觉得这本书最打动你的地方是什么？",
]

for i, q in enumerate(questions, 1):
    print(f"\n--- 第 {i} 轮 ---")
    print(f"读者：{q}")
    
    chunks = retriever.search(q)
    answer = ask(q, chunks, memory)
    print(f"AI：{answer[:200]}...")
    
    memory.add_turn(q, answer)
    print(f"  [记忆状态] turns={memory.total_turns}, recent={len(memory.recent)}条, "
          f"anchors角色={memory.anchors['discussed_characters']}")

# 测试持久化：重新加载
print("\n=== 测试会话恢复 ===")
memory2 = ConversationMemory(NOVEL)
print(f"恢复后：turns={memory2.total_turns}, recent={len(memory2.recent)}条")
print(f"锚点：{memory2.anchors}")

# 测试追问连续性：问一个需要上下文的问题
print("\n--- 追问测试（需要记忆） ---")
q = "你刚才说他和他爸的关系，能再展开说说吗？"
print(f"读者：{q}")
chunks = retriever.search(q)
answer = ask(q, chunks, memory2)
print(f"AI：{answer[:300]}...")

print("\n=== 测试完成 ===")
