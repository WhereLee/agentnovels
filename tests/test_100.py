"""100轮对话压力测试：验证记忆压缩、锚点提取、长对话连续性"""
import os
import json
import time
from memory import ConversationMemory
from vectorstore import HybridRetriever
from agent import ask

NOVEL = "龙族2·悼亡者之瞳"

# 清理旧会话
session_file = f"index/{NOVEL}/session.json"
if os.path.exists(session_file):
    os.remove(session_file)

# 初始化
memory = ConversationMemory(NOVEL)
retriever = HybridRetriever(NOVEL)
retriever.load_index()

# 100轮问题池（模拟真实读者的发散式聊天）
questions = [
    # 第一轮：角色初探
    "楚子航是个什么样的人？",
    "路明非呢？他是个什么样的人？",
    "诺诺是谁？",
    "恺撒是个什么角色？",
    "楚子航的爸爸是个怎样的人？",
    # 深入角色
    "你觉得楚子航为什么总是那么冷？",
    "路明非和楚子航谁更孤独？",
    "诺诺喜欢路明非吗？",
    "恺撒对诺诺是什么感情？",
    "楚子航恨他爸爸吗？",
    # 关系
    "路明非和楚子航是怎么认识的？",
    "他们俩之间有没有过真正的冲突？",
    "诺诺和苏茜是什么关系？",
    "楚子航和柳淼淼有过什么交集？",
    "路明非暗恋诺诺，诺诺知道吗？",
    # 情节
    "序幕里那场雨夜到底发生了什么？",
    "奥丁是什么？为什么会出现？",
    "楚子航的爸爸最后死了吗？",
    "迈巴赫那段为什么写得那么长？",
    "那首爱尔兰民歌在书里出现了几次？有什么意义？",
    # 主题
    "这本书的核心主题是什么？",
    "你觉得作者在探讨什么样的父子关系？",
    "血统这个设定在隐喻什么？",
    "书里反复出现的雨象征什么？",
    "为什么叫'悼亡者之瞳'？",
    # 写作手法
    "你觉得这本书的叙事节奏怎么样？",
    "作者为什么用楚子航的视角来开篇？",
    "序幕的信息量是不是太大了？",
    "书里的对话写得怎么样？",
    "你觉得哪个场景写得最好？",
    # 回调（测试记忆）
    "你之前说楚子航很孤独，能再具体说说吗？",
    "回到他爸爸这个话题，你觉得那个男人爱他吗？",
    "你刚才说的那首歌，它歌词是什么意思？",
    "你之前怎么评价诺诺的？",
    "我们之前聊过路明非和楚子航的关系，你觉得他们像什么？",
    # 更多角色
    "夏弥是谁？",
    "苏茜是个什么样的人？",
    "陈雯雯和路明非是什么关系？",
    "芬格尔是什么角色？",
    "校长是个什么样的人？",
    # 情节深入
    "卡塞尔学院是个什么地方？",
    "狮心会是什么组织？",
    "SS级任务是什么？",
    "龙血是什么？",
    "混血种是什么意思？",
    # 情感
    "这本书里最让你难过的场景是哪个？",
    "楚子航有没有哭过？",
    "路明非最脆弱的时刻是什么时候？",
    "你觉得诺诺幸福吗？",
    "恺撒有没有让你心疼的地方？",
    # 对比
    "楚子航和路明非对待感情的态度有什么不同？",
    "恺撒和楚子航谁更强大？",
    "诺诺和苏茜谁更通透？",
    "这本书和龙族1比怎么样？",
    "序幕和尾声有什么呼应？",
    # 假设性问题
    "如果楚子航的爸爸没有死，故事会怎么发展？",
    "如果路明非没有遇到楚子航，他会怎样？",
    "如果诺诺选了路明非，故事会不同吗？",
    "你觉得楚子航最终会原谅他爸爸吗？",
    "如果奥丁那晚没出现，楚子航会怎样？",
    # 细节
    "楚子航为什么每晚都要回忆他爸爸？",
    "那辆迈巴赫对楚子航意味着什么？",
    "楚子航为什么不给亲爸打电话？",
    "路明非为什么叫诺诺'诺诺'而不是全名？",
    "书里为什么反复提到'风筝线'这个意象？",
    # 评价
    "你觉得这本书的结局处理得好吗？",
    "哪个角色塑造得最成功？",
    "哪个角色你觉得写得不够？",
    "这本书适合什么样的人读？",
    "你会推荐这本书吗？怎么推荐？",
    # 发散
    "读完这本书你有什么感受？",
    "这本书让你想到了什么其他作品？",
    "你觉得江南写这本书的时候是什么状态？",
    "如果这本书拍成电影，谁来演楚子航？",
    "你最喜欢书里哪句话？",
    # 继续回调
    "你还记得我们最开始聊楚子航的时候你说了什么吗？",
    "你之前说雨有象征意义，具体是什么？",
    "我们聊过父子关系，你觉得这本书给出了什么答案？",
    "你之前评价过诺诺，现在有没有新的看法？",
    "回顾我们的对话，你觉得这本书最核心的情感是什么？",
    # 填充到100轮
    "楚子航在仕兰中学是什么地位？",
    "路明非的家庭背景是什么？",
    "恺撒的家世怎么样？",
    "这本书的世界观是怎么构建的？",
    "龙族和混血种的设定有什么深意？",
    "你觉得书里的爱情写得真实吗？",
    "楚子航对柳淼淼是什么态度？",
    "路明非为什么总觉得自己配不上诺诺？",
    "书里有没有让你觉得意外的反转？",
    "你觉得尾声为什么叫'每个人的心里都有个死小孩'？",
    "楚子航的'灵视'是什么？",
    "那个男人为什么要把箱子交给奥丁？",
    "你觉得楚子航最终会变成什么样的人？",
    "路明非和楚子航谁更适合当主角？",
    "这本书有没有让你想哭的地方？",
    "你怎么理解'一日是老爹，终生是老爹'这句话？",
    "楚子航为什么不肯接受继父的好？",
    "你觉得诺诺对路明非有没有一点喜欢？",
    "这本书最被低估的角色是谁？",
    "如果用一个词概括这本书，你会选什么？",
]

# 确保有100个问题
while len(questions) < 100:
    questions.append("还有什么我们没聊到的重要内容吗？")

questions = questions[:100]

# 输出文件
output_file = "test_100_output.txt"
with open(output_file, "w", encoding="utf-8") as f:
    f.write("=== 100轮对话压力测试 ===\n\n")

print(f"开始100轮对话测试...")
start_time = time.time()

for i, q in enumerate(questions, 1):
    chunks = retriever.search(q)
    try:
        answer = ask(q, chunks, memory)
    except Exception as e:
        answer = f"[ERROR: {e}]"
    
    memory.add_turn(q, answer)
    
    # 每10轮打印状态
    if i % 10 == 0:
        elapsed = time.time() - start_time
        print(f"  [{i}/100] turns={memory.total_turns}, recent={len(memory.recent)}条, "
              f"summary长度={len(memory.summary)}字, "
              f"角色={memory.anchors['discussed_characters'][:3]}, "
              f"耗时={elapsed:.0f}s")
    
    # 写入输出文件
    with open(output_file, "a", encoding="utf-8") as f:
        f.write(f"--- 第{i}轮 ---\n")
        f.write(f"读者：{q}\n")
        f.write(f"AI：{answer}\n\n")

# 最终状态
elapsed = time.time() - start_time
print(f"\n=== 测试完成 ===")
print(f"总耗时：{elapsed:.0f}s")
print(f"总轮数：{memory.total_turns}")
print(f"recent条数：{len(memory.recent)}")
print(f"summary长度：{len(memory.summary)}字")
print(f"锚点：")
print(f"  角色：{memory.anchors['discussed_characters']}")
print(f"  关注：{memory.anchors['user_focus']}")
print(f"  共识：{memory.anchors['consensus']}")
print(f"  疑问：{memory.anchors['open_questions']}")

# 写入最终状态
with open(output_file, "a", encoding="utf-8") as f:
    f.write(f"\n=== 最终记忆状态 ===\n")
    f.write(f"总轮数：{memory.total_turns}\n")
    f.write(f"摘要：{memory.summary}\n")
    f.write(f"锚点：{json.dumps(memory.anchors, ensure_ascii=False, indent=2)}\n")
