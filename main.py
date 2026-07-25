"""小说问答 AI Agent - CLI 入口"""
import argparse
import sys

from config import INDEX_DIR
from loader import build_all_chunks, list_novels
from vectorstore import HybridRetriever
from agent import ask


def cmd_build(novel_name: str):
    """构建索引"""
    print(f"=== 构建索引：{novel_name} ===\n")

    # 1. 切块
    print("正在切分文本块...")
    chunks = build_all_chunks(novel_name)
    print(f"切分完成：共 {len(chunks)} 个文本块")
    avg_len = sum(len(c["text"]) for c in chunks) / len(chunks)
    print(f"平均块大小：{avg_len:.0f} 字\n")

    # 2. 构建索引
    retriever = HybridRetriever(novel_name)
    retriever.build_index(chunks)

    print(f"\n=== 索引构建完成！===")
    print(f"索引位置：{INDEX_DIR / novel_name}")


def cmd_chat(novel_name: str):
    """交互式问答"""
    index_path = INDEX_DIR / novel_name
    if not index_path.exists():
        print(f"错误：未找到索引，请先运行 --build")
        print(f"  python main.py --build \"{novel_name}\"")
        sys.exit(1)

    print(f"=== 小说问答：{novel_name} ===")
    print("正在加载索引...")

    retriever = HybridRetriever(novel_name)
    retriever.load_index()

    print("\n输入问题开始聊天，输入 quit 或 q 退出。\n")

    history = []  # 对话历史

    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not question:
            continue
        if question.lower() in ("quit", "q", "exit"):
            print("再见！")
            break

        # 检索
        print("  [检索中...]")
        chunks = retriever.search(question)

        # 生成回答
        print("  [生成中...]")
        try:
            answer = ask(question, chunks, history)
        except Exception as e:
            print(f"  LLM 调用失败：{e}")
            continue

        print(f"\nAI：{answer}\n")

        # 记录历史
        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


def main():
    parser = argparse.ArgumentParser(description="小说问答 AI Agent")
    parser.add_argument("--build", type=str, help="构建索引（传入小说名）")
    parser.add_argument("--chat", type=str, help="开始问答（传入小说名）")
    parser.add_argument("--list", action="store_true", help="列出可用小说")

    args = parser.parse_args()

    if args.list:
        novels = list_novels()
        if novels:
            print("可用小说：")
            for n in novels:
                print(f"  - {n}")
        else:
            print("novels/raw/ 下暂无小说。")
        return

    if args.build:
        cmd_build(args.build)
    elif args.chat:
        cmd_chat(args.chat)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
