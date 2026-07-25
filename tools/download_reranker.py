"""下载 bge-reranker-v2-m3 模型到本地 models/ 目录"""
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
model_dir = project_root / "models" / "bge-reranker-v2-m3"

print("=" * 50)
print("  Reranker 模型下载工具")
print("=" * 50)
print(f"\n模型：BAAI/bge-reranker-v2-m3")
print(f"镜像：hf-mirror.com")
print(f"下载到：{model_dir}")
print(f"\n开始下载...\n")

snapshot_download(
    repo_id="BAAI/bge-reranker-v2-m3",
    local_dir=str(model_dir),
)

print("\n" + "=" * 50)
print("  ✓ 下载完成！")
print(f"  模型已保存到：{model_dir}")
print("=" * 50)
