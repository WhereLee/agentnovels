"""
向量化服务：编排 chunks → embedder → ChromaDB

功能：
- 从 SQLite 读取 chunks
- 调用 Embedder 计算向量
- 批量写入 ChromaDB（500条/批）
- 断点续传（collection.count()）
- 进度报告（线程安全）
"""
import time
import threading
import chromadb
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from config import NOVELS_RAW_DIR
from database import get_db_path, load_chunks
from embedder import Embedder


class Vectorizer:
    """
    单本小说的向量化管理器（sentence 策略 + ONNX INT8）。
    
    ChromaDB 存储路径: novels/{小说名}/chroma/
    Collection 名: "sentence"
    """

    def __init__(self, novel_name: str):
        self.novel_name = novel_name
        self.strategy = "sentence"
        self.novel_dir = NOVELS_RAW_DIR / novel_name

        # ChromaDB 初始化
        chroma_path = str(self.novel_dir / "chroma")
        Path(chroma_path).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_or_create_collection(
            name=strategy,
            metadata={"hnsw:space": "cosine"},
        )

        # 进度状态（线程安全）
        self._lock = threading.Lock()
        self._status = "idle"  # idle | running | done | error
        self._embedded = 0
        self._total = 0
        self._current_batch_done = 0
        self._start_time = 0.0
        self._error_msg = ""

    def get_progress(self) -> dict:
        """获取当前进度（线程安全）"""
        with self._lock:
            embedded = self.collection.count()
            # 获取总 chunk 数
            db_path = get_db_path(str(self.novel_dir))
            all_chunks = load_chunks(db_path, self.strategy)
            total = len(all_chunks)
            remaining = total - embedded

            elapsed = time.perf_counter() - self._start_time if self._start_time > 0 else 0
            # 预估剩余时间
            if self._status == "running" and self._current_batch_done > 0:
                rate = self._current_batch_done / max(elapsed, 0.1)
                eta = remaining / max(rate, 0.01)
            else:
                eta = 0

            return {
                "status": self._status,
                "embedded": embedded,
                "total": total,
                "remaining": remaining,
                "elapsed": round(elapsed, 1),
                "eta": round(eta, 1),
                "error": self._error_msg,
            }

    def run(
        self,
        count: int = -1,
        batch_size: int = 48,
        progress_callback: Optional[Callable] = None,
    ):
        """
        执行向量化（ONNX INT8）。
        
        参数:
            count: 处理多少块（-1 = 全部剩余）
            batch_size: 推理批次大小
            progress_callback: 外部进度回调
        """
        with self._lock:
            if self._status == "running":
                raise RuntimeError("向量化任务正在运行中")
            self._status = "running"
            self._error_msg = ""
            self._start_time = time.perf_counter()
            self._current_batch_done = 0

        try:
            # 1. 获取断点
            embedded = self.collection.count()

            # 2. 从 SQLite 取全部 chunks
            db_path = get_db_path(str(self.novel_dir))
            all_chunks = load_chunks(db_path, self.strategy)
            total = len(all_chunks)

            # 3. 确定处理范围
            remaining_chunks = all_chunks[embedded:]
            if count > 0:
                remaining_chunks = remaining_chunks[:count]

            if not remaining_chunks:
                with self._lock:
                    self._status = "done"
                return

            with self._lock:
                self._total = total
                self._embedded = embedded

            # 4. 加载 Embedder (ONNX INT8)
            embedder = Embedder(batch_size=batch_size)

            # 5. 提取文本
            texts = [c["content"] for c in remaining_chunks]

            # 6. 向量化（带进度回调）
            def internal_progress(done, total_batch, elapsed):
                with self._lock:
                    self._current_batch_done = done
                if progress_callback:
                    progress_callback(embedded + done, total, elapsed)

            embeddings = embedder.encode_with_pipeline(
                texts, progress_callback=internal_progress
            )

            # 7. 批量写入 ChromaDB（500条/批，带写前校验 + 写后验证）
            WRITE_BATCH = 500
            for i in range(0, len(remaining_chunks), WRITE_BATCH):
                batch_chunks = remaining_chunks[i:i + WRITE_BATCH]
                batch_embeddings = embeddings[i:i + WRITE_BATCH]

                # === 写前校验 ===
                ids = []
                metadatas = []
                valid_embeddings = []
                for j, (chunk, emb) in enumerate(zip(batch_chunks, batch_embeddings)):
                    # 维度校验
                    if len(emb) != 768:
                        raise RuntimeError(
                            f"chunk_{chunk['chunk_id']} embedding 维度异常: "
                            f"期望 768, 实际 {len(emb)}"
                        )
                    # 内容校验
                    if not chunk["content"].strip():
                        raise RuntimeError(
                            f"chunk_{chunk['chunk_id']} 内容为空，拒绝写入"
                        )
                    # metadata 不含 None（ChromaDB 不接受）
                    meta = {
                        "chunk_id": chunk["chunk_id"],
                        "chapter_index": chunk["chapter_index"],
                        "chapter_title": chunk["chapter_title"] or "",
                        "strategy": self.strategy,
                        "chars": chunk["chars"],
                    }
                    ids.append(str(chunk["chunk_id"]))
                    metadatas.append(meta)
                    valid_embeddings.append(emb)

                # === 写入 ===
                self.collection.add(
                    ids=ids,
                    embeddings=[e.tolist() if hasattr(e, 'tolist') else e for e in valid_embeddings],
                    metadatas=metadatas,
                )

                # === 写后验证 ===
                expected_count = embedded + i + len(batch_chunks)
                actual_count = self.collection.count()
                if actual_count != expected_count:
                    raise RuntimeError(
                        f"ChromaDB 写入校验失败: "
                        f"期望 {expected_count} 条, 实际 {actual_count} 条. "
                        f"可能发生了部分写入，建议清空 collection 重建."
                    )

            # 8. 完成
            with self._lock:
                self._status = "done"
                self._current_batch_done = len(texts)

        except Exception as e:
            with self._lock:
                self._status = "error"
                self._error_msg = str(e)
            raise
