"""
混合检索模块：向量检索 + BM25 + RRF 融合 + Rerank 精排

架构：
- 向量路：Embedder(ONNX+INT8) 编码查询 → ChromaDB collection.query() → top-16
- BM25路：jieba 分词 → rank_bm25 打分 → top-16
- RRF 融合：两路结果按 Reciprocal Rank Fusion 合并
- Rerank：bge-reranker-v2-m3 精排 → top-8

每本小说 + 每种策略独立一个 HybridRetriever 实例。
"""
import time
import threading
import numpy as np
import jieba
import chromadb
from pathlib import Path
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi

from config import (
    NOVELS_RAW_DIR, RERANKER_PATH, PROJECT_ROOT, logger,
    TOP_K, VECTOR_TOP_K, BM25_TOP_K,
)
from database import get_db_path, load_chunks
from embedder import Embedder
from dict_builder import load_dict


# RRF 常数
RRF_K = 60

# Reranker ONNX INT8 模型路径
RERANKER_ONNX_INT8_PATH = PROJECT_ROOT / "models" / "bge-reranker-v2-m3-onnx-int8"


class HybridRetriever:
    """
    混合检索器：向量 + BM25 + RRF + Rerank
    
    每本小说 + 每种策略（fixed/sentence）独立实例。
    BM25 索引在初始化时一次性构建，缓存在内存。
    Reranker 懒加载（首次查询时加载）。
    """

    def __init__(self, novel_name: str, strategy: str = "fixed"):
        self.novel_name = novel_name
        self.strategy = strategy
        self.novel_dir = NOVELS_RAW_DIR / novel_name

        # === 1. 加载自定义词典 ===
        load_dict(novel_name)

        # === 2. 加载 chunks 数据（BM25 用） ===
        db_path = get_db_path(str(self.novel_dir))
        self._chunks = load_chunks(db_path, strategy)
        if not self._chunks:
            raise ValueError(f"《{novel_name}》策略 {strategy} 无切块数据")

        # chunk_id → chunk 的映射（快速查找）
        self._chunk_map = {c["chunk_id"]: c for c in self._chunks}

        # === 3. 构建 BM25 索引 ===
        logger.info(f"构建 BM25 索引: {novel_name}/{strategy}, {len(self._chunks)} 块...")
        t0 = time.perf_counter()
        self._tokenized_corpus = [
            jieba.lcut(c["content"]) for c in self._chunks
        ]
        self._bm25 = BM25Okapi(self._tokenized_corpus)
        bm25_time = time.perf_counter() - t0
        logger.info(f"BM25 索引构建完成: {bm25_time:.2f}s")

        # === 4. 初始化 ChromaDB（向量检索用） ===
        chroma_path = str(self.novel_dir / "chroma")
        if not Path(chroma_path).exists():
            raise ValueError(f"《{novel_name}》尚未向量化，请先执行向量化")

        self._chroma_client = chromadb.PersistentClient(path=chroma_path)
        try:
            self._collection = self._chroma_client.get_collection(strategy)
        except Exception:
            raise ValueError(f"《{novel_name}》策略 {strategy} 尚未向量化")

        vector_count = self._collection.count()
        if vector_count == 0:
            raise ValueError(f"《{novel_name}》策略 {strategy} 向量库为空")

        logger.info(f"ChromaDB 加载: {vector_count} 向量")

        # === 5. Embedder（ONNX+INT8，用于查询编码） ===
        self._embedder: Optional[Embedder] = None

        # === 6. Reranker（懒加载） ===
        self._reranker = None
        self._reranker_lock = threading.Lock()

    def _get_embedder(self) -> Embedder:
        """懒加载 Embedder"""
        if self._embedder is None:
            self._embedder = Embedder(mode="onnx_int8", batch_size=1)
        return self._embedder

    def _get_reranker(self):
        """懒加载 Reranker（ONNX Runtime INT8，无 PyTorch 依赖）"""
        if self._reranker is None:
            with self._reranker_lock:
                if self._reranker is None:
                    from optimum.onnxruntime import ORTModelForSequenceClassification
                    from transformers import AutoTokenizer
                    logger.info(f"加载 Reranker (ONNX INT8): {RERANKER_ONNX_INT8_PATH}")
                    self._reranker_tokenizer = AutoTokenizer.from_pretrained(str(RERANKER_ONNX_INT8_PATH))
                    self._reranker = ORTModelForSequenceClassification.from_pretrained(str(RERANKER_ONNX_INT8_PATH))
        return self._reranker

    def _vector_search(self, query: str, top_n: int = VECTOR_TOP_K) -> List[Dict]:
        """
        向量检索路：query → embedding → ChromaDB query → top_n 结果
        返回: [{chunk_id, score}]
        """
        embedder = self._get_embedder()
        query_embedding = embedder.encode_batch([query])[0]  # (768,)

        results = self._collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_n, self._collection.count()),
        )

        hits = []
        if results and results["ids"] and results["ids"][0]:
            ids = results["ids"][0]
            distances = results["distances"][0] if results.get("distances") else [0] * len(ids)
            for i, (doc_id, dist) in enumerate(zip(ids, distances)):
                hits.append({
                    "chunk_id": int(doc_id),
                    "score": 1.0 - dist,  # cosine distance → similarity
                    "rank": i + 1,
                })
        return hits

    def _bm25_search(self, query: str, top_n: int = BM25_TOP_K) -> List[Dict]:
        """
        BM25 检索路：query → jieba 分词 → BM25 打分 → top_n 结果
        返回: [{chunk_id, score}]
        """
        query_tokens = jieba.lcut(query)
        scores = self._bm25.get_scores(query_tokens)

        # 取 top_n
        top_indices = np.argsort(scores)[::-1][:top_n]
        hits = []
        for rank, idx in enumerate(top_indices, 1):
            if scores[idx] > 0:
                hits.append({
                    "chunk_id": self._chunks[idx]["chunk_id"],
                    "score": float(scores[idx]),
                    "rank": rank,
                })
        return hits

    def _rrf_fusion(self, vector_hits: List[Dict], bm25_hits: List[Dict]) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 融合两路结果。
        
        公式: score = 1/(K + rank_vector) + 1/(K + rank_bm25)
        """
        scores = {}  # chunk_id → rrf_score

        for hit in vector_hits:
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + hit["rank"])

        for hit in bm25_hits:
            cid = hit["chunk_id"]
            scores[cid] = scores.get(cid, 0) + 1.0 / (RRF_K + hit["rank"])

        # 按 RRF 分数排序
        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused = [
            {"chunk_id": cid, "rrf_score": scores[cid]}
            for cid in sorted_ids
        ]
        return fused

    def _rerank(self, query: str, candidates: List[Dict], top_k: int = TOP_K) -> List[Dict]:
        """
        Rerank 精排：用 ONNX INT8 CrossEncoder 对候选重新打分。
        """
        if not candidates:
            return []

        reranker = self._get_reranker()

        # 构建 (query, passage) 对
        passages = []
        valid_candidates = []
        for cand in candidates:
            chunk = self._chunk_map.get(cand["chunk_id"])
            if chunk:
                passages.append(chunk["content"])
                valid_candidates.append(cand)

        if not passages:
            return []

        # ONNX 批量推理
        inputs = self._reranker_tokenizer(
            [query] * len(passages), passages,
            padding=True, truncation=True, max_length=512,
            return_tensors="np",
        )
        outputs = reranker(**{k: v for k, v in inputs.items()})
        rerank_scores = outputs.logits[:, 0]  # (N,)

        # 合并分数并排序
        for i, cand in enumerate(valid_candidates):
            cand["rerank_score"] = float(rerank_scores[i])

        valid_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return valid_candidates[:top_k]

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        完整检索管线：向量 + BM25 → RRF → Rerank → top_k
        
        返回: [{chunk_id, content, chapter_title, chapter_index, score, source_strategy}]
        """
        t0 = time.perf_counter()

        # 1. 向量检索
        vector_hits = self._vector_search(query, top_n=VECTOR_TOP_K)

        # 2. BM25 检索
        bm25_hits = self._bm25_search(query, top_n=BM25_TOP_K)

        # 3. RRF 融合
        fused = self._rrf_fusion(vector_hits, bm25_hits)

        # 4. Rerank 精排
        reranked = self._rerank(query, fused, top_k=top_k)

        # 5. 组装结果
        results = []
        for item in reranked:
            chunk = self._chunk_map.get(item["chunk_id"])
            if chunk:
                results.append({
                    "chunk_id": chunk["chunk_id"],
                    "content": chunk["content"],
                    "chapter_title": chunk["chapter_title"],
                    "chapter_index": chunk["chapter_index"],
                    "score": item.get("rerank_score", item.get("rrf_score", 0)),
                    "source_strategy": self.strategy,
                })

        elapsed = time.perf_counter() - t0
        logger.info(
            f"检索完成: '{query[:20]}...' → {len(results)} 结果, "
            f"耗时 {elapsed:.2f}s (向量{len(vector_hits)} + BM25{len(bm25_hits)} → RRF{len(fused)} → Rerank{len(results)})"
        )
        return results
