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
import json
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
from query_rewriter import rewrite_query


# RRF 常数
RRF_K = 60

# Reranker ONNX INT8 模型路径
RERANKER_ONNX_INT8_PATH = PROJECT_ROOT / "models" / "bge-reranker-v2-m3-onnx-int8"

# Reranker 并发控制（全局信号量，最多同时 1 个 Rerank 操作）
_rerank_semaphore = threading.Semaphore(1)
RERANK_TIMEOUT = 15  # 等待信号量超时（秒）

# 检索日志路径
RETRIEVAL_LOG_PATH = PROJECT_ROOT / "logs" / "retrieval.jsonl"


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
        带并发控制：如果 Reranker 繁忙超时，降级为 RRF 分数排序。
        """
        if not candidates:
            return []

        # 并发控制：尝试获取信号量
        acquired = _rerank_semaphore.acquire(timeout=RERANK_TIMEOUT)
        if not acquired:
            logger.warning("Reranker 繁忙（等待超时），降级为 RRF 分数排序")
            candidates.sort(key=lambda x: x.get("rrf_score", 0), reverse=True)
            return candidates[:top_k]

        try:
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

        finally:
            _rerank_semaphore.release()

    def _entity_boost(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """
        实体加分：如果查询含明确关键词，对包含该词的 chunk 加分。
        解决实体歧义问题：确保含目标实体名的片段排序靠前。
        """
        query_words = set(jieba.lcut(query))
        # 只关注 >= 2 字的词（排除单字停用词）
        query_keywords = {w for w in query_words if len(w) >= 2}

        if not query_keywords:
            return candidates

        for cand in candidates:
            chunk = self._chunk_map.get(cand["chunk_id"])
            if chunk:
                chunk_words = set(jieba.lcut(chunk["content"][:300]))
                # 计算关键词命中
                hits = query_keywords & chunk_words
                # >= 3 字的专有名词命中权重更高
                entity_hits = sum(1 for w in hits if len(w) >= 3)
                common_hits = len(hits) - entity_hits
                cand["entity_boost"] = entity_hits * 0.3 + common_hits * 0.1

        return candidates

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """
        完整检索管线：查询改写 → 多路向量+BM25 → RRF → 实体加分 → Rerank → 阈值判定 → top_k
        
        返回: [{chunk_id, content, chapter_title, chapter_index, score, source_strategy, low_confidence}]
        """
        t0 = time.perf_counter()

        # 1. 查询改写（语义鸿沟缓解）
        queries = rewrite_query(query)

        # 2. 多查询并行检索
        all_vector_hits = []
        all_bm25_hits = []
        for q in queries:
            all_vector_hits.extend(self._vector_search(q, top_n=VECTOR_TOP_K))
            all_bm25_hits.extend(self._bm25_search(q, top_n=BM25_TOP_K))

        # 去重（同一 chunk_id 只保留最高分）
        vector_hits = self._dedup_hits(all_vector_hits)
        bm25_hits = self._dedup_hits(all_bm25_hits)

        # 3. RRF 融合
        fused = self._rrf_fusion(vector_hits, bm25_hits)

        # 4. 实体加分
        fused = self._entity_boost(query, fused)
        # 将 entity_boost 加入 rrf_score
        for item in fused:
            item["rrf_score"] = item.get("rrf_score", 0) + item.get("entity_boost", 0)
        fused.sort(key=lambda x: x["rrf_score"], reverse=True)

        # 5. Rerank 精排（带并发控制）
        reranked = self._rerank(query, fused, top_k=top_k)

        # 6. 组装结果 + 低置信度判定
        results = []
        low_confidence = False
        if reranked:
            top_score = reranked[0].get("rerank_score", 0)
            # logits < 0 意味着模型认为不相关
            if top_score < 0:
                low_confidence = True
                logger.warning(f"低置信度检索: top_score={top_score:.2f}")
            # 极度不相关（logits < -5）→ 返回空
            if top_score < -5:
                logger.warning(f"检索结果极度不相关 (score={top_score:.2f})，返回空")
                reranked = []

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
                    "low_confidence": low_confidence,
                })

        elapsed = time.perf_counter() - t0
        logger.info(
            f"检索完成: '{query[:20]}...' → {len(results)} 结果, "
            f"耗时 {elapsed:.2f}s (查询{len(queries)}版本, "
            f"向量{len(vector_hits)} + BM25{len(bm25_hits)} → RRF{len(fused)} → Rerank{len(results)})"
        )

        # 7. 检索日志
        self._log_retrieval(query, results, elapsed, len(queries))

        return results

    def _dedup_hits(self, hits: List[Dict]) -> List[Dict]:
        """去重：同一 chunk_id 只保留最高分、最高排名"""
        best = {}  # chunk_id → hit
        for hit in hits:
            cid = hit["chunk_id"]
            if cid not in best or hit["score"] > best[cid]["score"]:
                best[cid] = hit
        # 重新排名
        sorted_hits = sorted(best.values(), key=lambda x: x["score"], reverse=True)
        for i, hit in enumerate(sorted_hits, 1):
            hit["rank"] = i
        return sorted_hits

    def _log_retrieval(self, query: str, results: List[Dict], elapsed: float, query_versions: int):
        """检索日志：写入 JSONL 文件，用于质量监控"""
        try:
            log_entry = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "novel": self.novel_name,
                "strategy": self.strategy,
                "query": query[:80],
                "query_versions": query_versions,
                "top_score": round(results[0]["score"], 4) if results else 0,
                "avg_score": round(sum(r["score"] for r in results) / max(len(results), 1), 4),
                "result_count": len(results),
                "low_confidence": results[0].get("low_confidence", False) if results else False,
                "elapsed": round(elapsed, 2),
            }
            RETRIEVAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(RETRIEVAL_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception:
            pass  # 日志失败不影响主流程
