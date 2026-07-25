"""向量存储与混合检索模块：BGE embedding + numpy 余弦相似度 + BM25"""
import json
import jieba
import numpy as np
from typing import List, Dict
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from config import MODEL_PATH, INDEX_DIR, VECTOR_TOP_K, BM25_TOP_K, TOP_K


class HybridRetriever:
    """混合检索器：向量检索 + BM25 关键词检索 + RRF 融合"""

    def __init__(self, novel_name: str):
        self.novel_name = novel_name
        self.index_dir = INDEX_DIR / novel_name
        self.chunks: List[Dict] = []
        self.embeddings: np.ndarray = None
        self.bm25: BM25Okapi = None
        self.model: SentenceTransformer = None

    def _load_model(self):
        """懒加载 embedding 模型"""
        if self.model is None:
            self.model = SentenceTransformer(str(MODEL_PATH))

    def _tokenize(self, text: str) -> List[str]:
        """中文分词（用于 BM25）"""
        return list(jieba.cut(text))

    def build_index(self, chunks: List[Dict]):
        """构建向量索引 + BM25 索引"""
        self._load_model()
        self.chunks = chunks

        # 1. 保存 chunks 元数据到 JSON
        self.index_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = self.index_dir / "chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False)

        # 2. 向量化并保存为 numpy 文件
        texts = [c["text"] for c in chunks]
        print(f"正在向量化 {len(texts)} 个文本块...")
        self.embeddings = self.model.encode(texts, show_progress_bar=True, batch_size=32)
        np.save(self.index_dir / "embeddings.npy", self.embeddings)
        print(f"向量索引构建完成，共 {len(chunks)} 条。")

        # 3. BM25 索引
        print("正在构建 BM25 索引...")
        tokenized_corpus = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print("BM25 索引构建完成。")

    def load_index(self):
        """加载已有索引"""
        self._load_model()

        # 从 JSON 加载 chunks 元数据
        chunks_file = self.index_dir / "chunks.json"
        with open(chunks_file, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        # 加载向量
        self.embeddings = np.load(self.index_dir / "embeddings.npy")

        # 重建 BM25
        tokenized_corpus = [self._tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"索引加载完成，共 {len(self.chunks)} 条文本块。")

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """混合检索：向量 + BM25 + RRF 融合"""
        # 向量检索（余弦相似度）
        query_embedding = self.model.encode([query])[0]  # shape: (768,)
        # 归一化
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        corpus_norms = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        similarities = corpus_norms @ query_norm  # 余弦相似度
        vector_top_indices = similarities.argsort()[::-1][:VECTOR_TOP_K]

        # BM25 检索
        query_tokens = self._tokenize(query)
        bm25_scores = self.bm25.get_scores(query_tokens)
        bm25_top_indices = bm25_scores.argsort()[::-1][:BM25_TOP_K]

        # RRF 融合
        rrf_scores = {}
        k = 60  # RRF 常数

        for rank, idx in enumerate(vector_top_indices):
            doc_id = self.chunks[idx]["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, idx in enumerate(bm25_top_indices):
            doc_id = self.chunks[idx]["id"]
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        # 按 RRF 分数排序
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:top_k]

        # 组装结果
        results = []
        chunk_map = {c["id"]: c for c in self.chunks}
        for doc_id in sorted_ids:
            chunk = chunk_map[doc_id]
            results.append({
                "text": chunk["text"],
                "chapter_title": chunk["chapter_title"],
                "score": rrf_scores[doc_id],
            })

        return results
