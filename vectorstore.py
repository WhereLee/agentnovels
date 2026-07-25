"""向量存储与混合检索模块：BGE embedding + numpy 余弦相似度 + BM25 + Rerank"""
import json
import jieba
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import MODEL_PATH, INDEX_DIR, VECTOR_TOP_K, BM25_TOP_K, TOP_K

# Reranker 模型路径（本地）
RERANKER_MODEL = str(MODEL_PATH.parent / "bge-reranker-v2-m3")


class HybridRetriever:
    """混合检索器：向量检索 + BM25 关键词检索 + RRF 融合"""

    def __init__(self, novel_name: str):
        self.novel_name = novel_name
        self.index_dir = INDEX_DIR / novel_name
        self.chunks: List[Dict] = []
        self.embeddings: np.ndarray = None
        self.bm25: BM25Okapi = None
        self.model: SentenceTransformer = None
        self.reranker: Optional[CrossEncoder] = None
        self._custom_dict_loaded = False

    def _load_model(self):
        """懒加载 embedding 模型"""
        if self.model is None:
            self.model = SentenceTransformer(str(MODEL_PATH))

    def _load_reranker(self):
        """懒加载 reranker 模型（本地路径）"""
        if self.reranker is None:
            print("正在加载 reranker 模型...")
            self.reranker = CrossEncoder(RERANKER_MODEL)

    def _load_custom_dict(self):
        """从章节标题提取角色名加入 jieba 自定义词典"""
        if self._custom_dict_loaded:
            return
        # 从 chunks 的 chapter_title 中提取中文名称（2-4字）
        names = set()
        for chunk in self.chunks:
            title = chunk.get("chapter_title", "")
            # 提取中文词组（可能是角色名）
            import re
            for match in re.findall(r'[\u4e00-\u9fff]{2,4}', title):
                names.add(match)
        # 常见小说角色名（硬编码补充）
        names.update(["楚子航", "路明非", "诺诺", "恺撒", "苏茜", "夏弥",
                      "柳淼淼", "陈雯雯", "芬格尔", "昂热", "奥丁",
                      "迈巴赫", "卡塞尔", "狮心会", "仕兰中学"])
        for name in names:
            jieba.add_word(name, freq=10000)
        self._custom_dict_loaded = True
        print(f"BM25 自定义词典已加载：{len(names)} 个词条")

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

        # 加载自定义词典（在 BM25 构建前）
        self._load_custom_dict()

        # 加载向量
        self.embeddings = np.load(self.index_dir / "embeddings.npy")

        # 重建 BM25
        tokenized_corpus = [self._tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"索引加载完成，共 {len(self.chunks)} 条文本块。")

    def search(self, query: str, top_k: int = TOP_K) -> List[Dict]:
        """混合检索：向量 + BM25 + RRF 融合 + Rerank"""
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

        # 按 RRF 分数排序，取更多候选用于 rerank
        rerank_k = top_k * 2
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: -rrf_scores[x])[:rerank_k]

        # Rerank（用 cross-encoder 精排）
        chunk_map = {c["id"]: c for c in self.chunks}
        candidates = [chunk_map[doc_id] for doc_id in sorted_ids]

        try:
            self._load_reranker()
            pairs = [(query, c["text"]) for c in candidates]
            rerank_scores = self.reranker.predict(pairs)

            # 按 rerank 分数排序
            scored = list(zip(candidates, rerank_scores))
            scored.sort(key=lambda x: -x[1])

            results = []
            for chunk, score in scored[:top_k]:
                results.append({
                    "text": chunk["text"],
                    "chapter_title": chunk["chapter_title"],
                    "score": float(score),
                })
            return results

        except Exception:
            # reranker 失败时回退到 RRF 排序
            results = []
            for doc_id in sorted_ids[:top_k]:
                chunk = chunk_map[doc_id]
                results.append({
                    "text": chunk["text"],
                    "chapter_title": chunk["chapter_title"],
                    "score": rrf_scores[doc_id],
                })
            return results
