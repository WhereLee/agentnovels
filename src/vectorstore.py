"""向量存储与混合检索模块：BGE embedding + numpy 余弦相似度 + BM25 + Rerank"""
import json
import re
import jieba
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

from config import (
    MODEL_PATH, RERANKER_PATH, INDEX_DIR,
    VECTOR_TOP_K, BM25_TOP_K, TOP_K, logger
)


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
            if not MODEL_PATH.exists():
                raise FileNotFoundError(
                    f"Embedding 模型未找到：{MODEL_PATH}\n"
                    f"请先运行 tools/download_model.exe 下载模型。"
                )
            logger.info(f"加载 embedding 模型：{MODEL_PATH}")
            self.model = SentenceTransformer(str(MODEL_PATH))
            logger.info("Embedding 模型加载完成")

    def _load_reranker(self):
        """懒加载 reranker 模型（本地路径）"""
        if self.reranker is None:
            if not RERANKER_PATH.exists():
                logger.warning(f"Reranker 模型未找到：{RERANKER_PATH}，将跳过精排")
                return
            logger.info(f"加载 reranker 模型：{RERANKER_PATH}")
            self.reranker = CrossEncoder(str(RERANKER_PATH))
            logger.info("Reranker 模型加载完成")

    def _load_custom_dict(self):
        """从章节标题提取角色名加入 jieba 自定义词典"""
        if self._custom_dict_loaded:
            return
        names = set()
        for chunk in self.chunks:
            title = chunk.get("chapter_title", "")
            for match in re.findall(r'[\u4e00-\u9fff]{2,4}', title):
                names.add(match)
        # 常见小说角色名（硬编码补充）
        names.update(["楚子航", "路明非", "诺诺", "恺撒", "苏茜", "夏弥",
                      "柳淼淼", "陈雯雯", "芬格尔", "昂热", "奥丁",
                      "迈巴赫", "卡塞尔", "狮心会", "仕兰中学"])
        for name in names:
            jieba.add_word(name, freq=10000)
        self._custom_dict_loaded = True
        logger.info(f"BM25 自定义词典已加载：{len(names)} 个词条")

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
        if not chunks_file.exists():
            raise FileNotFoundError(
                f"索引文件不存在：{chunks_file}\n"
                f"请先运行: python src/main.py --build \"{self.novel_name}\""
            )

        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"索引文件损坏（JSON 解析失败）：{chunks_file}\n错误：{e}")

        # 加载自定义词典（在 BM25 构建前）
        self._load_custom_dict()

        # 加载向量
        emb_file = self.index_dir / "embeddings.npy"
        if not emb_file.exists():
            raise FileNotFoundError(f"向量文件不存在：{emb_file}，请重建索引。")

        try:
            self.embeddings = np.load(emb_file)
        except Exception as e:
            raise ValueError(f"向量文件损坏：{emb_file}\n错误：{e}")

        # 校验维度一致性
        if len(self.chunks) != len(self.embeddings):
            raise ValueError(
                f"索引不一致：chunks={len(self.chunks)} 条，"
                f"embeddings={len(self.embeddings)} 条。请重建索引。"
            )

        # 重建 BM25
        tokenized_corpus = [self._tokenize(c["text"]) for c in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"索引加载完成：{len(self.chunks)} 条文本块")
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
            if self.reranker is None:
                raise RuntimeError("Reranker 未加载")
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
            logger.debug(f"Rerank 完成，返回 {len(results)} 条结果")
            return results

        except Exception as e:
            # reranker 失败时回退到 RRF 排序
            logger.warning(f"Rerank 失败，回退到 RRF 排序：{e}")
            results = []
            for doc_id in sorted_ids[:top_k]:
                chunk = chunk_map[doc_id]
                results.append({
                    "text": chunk["text"],
                    "chapter_title": chunk["chapter_title"],
                    "score": rrf_scores[doc_id],
                })
            return results
