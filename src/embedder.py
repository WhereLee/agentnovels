"""
Embedding 推理模块：支持 PyTorch / ONNX / ONNX+INT8 三种模式

优化：
- P 核线程绑定（i5-12500H 混合架构）
- 生产者-消费者流水线（分词与推理并行）
- 可配置 batch_size
"""
import os
import queue
import threading
import time
import numpy as np
from pathlib import Path
from typing import List, Callable, Optional

# === P 核绑定（在 import torch 之前设置）===
# i5-12500H: 4P + 8E，P 核含超线程 = 8 线程
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

import torch
torch.set_num_threads(8)

from config import MODEL_PATH, PROJECT_ROOT

# ONNX 模型路径
ONNX_MODEL_PATH = PROJECT_ROOT / "models" / "bge-base-zh-v1.5-onnx"
ONNX_INT8_MODEL_PATH = PROJECT_ROOT / "models" / "bge-base-zh-v1.5-onnx-int8"


class Embedder:
    """
    统一 Embedding 推理接口。
    
    支持三种模式：
    - "pytorch": 原始 PyTorch 推理（基线）
    - "onnx": ONNX Runtime FP32 推理
    - "onnx_int8": ONNX Runtime INT8 量化推理
    """

    def __init__(self, mode: str = "pytorch", batch_size: int = 48):
        self.mode = mode
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self):
        """根据模式加载对应模型"""
        if self.mode == "pytorch":
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(str(MODEL_PATH))

        elif self.mode == "onnx":
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
            self._model = ORTModelForFeatureExtraction.from_pretrained(str(ONNX_MODEL_PATH))
            self._tokenizer = AutoTokenizer.from_pretrained(str(ONNX_MODEL_PATH))

        elif self.mode == "onnx_int8":
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer
            self._model = ORTModelForFeatureExtraction.from_pretrained(str(ONNX_INT8_MODEL_PATH))
            self._tokenizer = AutoTokenizer.from_pretrained(str(ONNX_INT8_MODEL_PATH))

        else:
            raise ValueError(f"不支持的模式: {self.mode}，可选: pytorch / onnx / onnx_int8")

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        对一批文本执行推理，返回 embedding 矩阵 (N, 768)。
        """
        if self.mode == "pytorch":
            return self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        else:
            # ONNX 模式
            inputs = self._tokenizer(
                texts, padding=True, truncation=True,
                max_length=512, return_tensors="np"
            )
            outputs = self._model(**{k: v for k, v in inputs.items()})
            # Mean pooling
            embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
            # L2 归一化
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
            return embeddings

    def _mean_pooling(self, model_output, attention_mask) -> np.ndarray:
        """Mean pooling over token embeddings"""
        # model_output 可能是 tuple 或 BaseModelOutput
        if hasattr(model_output, 'last_hidden_state'):
            token_embeddings = model_output.last_hidden_state
        elif isinstance(model_output, (list, tuple)):
            token_embeddings = model_output[0]
        else:
            token_embeddings = model_output

        # 转为 numpy
        if hasattr(token_embeddings, 'numpy'):
            token_embeddings = token_embeddings.numpy()
        if hasattr(attention_mask, 'numpy'):
            attention_mask = attention_mask.numpy()

        # 扩展 mask 到 embedding 维度
        mask_expanded = np.expand_dims(attention_mask, axis=-1)  # (batch, seq, 1)
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.clip(np.sum(attention_mask, axis=1, keepdims=True), a_min=1e-9, a_max=None)
        return sum_embeddings / sum_mask

    def encode_with_pipeline(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int, float], None]] = None,
    ) -> np.ndarray:
        """
        生产者-消费者流水线推理。
        
        生产者线程：预分词（或预处理下一批文本）
        消费者（主线程）：执行推理
        
        progress_callback(done, total, elapsed_seconds)
        """
        total = len(texts)
        if total == 0:
            return np.array([])

        # 将文本分批
        batches = []
        for i in range(0, total, self.batch_size):
            batches.append(texts[i:i + self.batch_size])

        all_embeddings = []
        start_time = time.perf_counter()

        if self.mode == "pytorch":
            # PyTorch 模式：sentence-transformers 内部已优化
            # 用生产者线程预切片，主线程做 encode
            prefetch_queue = queue.Queue(maxsize=4)

            def producer():
                for batch in batches:
                    prefetch_queue.put(batch)
                prefetch_queue.put(None)  # 哨兵

            prod_thread = threading.Thread(target=producer, daemon=True)
            prod_thread.start()

            done = 0
            while True:
                batch = prefetch_queue.get()
                if batch is None:
                    break
                embeddings = self.encode_batch(batch)
                all_embeddings.append(embeddings)
                done += len(batch)
                if progress_callback:
                    elapsed = time.perf_counter() - start_time
                    progress_callback(done, total, elapsed)

            prod_thread.join()

        else:
            # ONNX 模式：生产者做分词，消费者做推理
            tokenized_queue = queue.Queue(maxsize=4)

            def tokenizer_producer():
                for batch in batches:
                    inputs = self._tokenizer(
                        batch, padding=True, truncation=True,
                        max_length=512, return_tensors="np"
                    )
                    tokenized_queue.put((batch, inputs))
                tokenized_queue.put(None)

            prod_thread = threading.Thread(target=tokenizer_producer, daemon=True)
            prod_thread.start()

            done = 0
            while True:
                item = tokenized_queue.get()
                if item is None:
                    break
                batch_texts, inputs = item
                # 推理
                outputs = self._model(**{k: v for k, v in inputs.items()})
                embeddings = self._mean_pooling(outputs, inputs["attention_mask"])
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                embeddings = embeddings / np.maximum(norms, 1e-12)
                all_embeddings.append(embeddings)
                done += len(batch_texts)
                if progress_callback:
                    elapsed = time.perf_counter() - start_time
                    progress_callback(done, total, elapsed)

            prod_thread.join()

        return np.vstack(all_embeddings) if all_embeddings else np.array([])
