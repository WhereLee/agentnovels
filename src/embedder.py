"""
Embedding 推理模块：ONNX Runtime INT8 量化

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

# === P 核绑定 ===
# i5-12500H: 4P + 8E，P 核含超线程 = 8 线程
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"

from config import PROJECT_ROOT

# ONNX INT8 模型路径
ONNX_INT8_MODEL_PATH = PROJECT_ROOT / "models" / "bge-base-zh-v1.5-onnx-int8"


class Embedder:
    """
    Embedding 推理接口（ONNX Runtime INT8）。
    """

    def __init__(self, batch_size: int = 48):
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None
        self._load_model()

    def _load_model(self):
        """加载 ONNX INT8 模型"""
        from optimum.onnxruntime import ORTModelForFeatureExtraction
        from transformers import AutoTokenizer
        self._model = ORTModelForFeatureExtraction.from_pretrained(str(ONNX_INT8_MODEL_PATH))
        self._tokenizer = AutoTokenizer.from_pretrained(str(ONNX_INT8_MODEL_PATH))

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        对一批文本执行推理，返回 embedding 矩阵 (N, 768)。
        """
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
        
        生产者线程：预分词
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

        # 生产者做分词，消费者做推理
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
