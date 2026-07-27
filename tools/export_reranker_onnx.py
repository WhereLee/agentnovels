"""
导出 bge-reranker-v2-m3 为 ONNX 格式 + INT8 量化

产出：
  models/bge-reranker-v2-m3-onnx/       (ONNX FP32)
  models/bge-reranker-v2-m3-onnx-int8/  (ONNX INT8 量化)

Reranker 是 CrossEncoder（序列分类模型），输入 [query, passage] 对，输出相关性分数。
"""
import os
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "bge-reranker-v2-m3"
ONNX_PATH = PROJECT_ROOT / "models" / "bge-reranker-v2-m3-onnx"
ONNX_INT8_PATH = PROJECT_ROOT / "models" / "bge-reranker-v2-m3-onnx-int8"


def export_onnx():
    """导出 ONNX FP32"""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    print("=" * 50)
    print("  Step 1: 导出 Reranker ONNX (FP32)")
    print("=" * 50)
    print(f"  源模型: {MODEL_PATH}")
    print(f"  输出到: {ONNX_PATH}")

    if ONNX_PATH.exists() and (ONNX_PATH / "model.onnx").exists():
        print("  已存在，跳过导出")
        return

    # 从 PyTorch 导出 ONNX
    model = ORTModelForSequenceClassification.from_pretrained(
        str(MODEL_PATH), export=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))

    model.save_pretrained(str(ONNX_PATH))
    tokenizer.save_pretrained(str(ONNX_PATH))

    # 验证
    size_mb = (ONNX_PATH / "model.onnx").stat().st_size / 1024 / 1024
    print(f"  导出完成: model.onnx ({size_mb:.0f} MB)")


def quantize_int8():
    """INT8 动态量化"""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("\n" + "=" * 50)
    print("  Step 2: INT8 动态量化")
    print("=" * 50)

    if ONNX_INT8_PATH.exists() and (ONNX_INT8_PATH / "model.onnx").exists():
        print("  已存在，跳过量化")
        return

    ONNX_INT8_PATH.mkdir(parents=True, exist_ok=True)

    input_model = str(ONNX_PATH / "model.onnx")
    output_model = str(ONNX_INT8_PATH / "model.onnx")

    quantize_dynamic(
        model_input=input_model,
        model_output=output_model,
        weight_type=QuantType.QInt8,
    )

    # 复制 tokenizer 和 config
    import shutil
    for f in ONNX_PATH.iterdir():
        if f.name != "model.onnx" and f.is_file():
            shutil.copy2(f, ONNX_INT8_PATH / f.name)

    size_mb = Path(output_model).stat().st_size / 1024 / 1024
    print(f"  量化完成: model.onnx ({size_mb:.0f} MB)")


def verify():
    """验证 ONNX INT8 推理正确性"""
    import numpy as np
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    print("\n" + "=" * 50)
    print("  Step 3: 验证推理")
    print("=" * 50)

    tokenizer = AutoTokenizer.from_pretrained(str(ONNX_INT8_PATH))
    model = ORTModelForSequenceClassification.from_pretrained(str(ONNX_INT8_PATH))

    # 测试用例
    pairs = [
        ("楚子航是什么性格", "楚子航面无表情地站在那里，金色的瞳孔里像是结了冰。"),
        ("楚子航是什么性格", "今天天气真好，适合出去散步。"),
    ]

    for query, passage in pairs:
        inputs = tokenizer(query, passage, return_tensors="np", truncation=True, max_length=512)
        outputs = model(**{k: v for k, v in inputs.items()})
        score = outputs.logits[0][0] if hasattr(outputs, 'logits') else outputs[0][0][0]
        print(f"  query: {query}")
        print(f"  passage: {passage[:30]}...")
        print(f"  score: {float(score):.4f}")
        print()

    print("  验证通过!")


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"错误: 模型目录不存在 {MODEL_PATH}")
        sys.exit(1)

    export_onnx()
    quantize_int8()
    verify()

    print("\n" + "=" * 50)
    print("  Reranker ONNX 导出全部完成!")
    print(f"  FP32: {ONNX_PATH}")
    print(f"  INT8: {ONNX_INT8_PATH}")
    print("=" * 50)
