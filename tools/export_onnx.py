"""
导出 BGE-base-zh-v1.5 为 ONNX 格式 + INT8 量化
一次性运行，产出：
  models/bge-base-zh-v1.5-onnx/       (ONNX FP32)
  models/bge-base-zh-v1.5-onnx-int8/  (ONNX INT8 量化)
"""
import os
import sys
from pathlib import Path

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "bge-base-zh-v1.5"
ONNX_PATH = PROJECT_ROOT / "models" / "bge-base-zh-v1.5-onnx"
ONNX_INT8_PATH = PROJECT_ROOT / "models" / "bge-base-zh-v1.5-onnx-int8"


def export_onnx():
    """导出 ONNX FP32"""
    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    print("=" * 50)
    print("  Step 1: 导出 ONNX (FP32)")
    print("=" * 50)
    print(f"  源模型: {MODEL_PATH}")
    print(f"  输出到: {ONNX_PATH}")

    if ONNX_PATH.exists() and (ONNX_PATH / "model.onnx").exists():
        print("  已存在，跳过导出")
        return

    # 加载并导出
    model = ORTModelForFeatureExtraction.from_pretrained(
        str(MODEL_PATH),
        export=True,
    )
    model.save_pretrained(str(ONNX_PATH))

    # 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_PATH))
    tokenizer.save_pretrained(str(ONNX_PATH))

    print("  ONNX 导出完成")


def quantize_int8():
    """INT8 动态量化"""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("\n" + "=" * 50)
    print("  Step 2: INT8 动态量化")
    print("=" * 50)

    input_model = ONNX_PATH / "model.onnx"
    output_model = ONNX_INT8_PATH / "model.onnx"

    if output_model.exists():
        print("  已存在，跳过量化")
        return

    ONNX_INT8_PATH.mkdir(parents=True, exist_ok=True)

    print(f"  输入: {input_model}")
    print(f"  输出: {output_model}")

    quantize_dynamic(
        model_input=str(input_model),
        model_output=str(output_model),
        weight_type=QuantType.QInt8,
    )

    # 复制 tokenizer 和配置文件到 INT8 目录
    import shutil
    for f in ONNX_PATH.iterdir():
        if f.name != "model.onnx" and not f.is_dir():
            shutil.copy2(f, ONNX_INT8_PATH / f.name)

    # 复制子目录（tokenizer 文件可能在子目录）
    for d in ONNX_PATH.iterdir():
        if d.is_dir():
            target = ONNX_INT8_PATH / d.name
            if not target.exists():
                shutil.copytree(d, target)

    print("  INT8 量化完成")

    # 显示文件大小对比
    fp32_size = input_model.stat().st_size / 1024 / 1024
    int8_size = output_model.stat().st_size / 1024 / 1024
    print(f"\n  FP32: {fp32_size:.1f} MB")
    print(f"  INT8: {int8_size:.1f} MB")
    print(f"  压缩比: {fp32_size/int8_size:.1f}x")


if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"错误: 模型目录不存在 {MODEL_PATH}")
        sys.exit(1)

    export_onnx()
    quantize_int8()

    print("\n" + "=" * 50)
    print("  全部完成!")
    print(f"  ONNX FP32: {ONNX_PATH}")
    print(f"  ONNX INT8: {ONNX_INT8_PATH}")
    print("=" * 50)
