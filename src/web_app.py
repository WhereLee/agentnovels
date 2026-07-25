"""
Web 服务：小说上传 + 管理 + 切块处理
FastAPI + 原生 HTML 前端
"""
import json
import shutil
import time
from pathlib import Path
from typing import List, Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from encoding_detector import read_file_auto_encoding, detect_encoding
from chapter_detector import detect_chapters
from chunker import chunk_chapters, DEFAULT_CHUNK_SIZE, DEFAULT_OVERLAP
from database import get_db_path, init_db, save_chapters, save_chunks as db_save_chunks, get_chunk_count

# === 路径配置 ===
PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOVELS_DIR = PROJECT_ROOT / "novels"
TEMPLATES_DIR = PROJECT_ROOT / "templates"

app = FastAPI(title="Novel RAG Manager", version="2.0")


# ============================================================
# API 路由
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def index():
    """主页"""
    html_file = TEMPLATES_DIR / "index.html"
    if not html_file.exists():
        return HTMLResponse("<h1>错误：未找到 templates/index.html</h1>", status_code=500)
    return HTMLResponse(html_file.read_text(encoding='utf-8'))


@app.get("/api/novels")
async def list_novels():
    """列出所有已上传的小说"""
    novels = []
    if not NOVELS_DIR.exists():
        return {"novels": []}

    for novel_dir in sorted(NOVELS_DIR.iterdir()):
        if not novel_dir.is_dir():
            continue
        meta_file = novel_dir / "meta.json"
        if meta_file.exists():
            with open(meta_file, 'r', encoding='utf-8') as f:
                meta = json.load(f)
            novels.append(meta)

    return {"novels": novels}


@app.post("/api/upload")
async def upload_novel(file: UploadFile = File(...)):
    """
    上传小说文件（.txt）
    
    处理流程：
    1. 保存原文
    2. 检测编码
    3. 读取全文
    4. 检测章节
    5. 切块
    6. 生成 meta.json
    """
    # 验证文件类型
    if not file.filename.endswith('.txt'):
        raise HTTPException(status_code=400, detail="仅支持 .txt 文件")

    # 从文件名提取小说名（去掉扩展名）
    novel_name = Path(file.filename).stem.strip()
    if not novel_name:
        raise HTTPException(status_code=400, detail="文件名无效")

    # 创建小说目录
    novel_dir = NOVELS_DIR / novel_name
    if novel_dir.exists():
        raise HTTPException(status_code=409, detail=f"《{novel_name}》已存在，请先删除再重新上传")

    novel_dir.mkdir(parents=True, exist_ok=True)

    try:
        # === 1. 保存原文 ===
        original_path = novel_dir / "original.txt"
        content_bytes = await file.read()
        with open(original_path, 'wb') as f:
            f.write(content_bytes)

        # === 2. 检测编码 ===
        encoding, confidence = detect_encoding(str(original_path))

        # === 3. 读取全文 ===
        text, encoding, confidence = read_file_auto_encoding(str(original_path))
        total_chars = len(text)

        if total_chars < 100:
            # 文件太短，清理并报错
            shutil.rmtree(novel_dir)
            raise HTTPException(status_code=400, detail="文件内容过短（<100字），请检查文件是否正确")

        # === 4. 检测章节 ===
        detection_result = detect_chapters(text)

        # === 5. 写入 SQLite ===
        db_path = get_db_path(str(novel_dir))
        init_db(db_path)
        save_chapters(db_path, detection_result.chapters)

        # === 6. 切块 ===
        chunks = chunk_chapters(detection_result.chapters)
        db_save_chunks(db_path, chunks, strategy="fixed")

        # === 7. 生成 meta.json ===
        meta = {
            "name": novel_name,
            "encoding": encoding,
            "encoding_confidence": round(confidence, 3),
            "total_chars": total_chars,
            "upload_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "chapter_detected": detection_result.detected,
            "chapter_pattern": detection_result.pattern_name,
            "chapter_count": len(detection_result.chapters),
            "chunk_count": len(chunks),
            "chunk_strategy": "fixed",
            "chunk_params": {
                "size": DEFAULT_CHUNK_SIZE,
                "overlap": DEFAULT_OVERLAP,
            },
            "chapters": [
                {
                    "index": ch["index"],
                    "title": ch["title"],
                    "chars": ch["chars"],
                }
                for ch in detection_result.chapters
            ],
            "status": "ready",
            "message": detection_result.message,
        }

        meta_file = novel_dir / "meta.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        return JSONResponse(content={
            "success": True,
            "message": f"《{novel_name}》处理完成",
            "meta": meta,
        })

    except HTTPException:
        raise
    except Exception as e:
        # 处理失败，清理目录
        if novel_dir.exists():
            shutil.rmtree(novel_dir)
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")


@app.delete("/api/novels/{novel_name}")
async def delete_novel(novel_name: str):
    """删除一本小说的所有数据"""
    novel_dir = NOVELS_DIR / novel_name
    if not novel_dir.exists():
        raise HTTPException(status_code=404, detail=f"《{novel_name}》不存在")

    shutil.rmtree(novel_dir)
    return {"success": True, "message": f"《{novel_name}》已删除"}


# ============================================================
# 工具函数
# ============================================================

def _sanitize_filename(name: str) -> str:
    """移除文件名中的非法字符"""
    import re
    # 移除 Windows 非法字符
    name = re.sub(r'[<>:"/\\|?*]', '', name)
    # 移除控制字符
    name = re.sub(r'[\x00-\x1f]', '', name)
    return name.strip() or "untitled"


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  Novel RAG Manager")
    print(f"  访问: http://127.0.0.1:8000")
    print(f"  数据目录: {NOVELS_DIR}")
    print("=" * 50)
    uvicorn.run(app, host="127.0.0.1", port=8000)
