from __future__ import annotations

import os
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .engine import analyze
from .llm import enabled, enhance
from .schemas import AnalysisResponse, HealthResponse


app = FastAPI(
    title="教情智析 API",
    description="AEF-R 证据引导多模态课程反馈分析服务",
    version="0.1.0",
)

origins = [item.strip() for item in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        engine="AEF-R Lite + optional OpenAI-compatible enhancement",
        llm_enabled=enabled(),
        llm_model=os.getenv("OPENAI_MODEL") if enabled() else None,
    )


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_feedback(
    mode: str = Form(...),
    text: str = Form(""),
    image: UploadFile | None = File(None),
) -> AnalysisResponse:
    started = time.perf_counter()
    if mode not in {"text", "image", "multimodal"}:
        raise HTTPException(status_code=422, detail="mode 必须是 text、image 或 multimodal")
    uses_text = mode in {"text", "multimodal"}
    uses_image = mode in {"image", "multimodal"}
    effective_text = text.strip() if uses_text else ""

    if uses_text and not effective_text:
        raise HTTPException(status_code=422, detail="当前模式需要文字反馈")
    if uses_image and image is None:
        raise HTTPException(status_code=422, detail="当前模式需要图片")

    image_bytes = None
    image_content_type = None
    if uses_image and image is not None:
        if image.content_type not in {"image/jpeg", "image/png", "image/webp"}:
            raise HTTPException(status_code=415, detail="仅支持 JPG、PNG 和 WebP 图片")
        image_bytes = await image.read()
        image_content_type = image.content_type
        if len(image_bytes) > 8 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="图片不能超过 8 MB")

    try:
        # Strict modality isolation: unused fields are discarded before either
        # the deterministic engine or the LLM sees the request.
        result = analyze(mode, effective_text, image_bytes)
        result = await enhance(result, effective_text, image_bytes, image_content_type)
        result.latency_ms = max(1, round((time.perf_counter() - started) * 1000))
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法解析输入：{exc}") from exc
