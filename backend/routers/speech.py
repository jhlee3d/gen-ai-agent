import os
import io
from typing import Annotated

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from routers.chat import ChatRequest, chat as chat_endpoint   # ← 기존 /chat 재사용
from .auth   import get_current_user_token                    # JWT 검증
from database import SessionLocal
import models

router = APIRouter(prefix="/speech", tags=["speech"])

# ── helpers ─────────────────────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

async def whisper_stt(file: UploadFile):
    raise HTTPException(501, "음성 기능은 현재 지원되지 않습니다.")

# ────────────────────────────────────────────────────
@router.post("/chat", status_code=201)
async def speech_chat(
    conversation_id: int | None = Form(None),
    timezone:        str | None = Form(None),
    audio: UploadFile = File(...),
    db : Session     = Depends(get_db),
    me : models.User = Depends(get_current_user_token)
):
    """
    • 짧은 음성 녹음을 받아 Whisper v3 로 전사  
    • /chat 엔드포인트에 그대로 전달해 답변·툴콜 처리까지 한 번에 수행  
    • 응답: /chat 결과 + {"stt_confidence": float}
    """
    text, conf = await whisper_stt(audio)
    await audio.close()

    print(f"conversation_id={conversation_id}")
    print(f"timezone={timezone}")

    req = ChatRequest(
        conversation_id = int(conversation_id) if conversation_id else None,
        question        = text,
        timezone        = timezone
    )
    # 기존 chat 로직 호출 (의존성 직접 주입)
    resp = chat_endpoint(req, db=db, me=me)
    resp["stt_confidence"] = conf
    resp["transcript"]     = text
    return JSONResponse(resp)

# 필요하면 “STT만” 하는 단일 엔드포인트도 유지
@router.post("/stt")
async def stt_only(
    audio: Annotated[UploadFile, File(description="≤ 25 MB audio")]
):
    text, conf = await whisper_stt(audio)
    await audio.close()
    return {"text": text, "confidence": conf}
