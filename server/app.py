# =====================================================================
# Module: server.app
# Purpose: FastAPI 애플리케이션 구성 및 라우터/메타 엔드포인트 등록.
# Responsibilities:
#   - CORS / 미들웨어 설정
#   - WebSocket / Recording / Verification / Face-Time 라우터 포함
#   - Auto-recording 런타임 조정 및 상태 조회 엔드포인트 제공
# Design Notes:
#   - 설정 변경 (/auto_recording POST)은 프로세스 전역 변수 수정 -> 멀티프로세스 환경 재검토 필요
#   - 상태 스냅샷은 server.core.state 를 통해 공유
# Extension Tips:
#   - 대규모 설정 변경 필요 시 pydantic Settings 도입 고려
#   - 관리자 전용 엔드포인트 분리 시 별도 prefix 라우터 제안
# =====================================================================
from __future__ import annotations
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .websocket_stream import router as ws_router
from .recording import router as recording_router
from .verification import router as verification_router
from .face_time_api import router as face_time_router
from .core import state
from .config import (
    AUTO_RECORD_ENABLED, AUTO_RECORD_THRESHOLD, AUTO_ZERO_TIMEOUT_SEC, AUTO_RECORD_DEBUG,
    USE_S3
)
from .config import STREAM_MAX_WIDTH, STREAM_JPEG_QUALITY, SERVER_REV
from fastapi import APIRouter, Body
import time
from datetime import datetime
import os
try:
    import cv2
    try:
        cv2.setNumThreads(1)
    except Exception:
        pass
except Exception:
    pass
# 과도한 BLAS/OpenMP 스레드 제한
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('NUMEXPR_NUM_THREADS', '1')

app = FastAPI(title="SafeView FastAPI Server", version="refactor-1", description="Modularized server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(ws_router)
app.include_router(recording_router)
app.include_router(verification_router)
app.include_router(face_time_router)

meta_router = APIRouter()

@meta_router.get("/health")
async def health():
    return {"status": "ok", "rev": SERVER_REV}

@meta_router.get("/last_detections")
async def get_last_detections():
    if state.last_stream_snapshot is None:
        return {"available": False, "message": "아직 종료된 스트림이 없습니다."}
    snap = state.last_stream_snapshot
    return {"available": True, **snap}

@meta_router.get("/auto_recording")
async def get_auto_recording():
    return {
        "enabled": AUTO_RECORD_ENABLED,
        "threshold": AUTO_RECORD_THRESHOLD,
        "is_recording": state.is_recording,
        "s3_configured": USE_S3
    }

# NOTE: 단순 전역 변수 변경 (프로세스 내) -> 프로덕션에서는 재설계 고려
@meta_router.post("/auto_recording")
async def set_auto_recording(payload: dict = Body(default={})):  # {enabled, threshold}
    import server.config as cfg  # runtime import to mutate
    if not isinstance(payload, dict):
        return {"error": "Invalid body"}
    if "enabled" in payload:
        cfg.AUTO_RECORD_ENABLED = bool(payload["enabled"])
    if "threshold" in payload:
        try:
            th = int(payload["threshold"])
            if th < 1:
                return {"error": "threshold must be >=1"}
            cfg.AUTO_RECORD_THRESHOLD = th
        except Exception:
            return {"error": "threshold invalid"}
    return {"enabled": cfg.AUTO_RECORD_ENABLED, "threshold": cfg.AUTO_RECORD_THRESHOLD}

@meta_router.get("/auto_recording/debug")
async def auto_recording_debug():
    state._auto_debug.update({
        "is_recording": state.is_recording,
        "last_nonzero_person_ts": state._last_nonzero_person_ts,
        "zero_since_ts": state._zero_since_ts,
        "zero_timeout_sec": AUTO_ZERO_TIMEOUT_SEC,
    })
    if state._last_nonzero_person_ts is not None:
        try:
            state._auto_debug["seconds_since_last_nonzero"] = round(time.time() - state._last_nonzero_person_ts, 2)
        except Exception:
            pass
    return dict(state._auto_debug)

@meta_router.post("/disconnect_ws")
async def disconnect_ws():
    closed = 0
    for ws in list(state.active_websockets):
        try:
            ws_id = id(ws)
            st = state.stream_stats.get(ws_id, {})
            try:
                await ws.send_text(
                    __import__('json').dumps({
                        "type": "final_stats",
                        "detections": int(st.get("detections", 0)),
                        "auto_starts": int(st.get("auto_starts", 0)),
                        "message": "Connection will be closed by server."
                    })
                )
            except Exception:
                pass
            await ws.close()
            closed += 1
        except Exception:
            pass
    return {"success": True, "disconnected": closed}

app.include_router(meta_router)

__all__ = ["app"]
