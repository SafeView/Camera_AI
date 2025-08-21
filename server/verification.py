# =====================================================================
# Module: server.verification
# Purpose: 카메라/클라이언트 키 검증 및 사용자 식별 정보 수신, 모자이크 강제 토글.
# Responsibilities:
#   - 백엔드 연동하여 accessToken + cameraId 검증
#   - userId 수신 및 server.core.state 에 저장
#   - 전체 연결에 대한 검증 상태 통계 제공
#   - force_mosaic 로 모든 연결을 비검증 상태로 전환
# Design Notes:
#   - aiohttp 를 통한 외부 HTTP 호출 (timeout=10)
#   - 실패시 예외 메시지를 사용자 친화적으로 래핑
# Extension Tips:
#   - 실패 재시도(backoff) 필요 시 verify_key_with_backend 내부 확장
#   - audit 로깅/레이트리밋 추가 고려
# =====================================================================
from __future__ import annotations
import json, time
from datetime import datetime
from typing import Dict, Any
import aiohttp
from fastapi import APIRouter, Body
from .core import state
from .config import AI_API_KEY, BACKEND_API_URL

router = APIRouter()

async def verify_key_with_backend(access_token: str, camera_id: str):
    if not AI_API_KEY:
        return {"success": False, "message": "AI API 키가 설정되지 않았습니다."}
    url = f"{BACKEND_API_URL}/api/decryption/keys/verify/ai"
    headers = {"AiApiKey": AI_API_KEY, "Content-Type": "application/json"}
    data = {"accessToken": access_token, "cameraId": camera_id}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data, timeout=10) as resp:
                if resp.status == 200:
                    return {"success": True, "data": await resp.json()}
                return {"success": False, "message": f"키 검증 실패: {await resp.text()}"}
    except Exception as e:  # pragma: no cover
        return {"success": False, "message": f"백엔드 연결 오류: {e}"}

@router.post('/client/user')
async def receive_user_id(payload: Dict[str, Any] = Body(default={})):
    if not isinstance(payload, dict):
        return {"success": False, "message": "JSON body가 필요합니다."}
    user_id = payload.get('userId')
    if not user_id or not isinstance(user_id, str):
        return {"success": False, "message": "userId(string)은 필수입니다."}
    ts = datetime.now().isoformat(timespec='seconds')
    state.last_user_info = {"userId": user_id, "received_at": ts}
    return {"success": True, "message": "userId 수신 완료", "received": {"userId": user_id}, "storedAt": ts}

@router.get('/verification_status')
async def get_verification_status():
    with state.verification_lock:
        verified_count = sum(1 for v in state.verified_users.values() if v.get('is_verified'))
        total = len(state.verified_users)
    return {"total_connections": total, "verified_connections": verified_count, "unverified_connections": total - verified_count}

@router.post('/force_mosaic')
async def force_mosaic_all():
    with state.verification_lock:
        for wsid in state.verified_users:
            state.verified_users[wsid]['is_verified'] = False
            state.verified_users[wsid]['decryption_token'] = None
    disconnected = 0
    for ws in list(state.active_websockets):
        try:
            await ws.send_text(json.dumps({"type": "force_mosaic", "message": "모든 연결에 모자이크가 강제 적용되었습니다."}))
        except Exception:
            disconnected += 1
    return {"message": "모든 사용자에게 모자이크가 강제 적용되었습니다.", "affected_connections": len(state.verified_users), "disconnected": disconnected}

__all__ = ['router','verify_key_with_backend']
