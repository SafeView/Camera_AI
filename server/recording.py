# =====================================================================
# Module: server.recording
# Purpose: 자동/수동 녹화 시작 및 종료, 결과 업로드/알림 처리.
# Responsibilities:
#   - VideoWriter 초기화/해제 (processed/raw)
#   - 녹화 상태 전환 및 메타 데이터(state.*) 업데이트
#   - S3 업로드 또는 로컬 저장 반환
#   - WebSocket 알림(finalize) 및 외부(Spring) 엔드포인트 후처리
# Design Notes:
#   - OpenCV VideoWriter 열기 실패 시 fallback fourcc 시도
#   - 동시성: recording_lock 으로 다중 코루틴 경쟁 제어
#   - finalize_and_notify 는 stop snapshot 을 캡쳐해 race 최소화
# Extension Tips:
#   - 업로드 큐잉/백그라운드 워커 도입 시 stop_and_finalize_recording 분리 가능
#   - 최소 녹화 길이/세그먼트 merge 로직 추가 가능
# =====================================================================
from __future__ import annotations
import os, time, json, asyncio
from datetime import datetime
from typing import Any, Dict, List, Tuple
import cv2
from fastapi import APIRouter, Body
from .config import USE_S3
from .core import state
from .storage.s3 import upload_recording, list_recordings as s3_list_recordings, generate_presigned_url

router = APIRouter()

async def auto_start_recording_from_frame(frame, init_person_count: int, ws_id: int) -> Tuple[bool, str]:
    from .config import AUTO_RECORD_ENABLED, AUTO_RECORD_DEBUG
    if not AUTO_RECORD_ENABLED:
        return False, "AUTO_RECORD_DISABLED"
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return False, "INVALID_FRAME_SIZE"
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def _open():
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        fn_proc = f'auto_recording_{ts}.mp4'
        fn_raw = f'auto_recording_{ts}_raw.mp4'
        path_proc = os.path.join(state.TEMP_DIR, fn_proc)
        path_raw = os.path.join(state.TEMP_DIR, fn_raw)
        vw_p = cv2.VideoWriter(path_proc, fourcc, fps, (w, h))
        if not vw_p.isOpened():
            try:
                fallback = cv2.VideoWriter_fourcc(*'MJPG')
                vw_p = cv2.VideoWriter(path_proc, fallback, fps, (w, h))
            except Exception:
                pass
        vw_r = cv2.VideoWriter(path_raw, fourcc, fps, (w, h))
        if not vw_r.isOpened():
            try:
                fallback = cv2.VideoWriter_fourcc(*'MJPG')
                vw_r = cv2.VideoWriter(path_raw, fallback, fps, (w, h))
            except Exception:
                pass
        return vw_p, vw_r, fn_proc, fn_raw

    vw_p, vw_r, fn_p, fn_r = await asyncio.to_thread(_open)
    with state.recording_lock:
        if state.is_recording:
            if vw_p is not None: vw_p.release()
            if vw_r is not None: vw_r.release()
            return False, "ALREADY_RECORDING_RACE"
        opened = False
        if vw_p is not None and vw_p.isOpened():
            state.video_writer = vw_p
            state.recording_filename = fn_p
            opened = True
        else:
            state.video_writer = None
            state.recording_filename = fn_p  # 이름은 유지 (파일 없을 수 있음)
        if vw_r is not None and vw_r.isOpened():
            state.video_writer_raw = vw_r
            state.recording_filename_raw = fn_r
            opened = True
        else:
            state.video_writer_raw = None
            state.recording_filename_raw = fn_r
    if not opened:
        # 생성 실패한 파일 제거 시도
        try:
            if state.recording_filename:
                p = os.path.join(state.TEMP_DIR, state.recording_filename)
                if os.path.exists(p): os.remove(p)
        except Exception:
            pass
        try:
            if state.recording_filename_raw:
                p = os.path.join(state.TEMP_DIR, state.recording_filename_raw)
                if os.path.exists(p): os.remove(p)
        except Exception:
            pass
        return False, "VIDEO_WRITER_OPEN_FAILED"

    state.is_recording = True
    state._recording_started_at_ts = time.time()
    state._recording_max_persons = max(0, int(init_person_count))
    state._recording_ws_id = ws_id
    # 자동 중단 타이머 기준 초기화
    try:
        state._last_nonzero_person_ts = state._recording_started_at_ts
        state._zero_since_ts = None
    except Exception:
        pass
    if AUTO_RECORD_DEBUG:
        print(f"[AUTO] recording started processed={state.recording_filename if state.video_writer else 'NONE'} raw={state.recording_filename_raw if state.video_writer_raw else 'NONE'} {w}x{h}")
    return True, "OK"

async def stop_and_finalize_recording() -> Dict[str, Any]:
    with state.recording_lock:
        if not state.is_recording:
            return {"error": "Not recording"}
        state.is_recording = False
        try:
            if state.video_writer: state.video_writer.release()
        except Exception: pass
        finally: state.video_writer = None
        try:
            if state.video_writer_raw: state.video_writer_raw.release()
        except Exception: pass
        finally: state.video_writer_raw = None
        items: List[tuple[str,str]] = []
        if state.recording_filename:
            p = os.path.join(state.TEMP_DIR, state.recording_filename)
            if os.path.exists(p): items.append((state.recording_filename, p))
        if state.recording_filename_raw:
            p = os.path.join(state.TEMP_DIR, state.recording_filename_raw)
            if os.path.exists(p): items.append((state.recording_filename_raw, p))
    if not items:
        return {"error": "Recording files not found"}
    if USE_S3:
        async def _upload(fn, path):
            ok, res = await asyncio.to_thread(upload_recording, path, fn)
            try: os.remove(path)
            except Exception: pass
            return fn, ok, res
        results = await asyncio.gather(*[_upload(fn,p) for fn,p in items])
        urls = []
        errs = []
        for fn, ok, res in results:
            if ok: urls.append(res)
            else: errs.append({fn: res})
        return {"status": "ok", "filenames": [fn for fn,_ in items], "s3_urls": urls, "errors": errs or None, "storage": "S3"}
    else:
        return {"status": "ok", "filenames": [fn for fn,_ in items], "local_paths": [p for _,p in items], "storage": "local"}

async def finalize_and_notify(websocket, started_ts, max_persons, rec_fn, rec_fn_raw):
    from .core.state import last_user_info, _recording_max_persons, _recording_ws_id, _recording_started_at_ts
    from .config import SPRING_MAKE_ENTITY_URL
    res = await stop_and_finalize_recording()
    duration = None
    if started_ts: duration = round(time.time() - started_ts, 2)
    payload = {"type": "auto_recording_finalized", "filename": rec_fn, "segment_max_persons": int(max_persons), "duration_sec": duration, "storage": res.get("storage")}
    if res.get("s3_urls") and rec_fn:
        try:
            pu = next((u for u in res['s3_urls'] if u.endswith(f"/{rec_fn}")), None)
            if pu: payload['s3_url'] = pu
        except Exception: pass
    try:
        await websocket.send_text(json.dumps(payload))
    except Exception as e:
        print("WS notify error", e)
    if last_user_info and res.get('s3_urls'):
        uid = last_user_info.get('userId') if isinstance(last_user_info, dict) else None
        if uid:
            async def _notify():
                import aiohttp
                body = {"userId": uid, "urls": res.get('s3_urls', [])}
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.post(SPRING_MAKE_ENTITY_URL, json=body, timeout=10) as resp:
                            # 응답 소비하여 연결 정상 종료
                            try:
                                await resp.read()
                            except Exception:
                                pass
                except Exception:
                    pass
            asyncio.create_task(_notify())
    state._recording_max_persons = 0
    state._recording_ws_id = None
    state._recording_started_at_ts = None

@router.post('/start_recording')
async def start_recording():
    with state.recording_lock:
        if state.is_recording:
            return {"error": "Already recording"}
        state._manual_start_requested = True
    return {"status": "Recording will start on next frame", "pending": True}

@router.post('/stop_recording')
async def stop_recording():
    return await stop_and_finalize_recording()

@router.get('/recording_status')
async def recording_status():
    from .config import USE_S3
    return {"is_recording": state.is_recording, "current_file": state.recording_filename if state.is_recording else None, "storage_type": "S3" if USE_S3 else 'local', "s3_configured": USE_S3}

@router.get('/recordings')
async def list_recordings():
    return s3_list_recordings()

@router.get('/recordings/{filename}')
async def get_recording_url(filename: str):
    return generate_presigned_url(filename)

__all__ = [
    'router','auto_start_recording_from_frame','stop_and_finalize_recording','finalize_and_notify'
]
