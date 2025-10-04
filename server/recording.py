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
from fastapi import APIRouter
# 원본 저장 토글
from .config import RECORD_SAVE_RAW
from .core import state
from .storage.s3 import upload_recording, list_recordings as s3_list_recordings, generate_presigned_url

router = APIRouter()

async def auto_start_recording_from_frame(frame, init_person_count: int, ws_id: int) -> Tuple[bool, str]:
    from .config import AUTO_RECORD_ENABLED, AUTO_RECORD_DEBUG, COOLDOWN_AFTER_STOP_SEC
    if not AUTO_RECORD_ENABLED:
        return False, "AUTO_RECORD_DISABLED"
    # 중단 후 쿨다운 검사
    try:
        last_stop = state._last_record_stop_ts
        now_ts = time.time()
        if last_stop is not None and (now_ts - last_stop) < float(COOLDOWN_AFTER_STOP_SEC):
            return False, "IN_COOLDOWN"
    except Exception:
        pass
    h, w = frame.shape[:2]
    if h == 0 or w == 0:
        return False, "INVALID_FRAME_SIZE"
    fps = 30.0

    def _open():
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        # 1) 우선 mp4v + .mp4 시도
        fn_proc = f'auto_recording_{ts}.mp4'
        fn_raw = f'auto_recording_{ts}_raw.mp4' if RECORD_SAVE_RAW else None
        path_proc = os.path.join(state.TEMP_DIR, fn_proc)
        path_raw = os.path.join(state.TEMP_DIR, fn_raw) if fn_raw else None
        fourcc_mp4v = cv2.VideoWriter_fourcc(*'mp4v')
        vw_p = cv2.VideoWriter(path_proc, fourcc_mp4v, fps, (w, h))
        vw_r = None
        if RECORD_SAVE_RAW and path_raw:
            vw_r = cv2.VideoWriter(path_raw, fourcc_mp4v, fps, (w, h))
        # 성공 여부 판단
        proc_ok = vw_p is not None and vw_p.isOpened()
        raw_ok = (not RECORD_SAVE_RAW) or (vw_r is not None and vw_r.isOpened())
        if proc_ok and raw_ok:
            return vw_p, vw_r, fn_proc, fn_raw
        # 2) 실패 시 MJPG + .avi 로 폴백 (컨테이너/확장자도 함께 변경)
        try:
            if vw_p is not None: vw_p.release()
        except Exception:
            pass
        try:
            if vw_r is not None: vw_r.release()
        except Exception:
            pass
        fn_proc_avi = f'auto_recording_{ts}.avi'
        fn_raw_avi = f'auto_recording_{ts}_raw.avi' if RECORD_SAVE_RAW else None
        path_proc_avi = os.path.join(state.TEMP_DIR, fn_proc_avi)
        path_raw_avi = os.path.join(state.TEMP_DIR, fn_raw_avi) if fn_raw_avi else None
        fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
        vw_p2 = cv2.VideoWriter(path_proc_avi, fourcc_mjpg, fps, (w, h))
        vw_r2 = None
        if RECORD_SAVE_RAW and path_raw_avi:
            vw_r2 = cv2.VideoWriter(path_raw_avi, fourcc_mjpg, fps, (w, h))
        proc_ok2 = vw_p2 is not None and vw_p2.isOpened()
        raw_ok2 = (not RECORD_SAVE_RAW) or (vw_r2 is not None and vw_r2.isOpened())
        if proc_ok2 and raw_ok2:
            return vw_p2, vw_r2, fn_proc_avi, fn_raw_avi
        # 최종 실패
        return None, None, None, None

    vw_p, vw_r, fn_p, fn_r = await asyncio.to_thread(_open)
    with state.recording_lock:
        if state.is_recording:
            if vw_p is not None: vw_p.release()
            if vw_r is not None: vw_r.release()
            return False, "ALREADY_RECORDING_RACE"
        opened = False
        if vw_p is not None and vw_p.isOpened() and fn_p:
            state.video_writer = vw_p
            state.recording_filename = fn_p
            opened = True
        else:
            state.video_writer = None
            state.recording_filename = fn_p  # None 일 수 있음
        if RECORD_SAVE_RAW:
            if vw_r is not None and vw_r.isOpened() and fn_r:
                state.video_writer_raw = vw_r
                state.recording_filename_raw = fn_r
                opened = True
            else:
                state.video_writer_raw = None
                state.recording_filename_raw = None
        else:
            state.video_writer_raw = None
            state.recording_filename_raw = None
    if not opened:
        # 생성 실패한 파일 제거 시도
        for fn in [state.recording_filename, state.recording_filename_raw]:
            try:
                if fn:
                    p = os.path.join(state.TEMP_DIR, fn)
                    if os.path.exists(p): os.remove(p)
            except Exception:
                pass
        return False, "VIDEO_WRITER_OPEN_FAILED"

    state.is_recording = True
    state._recording_started_at_ts = time.time()
    state._last_record_start_ts = state._recording_started_at_ts
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
    import server.config as cfg
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
        state._last_record_stop_ts = time.time()
        items: List[tuple[str,str]] = []
        if state.recording_filename:
            p = os.path.join(state.TEMP_DIR, state.recording_filename)
            if os.path.exists(p): items.append((state.recording_filename, p))
        if RECORD_SAVE_RAW and state.recording_filename_raw:
            p = os.path.join(state.TEMP_DIR, state.recording_filename_raw)
            if os.path.exists(p): items.append((state.recording_filename_raw, p))
    if not items:
        return {"error": "Recording files not found"}
    if cfg.USE_S3:
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
    from .core.state import last_user_info
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
    import server.config as cfg
    return {"is_recording": state.is_recording, "current_file": state.recording_filename if state.is_recording else None, "storage_type": "S3" if cfg.USE_S3 else 'local', "s3_configured": cfg.USE_S3}

@router.get('/recordings')
async def list_recordings():
    return s3_list_recordings()

@router.get('/recordings/{filename}')
async def get_recording_url(filename: str):
    return generate_presigned_url(filename)

__all__ = [
    'router','auto_start_recording_from_frame','stop_and_finalize_recording','finalize_and_notify'
]
