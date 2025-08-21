# =====================================================================
# Module: server.websocket_stream
# Purpose: 단일 WebSocket 엔드포인트에서 영상 프레임 수신, 사람 수 추정, 모자이크/녹화 처리, JPEG 스트림 송신.
# Responsibilities:
#   - 클라이언트로부터 JPEG 바이너리 프레임 수신 및 디코드
#   - 사람 수 추정(analytics.person_count) 및 presence 히스토리 관리
#   - 자동 녹화 시작/중단 조건 판별 및 이벤트 알림
#   - 검증 여부에 따라 모자이크 적용 / 해제
#   - 전송 FPS 스로틀 및 코얼레싱(보류 프레임) 처리
# Design Notes:
#   - state.stream_stats 에 per-connection 메타 저장 (프레임 카운터, presence_hist 등)
#   - 녹화 자원 접근은 recording_lock, 검증 상태는 verification_lock 일부 사용
#   - 시작/중단 race 방지를 위해 finalize 시 snapshot 사용
# Extension Tips:
#   - backpressure 심할 경우 메시지 큐( asyncio.Queue ) 도입 고려
#   - 멀티모델 추론 파이프라인 필요 시 프레임 디코드 직후 task 분할
# =====================================================================
from __future__ import annotations
import os, cv2, json, time, asyncio
import numpy as np
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from fastapi import APIRouter
from .core import state
from .config import (
    DETECT_EVERY_N, MOSAIC_EVERY_N, MOSAIC_PROCESS_MAX_WIDTH, AUTO_RECORD_TEMP_DISABLED,
    AUTO_PRESENCE_MIN_HITS, AUTO_PRESENCE_WINDOW, STREAM_MAX_WIDTH, STREAM_JPEG_QUALITY, STREAM_TARGET_FPS, USE_S3,
    AUTO_ZERO_TIMEOUT_SEC, AUTO_RECORD_DEBUG
)
import server.config as cfg  # 런타임 설정 동적 참조
from .analytics.person_count import estimate_person_count
from .recording import auto_start_recording_from_frame, finalize_and_notify
from AI_processor import process_frame

router = APIRouter()

async def _ws_send_coalesced(websocket: WebSocket, ws_id: int, first_bytes: bytes):
    try:
        await websocket.send_bytes(first_bytes)
        while True:
            st = state.stream_stats.get(ws_id)
            if not st: break
            pending = st.pop('pending_jpg', None)
            if not pending: break
            await websocket.send_bytes(pending)
    except Exception:
        pass
    finally:
        st = state.stream_stats.get(ws_id)
        if st is not None:
            st['send_busy'] = False

@router.websocket('/ws/video')
async def unified_video_ws(websocket: WebSocket):
    await websocket.accept()
    ws_id = id(websocket)
    state.active_websockets.add(websocket)
    state.stream_stats[ws_id] = {"detections":0, "auto_starts":0, "prev_person_count":0, "active": True, "last_updated": datetime.now().isoformat(timespec='seconds')}
    with state.verification_lock:
        state.verified_users[ws_id] = {"is_verified": False, "decryption_token": None, "camera_id": None}
    try:
        while True:
            try:
                message = await websocket.receive()
                if 'text' in message:
                    # key verification or disconnect command
                    try:
                        data = json.loads(message['text'])
                    except Exception:
                        await websocket.send_text(json.dumps({"type": "error", "message": "잘못된 JSON 형식"}))
                        continue
                    if data.get('type') == 'key_verification':
                        from .verification import verify_key_with_backend
                        token = data.get('accessToken'); camera_id = data.get('cameraId')
                        if not token or not camera_id:
                            await websocket.send_text(json.dumps({"type":"verification_result","success":False,"message":"accessToken,cameraId 필요"}))
                            continue
                        result = await verify_key_with_backend(token, camera_id)
                        if result['success']:
                            backend = result['data']
                            if backend.get('isSuccess') and 'data' in backend:
                                body = backend['data']
                                is_valid = body.get('valid', False)
                                can_decrypt = body.get('canDecrypt', False)
                                if is_valid and can_decrypt:
                                    with state.verification_lock:
                                        state.verified_users[ws_id] = {"is_verified": True, "decryption_token": body.get('decryptionToken'), "camera_id": camera_id}
                                    await websocket.send_text(json.dumps({"type":"verification_result","success":True,"message":"키 검증 성공","canDecrypt":True,"isValid":True,"decryptionToken": body.get('decryptionToken')}))
                                else:
                                    await websocket.send_text(json.dumps({"type":"verification_result","success":False,"message": body.get('message','검증 실패'),"canDecrypt":False,"isValid": is_valid}))
                            else:
                                await websocket.send_text(json.dumps({"type":"verification_result","success":False,"message": backend.get('message','API 오류')}))
                        else:
                            await websocket.send_text(json.dumps({"type":"verification_result","success":False,"message": result['message']}))
                    elif data.get('type') == 'disconnect':
                        st = state.stream_stats.get(ws_id, {})
                        await websocket.send_text(json.dumps({"type":"disconnect_result","success": True,"message":"Connection will be closed.","detections": int(st.get('detections',0)),"auto_starts": int(st.get('auto_starts',0))}))
                        await websocket.close(); break
                elif 'bytes' in message:
                    nparr = np.frombuffer(message['bytes'], np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue
                    st = state.stream_stats.get(ws_id) or {}
                    idx = int(st.get('frame_idx',0)) + 1
                    st['frame_idx'] = idx
                    state.stream_stats[ws_id] = st
                    # --- People Counting & Presence Update ---------------------------------
                    # person count sampling
                    if idx == 1 or DETECT_EVERY_N <= 1 or (idx % max(1, DETECT_EVERY_N) == 0):
                        pcnt = await asyncio.to_thread(estimate_person_count, frame)
                        st['last_pcnt'] = pcnt
                    else:
                        pcnt = int(st.get('last_pcnt',0))
                    if AUTO_RECORD_DEBUG:
                        state._auto_debug.update({"enabled": cfg.AUTO_RECORD_ENABLED, "threshold": cfg.AUTO_RECORD_THRESHOLD, "last_check_at": datetime.now().isoformat(timespec='seconds'), "last_person_count": pcnt, "attempted_start": False, "started": state.is_recording})
                    # detection transitions
                    prev = int(st.get('prev_person_count',0))
                    if prev == 0 and pcnt >= 1:
                        st['detections'] = int(st.get('detections',0)) + 1
                    st['prev_person_count'] = pcnt
                    st['last_updated'] = datetime.now().isoformat(timespec='seconds')
                    from collections import deque
                    hist = st.get('presence_hist')
                    if not isinstance(hist, deque):
                        hist = deque(maxlen=max(1, int(AUTO_PRESENCE_WINDOW)))
                        st['presence_hist'] = hist
                    hist.append(1 if pcnt >= 1 else 0)
                    hits = sum(hist); window_len = len(hist)
                    st['presence_hits'] = hits; st['presence_window'] = window_len
                    state._auto_debug['presence_hits'] = hits
                    state._auto_debug['presence_window'] = window_len
                    # presence history 업데이트
                    # ...existing code...
                    # auto start conditions
                    should_start = False
                    required_hits = max(1,int(AUTO_PRESENCE_MIN_HITS))
                    presence_ready = (hits >= required_hits) or (window_len < required_hits and hits == window_len and pcnt >= cfg.AUTO_RECORD_THRESHOLD)
                    if (not AUTO_RECORD_TEMP_DISABLED and not state.is_recording and cfg.AUTO_RECORD_ENABLED and pcnt >= cfg.AUTO_RECORD_THRESHOLD and presence_ready):
                        should_start = True
                    if should_start:
                        state._auto_debug['attempted_start'] = True
                        ok, reason = await auto_start_recording_from_frame(frame, pcnt, ws_id)
                        if not ok:
                            state._auto_debug['start_fail_reason'] = reason
                            try:
                                await websocket.send_text(json.dumps({"type":"auto_recording_start_failed","reason": reason, "threshold": cfg.AUTO_RECORD_THRESHOLD, "persons": pcnt}))
                            except Exception:
                                pass
                        else:
                            state._auto_debug.pop('start_fail_reason', None)
                            try:
                                st['auto_starts'] = int(st.get('auto_starts',0)) + 1
                            except Exception:
                                pass
                            if state.is_recording:
                                started_iso = datetime.fromtimestamp(state._recording_started_at_ts).isoformat(timespec='seconds') if state._recording_started_at_ts else None
                                await websocket.send_text(json.dumps({"type":"auto_recording_started","filename": state.recording_filename if state.video_writer else None, "started_at": started_iso, "initial_persons": int(pcnt), "storage": 'S3' if USE_S3 else 'local', "threshold": cfg.AUTO_RECORD_THRESHOLD}))
                    # manual start
                    if state._manual_start_requested and (not state.is_recording):
                        ok, reason = await auto_start_recording_from_frame(frame, pcnt, ws_id)
                        if not ok:
                            state._auto_debug['manual_start_fail_reason'] = reason
                            try:
                                await websocket.send_text(json.dumps({"type":"manual_recording_start_failed","reason": reason}))
                            except Exception:
                                pass
                        else:
                            state._auto_debug.pop('manual_start_fail_reason', None)
                            if state.is_recording:
                                started_iso = datetime.fromtimestamp(state._recording_started_at_ts).isoformat(timespec='seconds') if state._recording_started_at_ts else None
                                await websocket.send_text(json.dumps({"type":"manual_recording_started","filename": state.recording_filename if state.video_writer else None, "started_at": started_iso, "initial_persons": int(pcnt), "storage": 'S3' if USE_S3 else 'local', "threshold": cfg.AUTO_RECORD_THRESHOLD}))
                        state._manual_start_requested = False
                    if state.is_recording:
                        state._recording_max_persons = max(state._recording_max_persons, int(pcnt))
                    # --- Recording Stop Condition Evaluation ------------------------------
                    # stop condition (revised): 직접 person count 기준으로 타이머 측정
                    now_ts = time.time()
                    if state.is_recording:
                        if pcnt >= cfg.AUTO_RECORD_THRESHOLD:
                            # 사람 다시 감지되면 타이머 리셋
                            state._last_nonzero_person_ts = now_ts
                            state._zero_since_ts = None  # 호환성 유지
                            state._auto_debug.pop('zero_gap_sec', None)
                        else:
                            # 아직 기록 중 & threshold 미만 -> 경과 측정
                            if state._last_nonzero_person_ts is None:
                                # 녹화 시작 직후 threshold 미만 경우 대비 초기화
                                state._last_nonzero_person_ts = now_ts
                            zero_gap = now_ts - state._last_nonzero_person_ts
                            state._auto_debug['zero_gap_sec'] = round(zero_gap,2)
                            if (
                                not AUTO_RECORD_TEMP_DISABLED and zero_gap >= AUTO_ZERO_TIMEOUT_SEC
                            ):
                                if state._stop_task is None or state._stop_task.done():
                                    try:
                                        await websocket.send_text(json.dumps({"type": "auto_recording_will_finalize", "zero_gap_sec": round(zero_gap,2)}))
                                    except Exception:
                                        pass
                                    start_ts_snapshot = state._recording_started_at_ts
                                    max_persons_snapshot = state._recording_max_persons
                                    rec_fn_snapshot = state.recording_filename
                                    rec_fn_raw_snapshot = state.recording_filename_raw
                                    state._stop_task = asyncio.create_task(
                                        finalize_and_notify(
                                            websocket,
                                            start_ts_snapshot,
                                            max_persons_snapshot,
                                            rec_fn_snapshot,
                                            rec_fn_raw_snapshot,
                                        )
                                    )
                    # --- Mosaic / Anonymization Pipeline ----------------------------------
                    # 모자이크 캐싱 + 필요 프레임에서만 재계산
                    raw_frame = frame.copy()
                    with state.verification_lock:
                        v = state.verified_users.get(ws_id, {"is_verified": False})
                    if v.get('is_verified'):
                        processed_frame = raw_frame
                    else:
                        cache = st.get('last_mosaic'); need_new = True
                        if MOSAIC_EVERY_N > 1 and idx % MOSAIC_EVERY_N != 0 and cache is not None:
                            need_new = False
                        if need_new:
                            try:
                                h0,w0 = raw_frame.shape[:2]
                                if MOSAIC_PROCESS_MAX_WIDTH and MOSAIC_PROCESS_MAX_WIDTH>0 and w0 > MOSAIC_PROCESS_MAX_WIDTH:
                                    scale = MOSAIC_PROCESS_MAX_WIDTH/float(w0)
                                    small = cv2.resize(raw_frame, (int(w0*scale), int(h0*scale)))
                                    mosaic_small = await asyncio.to_thread(process_frame, small.copy(), mode='face_plate')
                                    processed_frame = cv2.resize(mosaic_small, (w0,h0))
                                else:
                                    processed_frame = await asyncio.to_thread(process_frame, raw_frame.copy(), mode='face_plate')
                                st['last_mosaic'] = processed_frame
                            except Exception:
                                processed_frame = raw_frame
                        else:
                            processed_frame = cache
                    display_frame = processed_frame
                    try:
                        if STREAM_MAX_WIDTH and STREAM_MAX_WIDTH>0:
                            h0,w0 = display_frame.shape[:2]
                            if w0 > STREAM_MAX_WIDTH:
                                sc = STREAM_MAX_WIDTH/float(w0)
                                display_frame = cv2.resize(display_frame, (int(w0*sc), int(h0*sc)))
                    except Exception:
                        pass
                    q = max(1,min(100,int(STREAM_JPEG_QUALITY)))
                    def _encode(img, quality):
                        r,a = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                        return a.tobytes() if r else b''
                    jpg_bytes = await asyncio.to_thread(_encode, display_frame, q)
                    # --- Recording Write (raw + processed) --------------------------------
                    # recording write
                    with state.recording_lock:
                        if state.is_recording:
                            if state.video_writer_raw: state.video_writer_raw.write(raw_frame)
                            if state.video_writer:
                                try:
                                    if v.get('is_verified'):
                                        cache = st.get('last_mosaic'); need_new=True
                                        if MOSAIC_EVERY_N>1 and idx % MOSAIC_EVERY_N != 0 and cache is not None:
                                            need_new=False
                                        if need_new:
                                            h0,w0=raw_frame.shape[:2]
                                            if MOSAIC_PROCESS_MAX_WIDTH and MOSAIC_PROCESS_MAX_WIDTH>0 and w0> MOSAIC_PROCESS_MAX_WIDTH:
                                                sc = MOSAIC_PROCESS_MAX_WIDTH/float(w0)
                                                small = cv2.resize(raw_frame,(int(w0*sc), int(h0*sc)))
                                                mosaic_small = process_frame(small.copy(), mode='face_plate')
                                                rec_proc = cv2.resize(mosaic_small,(w0,h0))
                                            else:
                                                rec_proc = process_frame(raw_frame.copy(), mode='face_plate')
                                            st['last_mosaic'] = rec_proc
                                        else:
                                            rec_proc = cache
                                    else:
                                        rec_proc = processed_frame
                                except Exception:
                                    rec_proc = processed_frame
                                state.video_writer.write(rec_proc)
                    # --- Throttled JPEG Send / Coalescing ---------------------------------
                    # send throttled
                    try:
                        now_gate = time.time()
                        can_send = True
                        last_ts = float(st.get('last_stream_send_ts',0))
                        if STREAM_TARGET_FPS and STREAM_TARGET_FPS>0:
                            if (now_gate - last_ts) < (1.0/STREAM_TARGET_FPS):
                                can_send = False
                            else:
                                st['last_stream_send_ts'] = now_gate
                        if can_send:
                            if st.get('send_busy'):
                                st['pending_jpg'] = jpg_bytes
                            else:
                                st['send_busy'] = True
                                asyncio.create_task(_ws_send_coalesced(websocket, ws_id, jpg_bytes))
                    except Exception:
                        try: await websocket.send_bytes(jpg_bytes)
                        except Exception: pass
            except WebSocketDisconnect:
                break
            except Exception as e:
                print('WebSocket message error', e)
                break
    finally:
        state.active_websockets.discard(websocket)
        with state.verification_lock:
            state.verified_users.pop(ws_id, None)
        try:
            if ws_id in state.stream_stats:
                state.stream_stats[ws_id]['active'] = False
                state.stream_stats[ws_id]['last_updated'] = datetime.now().isoformat(timespec='seconds')
                st = state.stream_stats.get(ws_id, {})
                state.last_stream_snapshot = {"stream_id": str(ws_id), "detections": int(st.get('detections',0)), "auto_starts": int(st.get('auto_starts',0)), "ended_at": datetime.now().isoformat(timespec='seconds')}
        except Exception:
            pass

__all__ = ['router']
