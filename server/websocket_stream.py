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
import cv2, json, time, asyncio
import numpy as np
from datetime import datetime
from fastapi import WebSocket, WebSocketDisconnect
from fastapi import APIRouter
from .core import state
from .config import (
    DETECT_EVERY_N, MOSAIC_EVERY_N, MOSAIC_PROCESS_MAX_WIDTH, AUTO_RECORD_TEMP_DISABLED,
    AUTO_PRESENCE_MIN_HITS, AUTO_PRESENCE_WINDOW, STREAM_MAX_WIDTH, STREAM_JPEG_QUALITY, STREAM_TARGET_FPS, USE_S3,
    AUTO_ZERO_TIMEOUT_SEC, AUTO_RECORD_DEBUG, RECORD_BY_MOSAIC
)
# 안정화 파라미터 추가 임포트
from .config import MIN_RECORD_DURATION_SEC
import server.config as cfg  # 런타임 설정 동적 참조
from .analytics.person_count import estimate_person_count
from .recording import auto_start_recording_from_frame, finalize_and_notify, stop_and_finalize_recording
from AI_processor import process_frame, process_frame_with_meta

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
                    # --- People Counting & Presence Update (skip when RECORD_BY_MOSAIC) ----
                    pcnt = int(st.get('last_pcnt', 0))  # mosaic 모드 안전 기본값
                    hits = int(st.get('presence_hits', 0))
                    window_len = int(st.get('presence_window', 0))
                    if not RECORD_BY_MOSAIC:
                        if idx == 1 or DETECT_EVERY_N <= 1 or (idx % max(1, DETECT_EVERY_N) == 0):
                            pcnt = await asyncio.to_thread(estimate_person_count, frame)
                            st['last_pcnt'] = pcnt
                        else:
                            pcnt = int(st.get('last_pcnt',0))
                        if AUTO_RECORD_DEBUG:
                            state._auto_debug.update({"enabled": cfg.AUTO_RECORD_ENABLED, "threshold": cfg.AUTO_RECORD_THRESHOLD, "last_check_at": datetime.now().isoformat(timespec='seconds'), "last_person_count": pcnt, "attempted_start": False, "started": state.is_recording})
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
                        # presence history 기반 auto start
                        should_start = False
                        required_hits = max(1,int(AUTO_PRESENCE_MIN_HITS))
                        presence_ready = (hits >= required_hits)
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
                        now_ts = time.time()
                        if state.is_recording:
                            # 감지 재등장 시 중지 예약 취소
                            if pcnt >= cfg.AUTO_RECORD_THRESHOLD:
                                state._last_nonzero_person_ts = now_ts
                                state._zero_since_ts = None
                                # 예약된 중단 태스크가 있으면 취소
                                try:
                                    if state._stop_task is not None and not state._stop_task.done():
                                        state._stop_task.cancel(); state._stop_task = None
                                except Exception:
                                    pass
                                state._auto_debug.pop('zero_gap_sec', None)
                            else:
                                if state._last_nonzero_person_ts is None:
                                    state._last_nonzero_person_ts = now_ts
                                zero_gap = now_ts - state._last_nonzero_person_ts
                                state._auto_debug['zero_gap_sec'] = round(zero_gap,2)
                                # 최소 녹화 시간 보장
                                rec_dur = 0.0
                                try:
                                    if state._recording_started_at_ts:
                                        rec_dur = now_ts - state._recording_started_at_ts
                                        state._auto_debug['record_duration_sec'] = round(rec_dur,2)
                                except Exception:
                                    pass
                                if rec_dur < float(MIN_RECORD_DURATION_SEC):
                                    # 최소 시간 전에는 중단 예약 금지
                                    pass
                                else:
                                    if (not AUTO_RECORD_TEMP_DISABLED and zero_gap >= AUTO_ZERO_TIMEOUT_SEC):
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
                    # 모자이크 캐싱 + 필요 프레임에서만 재계산 (메타 포함)
                    raw_frame = frame.copy()
                    with state.verification_lock:
                        v = state.verified_users.get(ws_id, {"is_verified": False})
                    cache = st.get('last_mosaic'); cache_meta = st.get('last_mosaic_meta')
                    need_new = True
                    if MOSAIC_EVERY_N > 1 and idx % MOSAIC_EVERY_N != 0 and cache is not None and cache_meta is not None:
                        need_new = False
                    if need_new:
                        try:
                            h0,w0 = raw_frame.shape[:2]
                            mode = 'face'
                            if MOSAIC_PROCESS_MAX_WIDTH and MOSAIC_PROCESS_MAX_WIDTH>0 and w0 > MOSAIC_PROCESS_MAX_WIDTH:
                                scale = MOSAIC_PROCESS_MAX_WIDTH/float(w0)
                                small = cv2.resize(raw_frame, (int(w0*scale), int(h0*scale)))
                                mosaic_small, meta = await asyncio.to_thread(process_frame_with_meta, small.copy(), mode)
                                processed_frame = cv2.resize(mosaic_small, (w0,h0))
                            else:
                                processed_frame, meta = await asyncio.to_thread(process_frame_with_meta, raw_frame.copy(), mode)
                            st['last_mosaic'] = processed_frame
                            st['last_mosaic_meta'] = meta
                        except Exception:
                            processed_frame = raw_frame
                            meta = {"faces": 0}
                            st['last_mosaic'] = processed_frame
                            st['last_mosaic_meta'] = meta
                    else:
                        processed_frame = cache
                        meta = cache_meta if isinstance(cache_meta, dict) else {"faces": 0}
                    # 디버그: faces, faces_fresh 변화 또는 1초 주기로 메타 전송
                    try:
                        last_faces = int(st.get('last_faces', -1))
                        last_fresh = int(st.get('last_faces_fresh', -1))
                        last_meta_ts = float(st.get('last_meta_ts', 0.0))
                        now_ts_dbg = time.time()
                        faces_now = int(meta.get('faces', 0))
                        faces_fresh_now = int(meta.get('faces_fresh', -1)) if 'faces_fresh' in meta else -1
                        if faces_now != last_faces or faces_fresh_now != last_fresh or (now_ts_dbg - last_meta_ts) >= 1.0:
                            st['last_faces'] = faces_now
                            st['last_faces_fresh'] = faces_fresh_now
                            st['last_meta_ts'] = now_ts_dbg
                            state.stream_stats[ws_id] = st
                            payload = {"type":"mosaic_meta","faces":faces_now}
                            if faces_fresh_now >= 0:
                                payload["faces_fresh"] = faces_fresh_now
                            await websocket.send_text(json.dumps(payload))
                    except Exception:
                        pass
                    # RECORD_BY_MOSAIC: 메타 기반 즉시 녹화 제어 (항상 적용) + faces_fresh 기준 시간 중단
                    faces = int(meta.get('faces', 0))
                    faces_fresh = int(meta.get('faces_fresh', -1)) if 'faces_fresh' in meta else -1
                    now_ts_m = time.time()
                    # 모자이크 기반 presence history (신선 감지 우선)
                    from collections import deque
                    m_hist = st.get('mosaic_presence_hist')
                    if not isinstance(m_hist, deque):
                        m_hist = deque(maxlen=max(1, int(AUTO_PRESENCE_WINDOW)))
                        st['mosaic_presence_hist'] = m_hist
                    fresh_active = (faces_fresh > 0) if faces_fresh >= 0 else (faces > 0)
                    m_hist.append(1 if fresh_active else 0)
                    st['mosaic_presence_hits'] = sum(m_hist)
                    st['mosaic_presence_window'] = len(m_hist)
                    # 시작 조건: 신선 감지/히스토리 충족 시에만
                    if not state.is_recording and RECORD_BY_MOSAIC:
                        required_hits = max(1,int(AUTO_PRESENCE_MIN_HITS))
                        presence_ready_m = (st['mosaic_presence_hits'] >= required_hits)
                        if fresh_active and presence_ready_m:
                            state._auto_debug['attempted_start'] = True
                            try:
                                await websocket.send_text(json.dumps({"type":"auto_recording_attempt","by":"mosaic","faces":faces,"fresh":faces_fresh}))
                            except Exception:
                                pass
                            ok, reason = await auto_start_recording_from_frame(raw_frame, faces, ws_id)
                            if not ok:
                                state._auto_debug['start_fail_reason'] = reason
                                try:
                                    await websocket.send_text(json.dumps({"type":"auto_recording_start_failed","reason": reason, "by":"mosaic"}))
                                except Exception:
                                    pass
                            else:
                                state._auto_debug.pop('start_fail_reason', None)
                                try:
                                    st['auto_starts'] = int(st.get('auto_starts',0)) + 1
                                except Exception:
                                    pass
                                if state.is_recording:
                                    try:
                                        started_iso = datetime.fromtimestamp(state._recording_started_at_ts).isoformat(timespec='seconds') if state._recording_started_at_ts else None
                                        await websocket.send_text(json.dumps({
                                            "type":"auto_recording_started",
                                            "filename": state.recording_filename if state.video_writer else None,
                                            "started_at": started_iso,
                                            "initial_persons": int(faces),
                                            "storage": 'S3' if USE_S3 else 'local',
                                            "by": "mosaic"
                                        }))
                                    except Exception:
                                        pass
                    # 중단 로직: 모자이크 모드에서는 faces_fresh>0일 때만 '최근 감지 시각' 갱신
                    if state.is_recording and RECORD_BY_MOSAIC:
                        if faces_fresh > 0:
                            state._last_nonzero_person_ts = now_ts_m
                            state._zero_since_ts = None
                            # 예약된 중단 태스크가 있으면 취소
                            try:
                                if state._stop_task is not None and not state._stop_task.done():
                                    state._stop_task.cancel(); state._stop_task = None
                            except Exception:
                                pass
                            state._auto_debug.pop('zero_gap_sec', None)
                        else:
                            if state._last_nonzero_person_ts is None:
                                state._last_nonzero_person_ts = now_ts_m
                            zero_gap = now_ts_m - state._last_nonzero_person_ts
                            state._auto_debug['zero_gap_sec'] = round(zero_gap, 2)
                            # 최소 녹화 시간 보장
                            rec_dur = 0.0
                            try:
                                if state._recording_started_at_ts:
                                    rec_dur = now_ts_m - state._recording_started_at_ts
                                    state._auto_debug['record_duration_sec'] = round(rec_dur,2)
                            except Exception:
                                pass
                            if rec_dur < float(MIN_RECORD_DURATION_SEC):
                                pass
                            else:
                                if (not AUTO_RECORD_TEMP_DISABLED and zero_gap >= AUTO_ZERO_TIMEOUT_SEC):
                                    if state._stop_task is None or state._stop_task.done():
                                        try:
                                            await websocket.send_text(json.dumps({"type": "auto_recording_will_finalize", "zero_gap_sec": round(zero_gap,2), "by":"mosaic"}))
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
                    # --- Choose frame to stream (verified -> raw, else mosaic) -------------
                    to_stream = raw_frame if v.get('is_verified') else processed_frame
                    # --- Write frames to recorders if recording --------------------------------
                    if state.is_recording:
                        try:
                            if state.video_writer is not None:
                                state.video_writer.write(processed_frame)
                        except Exception:
                            pass
                        try:
                            if state.video_writer_raw is not None:
                                state.video_writer_raw.write(raw_frame)
                        except Exception:
                            pass
                    # --- Streaming: resize, JPEG encode, throttle & coalesce -----------------
                    try:
                        h, w = to_stream.shape[:2]
                        if STREAM_MAX_WIDTH and STREAM_MAX_WIDTH>0 and w > STREAM_MAX_WIDTH:
                            scale = STREAM_MAX_WIDTH/float(w)
                            tw = STREAM_MAX_WIDTH; th = int(h*scale)
                            to_send = cv2.resize(to_stream, (tw, th))
                        else:
                            to_send = to_stream
                        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(STREAM_JPEG_QUALITY)]
                        ok, jpg = cv2.imencode('.jpg', to_send, encode_param)
                        if not ok:
                            continue
                        jpg_bytes = jpg.tobytes()
                        now = time.time()
                        last_sent = float(st.get('last_sent_ts', 0.0))
                        min_dt = 1.0 / max(1.0, float(STREAM_TARGET_FPS))
                        if (now - last_sent) >= min_dt and not st.get('send_busy', False):
                            st['send_busy'] = True
                            st['last_sent_ts'] = now
                            state.stream_stats[ws_id] = st
                            asyncio.create_task(_ws_send_coalesced(websocket, ws_id, jpg_bytes))
                        else:
                            # coalesce latest frame
                            st['pending_jpg'] = jpg_bytes
                            state.stream_stats[ws_id] = st
                    except Exception:
                        # 인코딩/전송 중 오류는 스트림을 끊지 않고 계속 진행
                        pass
            except WebSocketDisconnect:
                break
            except Exception:
                # 루프 내부 예외는 세션을 유지하면서 무시
                pass
    finally:
        # 연결 종료/정리
        try:
            # 이 세션이 녹화 주체였다면 대기 작업 취소 및 즉시 정리
            try:
                if state._recording_ws_id == ws_id:
                    try:
                        if state._stop_task is not None and not state._stop_task.done():
                            state._stop_task.cancel()
                    except Exception:
                        pass
                    if state.is_recording:
                        try:
                            await stop_and_finalize_recording()
                        except Exception:
                            pass
            except Exception:
                pass
            with state.verification_lock:
                state.verified_users.pop(ws_id, None)
            st = state.stream_stats.get(ws_id, {})
            st['active'] = False
            st['last_updated'] = datetime.now().isoformat(timespec='seconds')
            state.stream_stats[ws_id] = st
            try:
                state.active_websockets.discard(websocket)
            except Exception:
                pass
            # 마지막 스냅샷 저장(간단 메타)
            state.last_stream_snapshot = {
                "ws_id": ws_id,
                "last_updated": st.get('last_updated'),
                "detections": int(st.get('detections',0)),
                "auto_starts": int(st.get('auto_starts',0)),
            }
        except Exception:
            pass
