import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from AI_processor import process_frame
import io
import os
from datetime import datetime
import time
import threading
import boto3
from botocore.exceptions import ClientError
import tempfile
from dotenv import load_dotenv
import json
import aiohttp
import asyncio
from typing import Dict, Any

# Optional: lightweight person detector via AnalyticsEngine
try:
    from analytics import AnalyticsEngine
except Exception:
    AnalyticsEngine = None  # type: ignore

# .env 파일 로드
load_dotenv()

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용으로 모든 도메인 허용, 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 연결된 WebSocket 추적용
active_websockets = set()

# 스트림별 감지/자동녹화 카운터
stream_stats: Dict[int, Dict[str, Any]] = {}
# 마지막 종료된 스트림의 스냅샷(감지 횟수 등)
last_stream_snapshot: Dict[str, Any] | None = None

# 키 검증 관련 설정
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080")  # 백엔드 API URL
AI_API_KEY = os.getenv("AI_API_KEY")  # AI API 키

# 검증된 사용자별 모자이크 해제 상태 관리
verified_users = {}  # {websocket_id: {"is_verified": bool, "decryption_token": str, "camera_id": str}}
verification_lock = threading.Lock()

# 녹화 관련 변수
is_recording = False
video_writer = None
recording_filename = None
recording_lock = threading.Lock()
TEMP_DIR = tempfile.gettempdir()

# 현재 녹화 세션 상태(자동 녹화용)
_recording_started_at_ts: float | None = None
_recording_max_persons: int = 0
_recording_ws_id: int | None = None

# Auto recording settings
AUTO_RECORD_ENABLED = True
AUTO_RECORD_THRESHOLD = 1  # start when >= this many persons detected (user request)
AUTO_ZERO_TIMEOUT_SEC = 3.0  # stop if no person for this duration
AUTO_RECORD_DEBUG = True

# Debug state for auto recording
_auto_debug = {
    "enabled": AUTO_RECORD_ENABLED,
    "threshold": AUTO_RECORD_THRESHOLD,
    "last_check_at": None,
    "last_person_count": None,
    "engine_available": False,
    "hog_available": False,
    "attempted_start": False,
    "started": False,
    "last_error": None,
}

# Last time we saw at least 1 person (epoch seconds)
_last_nonzero_person_ts: float | None = None
# First time we detected zero persons after being non-zero (epoch seconds)
_zero_since_ts: float | None = None

# Shared analytics engine for auto-record person counting
AUTO_ENGINE = None
if AnalyticsEngine is not None:
    try:
        _yolo_cfg = {
            "yolo_model": "yolov8n.pt",
            "device": None,
            "conf": 0.35,
            "iou": 0.45,
            "sample_rate": 1,
        }
        try:
            cfg_path = os.path.join(os.getcwd(), "yolo_config.json")
            if os.path.exists(cfg_path):
                with open(cfg_path, "r", encoding="utf-8") as f:
                    import json as _json
                    user_cfg = _json.load(f)
                    if isinstance(user_cfg, dict):
                        _yolo_cfg.update(user_cfg)
        except Exception as _ce:
            print(f"YOLO config load warning: {_ce}")

        AUTO_ENGINE = AnalyticsEngine(
            yolo_model=_yolo_cfg.get("yolo_model", "yolov8n.pt"),
            device=_yolo_cfg.get("device"),
            conf=float(_yolo_cfg.get("conf", 0.35)),
            iou=float(_yolo_cfg.get("iou", 0.45)),
            sample_rate=int(_yolo_cfg.get("sample_rate", 1))
        )
        if AUTO_RECORD_DEBUG:
            print("[AUTO] AnalyticsEngine initialized for person counting")
    except Exception as _e:
        print(f"Auto-record AnalyticsEngine init failed: {_e}")

# Fallback HOG person detector (if YOLO unavailable)
_HOG = None
try:
    _HOG = cv2.HOGDescriptor()
    _HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    _auto_debug["hog_available"] = True
except Exception as _he:
    _HOG = None
    if AUTO_RECORD_DEBUG:
        print(f"[AUTO] HOG init failed: {_he}")


def _estimate_person_count(frame: np.ndarray) -> int:
    """Estimate number of persons using YOLO if available, else HOG. Safe and fast-ish."""
    count = 0
    try:
        if 'AUTO_ENGINE' in globals() and AUTO_ENGINE is not None and getattr(AUTO_ENGINE, 'yolo', None) is not None:
            persons = AUTO_ENGINE._detect_persons(frame) or []
            count = len(persons) if isinstance(persons, list) else 0
            _auto_debug["engine_available"] = True
            return count
    except Exception as e:
        _auto_debug["last_error"] = f"YOLO detect error: {e}"
    # fallback HOG (downscale for speed)
    try:
        if _HOG is not None:
            h, w = frame.shape[:2]
            scale = 640.0 / max(w, h)
            if scale < 1.0:
                small = cv2.resize(frame, (int(w*scale), int(h*scale)))
            else:
                small = frame
            rects, _ = _HOG.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
            count = len(rects) if rects is not None else 0
            return count
    except Exception as e:
        _auto_debug["last_error"] = f"HOG detect error: {e}"
    return 0

# S3 설정 (환경변수로 설정 권장)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# S3 사용 가능 여부 확인
USE_S3 = all([S3_BUCKET_NAME, S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY])

# S3 클라이언트 초기화
s3_client = None
if USE_S3:
    try:
        s3_client = boto3.client(
            's3',
            region_name=S3_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        print("S3 클라이언트 초기화 성공")
    except Exception as e:
        print(f"S3 클라이언트 초기화 실패: {e}")
        USE_S3 = False
else:
    print("S3 환경변수가 설정되지 않음")

# S3 업로드 함수
def upload_to_s3(file_path, filename):
    if not USE_S3 or not s3_client:
        return False, "S3 not configured"
    
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, f"recordings/{filename}")
        return True, f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/recordings/{filename}"
    except Exception as e:
        return False, str(e)


async def _auto_start_recording_from_frame(frame: np.ndarray, init_person_count: int, ws_id: int) -> None:
    """Start recording using the current frame size if auto-record is enabled and not already recording."""
    global is_recording, video_writer, recording_filename, _recording_started_at_ts, _recording_max_persons, _recording_ws_id
    if not AUTO_RECORD_ENABLED:
        return
    h, w = frame.shape[:2]
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    with recording_lock:
        if is_recording:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"auto_recording_{timestamp}.mp4"
        filepath = os.path.join(TEMP_DIR, recording_filename)
        vw = cv2.VideoWriter(filepath, fourcc, fps, (w, h))
        if vw.isOpened():
            video_writer = vw
            is_recording = True
            _recording_started_at_ts = time.time()
            _recording_max_persons = max(0, int(init_person_count))
            _recording_ws_id = ws_id
            print(f"Auto recording started: {recording_filename} ({w}x{h}@{fps}), storage={'S3' if USE_S3 else 'local'}")
        else:
            # try fallback codec
            try:
                fallback_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                vw2 = cv2.VideoWriter(filepath, fallback_fourcc, fps, (w, h))
                if vw2.isOpened():
                    video_writer = vw2
                    is_recording = True
                    _recording_started_at_ts = time.time()
                    _recording_max_persons = max(0, int(init_person_count))
                    _recording_ws_id = ws_id
                    print(f"Auto recording started with MJPG: {recording_filename} ({w}x{h}@{fps})")
                else:
                    print("Failed to start auto recording (VideoWriter not opened)")
            except Exception as e:
                print(f"Auto recording writer error: {e}")


async def _stop_and_finalize_recording():
    """Stop current recording and upload/move if needed. Returns dict result."""
    global is_recording, video_writer, recording_filename
    with recording_lock:
        if not is_recording:
            return {"error": "Not recording"}
        is_recording = False
        if video_writer is not None:
            try:
                video_writer.release()
            except Exception:
                pass
            video_writer = None

        if not recording_filename:
            return {"status": "Recording stopped", "filename": None}

        temp_filepath = os.path.join(TEMP_DIR, recording_filename)
        if not os.path.exists(temp_filepath):
            print(f"Recording stop requested but file missing: {temp_filepath}")
            return {"error": "Recording file not found"}

        # If S3 configured, upload then delete local
        if USE_S3 and s3_client:
            success, result = upload_to_s3(temp_filepath, recording_filename)
            try:
                os.remove(temp_filepath)
            except Exception:
                pass
            if success:
                print(f"Recording finalized - uploaded to S3: {recording_filename} -> {result}")
                return {
                    "status": "Recording stopped and uploaded to S3",
                    "filename": recording_filename,
                    "s3_url": result,
                    "storage": "S3",
                    "error": "no error",
                }
            else:
                print(f"Recording finalized - S3 upload failed: {recording_filename}, error: {result}")
                return {
                    "status": "Recording stopped but S3 upload failed",
                    "filename": recording_filename,
                    "error": result,
                    "storage": "S3 (실패)",
                }
        # Else keep local file
        print(f"Recording finalized - kept locally: {temp_filepath}")
        return {"status": "Recording stopped", "filename": recording_filename, "storage": "local"}

# 키 검증 함수
async def verify_key_with_backend(access_token: str, camera_id: str):
    """백엔드 API로 키 검증 요청"""
    if not AI_API_KEY:
        return {"success": False, "message": "AI API 키가 설정되지 않았습니다."}
    
    url = f"{BACKEND_API_URL}/api/decryption/keys/verify/ai"
    headers = {
        "AiApiKey": AI_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "accessToken": access_token,
        "cameraId": camera_id
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"success": True, "data": result}
                else:
                    error_text = await response.text()
                    return {"success": False, "message": f"키 검증 실패: {error_text}"}
    except Exception as e:
        return {"success": False, "message": f"백엔드 연결 오류: {str(e)}"}

# 통합 비디오 WebSocket 엔드포인트
@app.websocket("/ws/video")
async def unified_video_ws(websocket: WebSocket):
    global is_recording, video_writer, recording_filename, _last_nonzero_person_ts, _zero_since_ts, last_stream_snapshot
    global _recording_started_at_ts, _recording_max_persons, _recording_ws_id
    await websocket.accept()
    websocket_id = id(websocket)
    active_websockets.add(websocket)
    # 통계 초기화
    stream_stats[websocket_id] = {
        "detections": 0,            # 0 -> >=1 전환 횟수
        "auto_starts": 0,           # 자동 녹화 시작 횟수
        "prev_person_count": 0,
        "active": True,
        "last_updated": datetime.now().isoformat(timespec='seconds')
    }
    
    # 초기 상태: 모자이크 적용 (미검증 상태)
    with verification_lock:
        verified_users[websocket_id] = {
            "is_verified": False,
            "decryption_token": None,
            "camera_id": None
        }
    
    try:
        while True:
            try:
                # 메시지 수신 (JSON 또는 바이너리)
                message = await websocket.receive()
                
                if "text" in message:
                    # JSON 메시지 처리 (키 검증 요청)
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") == "key_verification":
                            access_token = data.get("accessToken")
                            camera_id = data.get("cameraId")
                            
                            if not access_token or not camera_id:
                                await websocket.send_text(json.dumps({
                                    "type": "verification_result",
                                    "success": False,
                                    "message": "accessToken과 cameraId는 필수입니다."
                                }))
                                continue
                            
                            # 백엔드로 키 검증 요청
                            verification_result = await verify_key_with_backend(access_token, camera_id)
                            
                            if verification_result["success"]:
                                # 백엔드 응답 구조: ApiResponse<KeyVerificationResponseDto>
                                backend_response = verification_result["data"]
                                
                                # ApiResponse에서 data 필드 추출
                                if backend_response.get("isSuccess") and "data" in backend_response:
                                    response_data = backend_response["data"]
                                    
                                    # KeyVerificationResponseDto 필드 확인
                                    is_valid = response_data.get("valid", False)  # 'isValid' 대신 'valid' 사용
                                    can_decrypt = response_data.get("canDecrypt", False)
                                    
                                    if is_valid and can_decrypt:
                                        with verification_lock:
                                            verified_users[websocket_id] = {
                                                "is_verified": True,
                                                "decryption_token": response_data.get("decryptionToken"),
                                                "camera_id": camera_id
                                            }
                                        
                                        await websocket.send_text(json.dumps({
                                            "type": "verification_result",
                                            "success": True,
                                            "message": "키 검증 성공 - 모자이크가 해제됩니다.",
                                            "canDecrypt": True,
                                            "isValid": True,
                                            "expiresAt": response_data.get("expiresAt"),
                                            "remainingUses": response_data.get("remainingUses"),
                                            "decryptionToken": response_data.get("decryptionToken"),
                                            "verifiedAt": response_data.get("verifiedAt"),
                                            "blockchainVerified": response_data.get("blockchainVerified")
                                        }))
                                    else:
                                        await websocket.send_text(json.dumps({
                                            "type": "verification_result",
                                            "success": False,
                                            "message": response_data.get("message", f"키 검증 실패 - 권한이 없습니다."),
                                            "canDecrypt": False,
                                            "isValid": is_valid
                                        }))
                                else:
                                    # API 응답이 실패인 경우
                                    error_message = backend_response.get("message", "API 응답 오류")
                                    await websocket.send_text(json.dumps({
                                        "type": "verification_result",
                                        "success": False,
                                        "message": f"백엔드 API 오류: {error_message}",
                                        "canDecrypt": False
                                    }))
                            else:
                                await websocket.send_text(json.dumps({
                                    "type": "verification_result",
                                    "success": False,
                                    "message": verification_result["message"],
                                    "canDecrypt": False
                                }))
                        
                        elif data.get("type") == "disconnect":
                            # 개별 사용자 연결 해제 요청
                            # 종료 직전 최종 통계 포함 반환
                            st = stream_stats.get(websocket_id, {})
                            await websocket.send_text(json.dumps({
                                "type": "disconnect_result",
                                "success": True,
                                "message": "Connection will be closed.",
                                "detections": int(st.get("detections", 0)),
                                "auto_starts": int(st.get("auto_starts", 0))
                            }))
                            # WebSocket 연결 종료
                            await websocket.close()
                            break
                    
                    except json.JSONDecodeError:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "잘못된 JSON 형식입니다."
                        }))
                
                elif "bytes" in message:
                    # 비디오 프레임 처리
                    data = message["bytes"]
                    nparr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        if AUTO_RECORD_DEBUG:
                            print("[AUTO] Received bytes but failed to decode frame")
                        continue

                    # 자동 녹화: 인원 수 기준으로 시작
                    try:
                        # Estimate current person count
                        pcnt = _estimate_person_count(frame)
                        if AUTO_RECORD_DEBUG:
                            _auto_debug.update({
                                "enabled": AUTO_RECORD_ENABLED,
                                "threshold": AUTO_RECORD_THRESHOLD,
                                "last_check_at": datetime.now().isoformat(timespec='seconds'),
                                "last_person_count": pcnt,
                                "attempted_start": False,
                                "started": is_recording,
                            })
                        now_ts = time.time()
                        # 감지 Episode: 0 -> >=1 전환 시 카운트 증가
                        try:
                            st = stream_stats.get(websocket_id)
                            if st is not None:
                                prev = int(st.get("prev_person_count", 0))
                                if prev == 0 and pcnt >= 1:
                                    st["detections"] = int(st.get("detections", 0)) + 1
                                st["prev_person_count"] = pcnt
                                st["last_updated"] = datetime.now().isoformat(timespec='seconds')
                        except Exception:
                            pass
                        # auto-start when persons >= threshold
                        if pcnt >= AUTO_RECORD_THRESHOLD and not is_recording and AUTO_RECORD_ENABLED:
                            _auto_debug["attempted_start"] = True
                            await _auto_start_recording_from_frame(frame, pcnt, websocket_id)
                            _auto_debug["started"] = is_recording
                            # 자동 녹화 시작 카운트 증가
                            try:
                                st = stream_stats.get(websocket_id)
                                if st is not None:
                                    st["auto_starts"] = int(st.get("auto_starts", 0)) + 1
                                    st["last_updated"] = datetime.now().isoformat(timespec='seconds')
                            except Exception:
                                pass
                            # 시작 알림(WebSocket 푸시)
                            if is_recording:
                                try:
                                    started_iso = None
                                    if _recording_started_at_ts is not None:
                                        started_iso = datetime.fromtimestamp(_recording_started_at_ts).isoformat(timespec='seconds')
                                    payload_start = {
                                        "type": "auto_recording_started",
                                        "filename": recording_filename,
                                        "started_at": started_iso,
                                        "initial_persons": int(pcnt),
                                        "storage": "S3" if USE_S3 else "local",
                                    }
                                    await websocket.send_text(json.dumps(payload_start))
                                except Exception as _se2:
                                    print(f"WS start notify error: {_se2}")
                        # 녹화 중 최대 인원수 갱신
                        if is_recording:
                            try:
                                global _recording_max_persons
                                _recording_max_persons = max(_recording_max_persons, int(pcnt))
                            except Exception:
                                pass
                        # update zero/nonzero timers with hysteresis
                        if pcnt >= 1:
                            _last_nonzero_person_ts = now_ts
                            _zero_since_ts = None
                            _auto_debug.pop("zero_gap_sec", None)
                        else:
                            if _zero_since_ts is None:
                                _zero_since_ts = now_ts
                            zero_gap = now_ts - _zero_since_ts
                            _auto_debug["zero_gap_sec"] = round(zero_gap, 2)
                            # auto-stop when zero sustained beyond timeout
                            if is_recording and zero_gap >= AUTO_ZERO_TIMEOUT_SEC:
                                res = await _stop_and_finalize_recording()
                                if AUTO_RECORD_DEBUG:
                                    print(f"Auto stop after {zero_gap:.1f}s with zero persons -> {res}")
                                # 자동 녹화 종료 알림(WebSocket 푸시)
                                try:
                                    duration = None
                                    if _recording_started_at_ts is not None:
                                        duration = round(time.time() - _recording_started_at_ts, 2)
                                    payload = {
                                        "type": "auto_recording_finalized",
                                        "filename": recording_filename,
                                        "segment_max_persons": int(_recording_max_persons),
                                        "duration_sec": duration,
                                        "storage": res.get("storage"),
                                    }
                                    if res.get("s3_url"):
                                        payload["s3_url"] = res["s3_url"]
                                    await websocket.send_text(json.dumps(payload))
                                except Exception as _se:
                                    print(f"WS notify error: {_se}")
                                finally:
                                    # 세션 상태 리셋
                                    _recording_max_persons = 0
                                    _recording_ws_id = None
                                    _recording_started_at_ts = None
                    except Exception as e:
                        # Do not break stream on detection errors
                        print(f"Auto record check error: {e}")
                        _auto_debug["last_error"] = f"check error: {e}"
                    
                    # 사용자 검증 상태에 따라 모자이크 적용 여부 결정
                    with verification_lock:
                        user_status = verified_users.get(websocket_id, {"is_verified": False})
                    
                    if user_status["is_verified"]:
                        # 검증된 사용자: 원본 영상
                        processed_frame = frame
                    else:
                        # 미검증 사용자: 모자이크 적용
                        processed_frame = process_frame(frame, mode="face_plate")
                    
                    _, jpg = cv2.imencode('.jpg', processed_frame)
                    
                    # 녹화 중이면 프레임 저장 (원본 또는 처리된 프레임)
                    with recording_lock:
                        if is_recording and video_writer is not None:
                            video_writer.write(processed_frame)
                    
                    await websocket.send_bytes(jpg.tobytes())
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket message processing error: {e}")
                break
                
    except Exception as e:
        print("WebSocket connection error:", e)
    finally:
        active_websockets.discard(websocket)
        # 사용자 상태 정리
        with verification_lock:
            if websocket_id in verified_users:
                del verified_users[websocket_id]
        # 스트림 상태 비활성 처리
        try:
            if websocket_id in stream_stats:
                stream_stats[websocket_id]["active"] = False
                stream_stats[websocket_id]["last_updated"] = datetime.now().isoformat(timespec='seconds')
                # 종료 스냅샷 업데이트
                st = stream_stats.get(websocket_id, {})
                last_stream_snapshot = {
                    "stream_id": str(websocket_id),
                    "detections": int(st.get("detections", 0)),
                    "auto_starts": int(st.get("auto_starts", 0)),
                    "ended_at": datetime.now().isoformat(timespec='seconds')
                }
        except Exception:
            pass

 

# 마지막 종료된 스트림의 감지 횟수 조회
@app.get("/last_detections")
async def get_last_detections():
    if last_stream_snapshot is None:
        return {"available": False, "message": "아직 종료된 스트림이 없습니다."}
    return {
        "available": True,
        "stream_id": last_stream_snapshot.get("stream_id"),
        "detections": last_stream_snapshot.get("detections", 0),
        "auto_starts": last_stream_snapshot.get("auto_starts", 0),
        "ended_at": last_stream_snapshot.get("ended_at")
    }

# Auto-recording debug endpoint
@app.get("/auto_recording/debug")
async def auto_recording_debug():
    state = dict(_auto_debug)
    state.update({
        "is_recording": is_recording,
        "last_nonzero_person_ts": _last_nonzero_person_ts,
    "zero_since_ts": _zero_since_ts,
        "zero_timeout_sec": AUTO_ZERO_TIMEOUT_SEC,
    })
    # compute seconds since last_nonzero if available
    try:
        if _last_nonzero_person_ts is not None:
            state["seconds_since_last_nonzero"] = round(time.time() - _last_nonzero_person_ts, 2)
    except Exception:
        pass
    return state

# ---- Auto-recording settings API ----
@app.get("/auto_recording")
async def get_auto_recording():
    return {
        "enabled": AUTO_RECORD_ENABLED,
        "threshold": AUTO_RECORD_THRESHOLD,
        "is_recording": is_recording,
        "s3_configured": USE_S3
    }


@app.post("/auto_recording")
async def set_auto_recording(payload: Dict[str, Any] = Body(default={})):  # expects {"enabled": bool, "threshold": int}
    global AUTO_RECORD_ENABLED, AUTO_RECORD_THRESHOLD
    if not isinstance(payload, dict):
        return {"error": "Invalid body"}
    if "enabled" in payload:
        try:
            AUTO_RECORD_ENABLED = bool(payload["enabled"])
        except Exception:
            return {"error": "'enabled' must be boolean"}
    if "threshold" in payload:
        try:
            th = int(payload["threshold"])
            if th < 1:
                return {"error": "'threshold' must be >= 1"}
            AUTO_RECORD_THRESHOLD = th
        except Exception:
            return {"error": "'threshold' must be integer"}
    return {
        "enabled": AUTO_RECORD_ENABLED,
        "threshold": AUTO_RECORD_THRESHOLD
    }

# 녹화 시작 엔드포인트
@app.post("/start_recording")
async def start_recording():
    global is_recording, video_writer, recording_filename
    
    if not USE_S3 or not s3_client:
        return {"error": "S3 연결 오류: 환경변수를 확인하세요"}
    
    with recording_lock:
        if is_recording:
            return {"error": "Already recording"}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"recording_{timestamp}.mp4"
        filepath = os.path.join(TEMP_DIR, recording_filename)
        
        # VideoWriter 초기화 (640x480, 30fps 기본값)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filepath, fourcc, 30.0, (640, 480))
        
        if video_writer.isOpened():
            is_recording = True
            return {
                "status": "Recording started (저장: S3)", 
                "filename": recording_filename,
                "storage": "S3",
                "error": "no error"
            }
        else:
            return {"error": "Failed to start recording"}

# 녹화 중단 엔드포인트
@app.post("/stop_recording")
async def stop_recording():
    # Reuse common stop logic (handles S3 present/absent)
    return await _stop_and_finalize_recording()

# S3에서 녹화 파일 목록 조회
@app.get("/recordings")
async def list_recordings():
    if not USE_S3 or not s3_client:
        return {"error": "S3 연결 오류: 환경변수를 확인하세요"}
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix="recordings/"
        )
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                filename = obj['Key'].replace('recordings/', '')
                if filename.endswith('.mp4') and filename:  # 빈 파일명 제외
                    files.append({
                        "filename": filename,
                        "size": obj['Size'],
                        "last_modified": obj['LastModified'].isoformat(),
                        "url": f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{obj['Key']}",
                        "storage": "S3"
                    })
        
        files.sort(key=lambda x: x['last_modified'], reverse=True)  # 최신 파일 먼저
        return {"recordings": files, "storage": "S3"}
    
    except Exception as e:
        return {"error": f"S3 error: {str(e)}"}

# S3에서 녹화 파일 직접 접근 URL 생성
@app.get("/recordings/{filename}")
async def get_recording_url(filename: str):
    if not USE_S3 or not s3_client:
        return {"error": "S3 연결 오류: 환경변수를 확인하세요"}
    
    try:
        # S3에서 파일 존재 확인
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=f"recordings/{filename}")
        
        # Pre-signed URL 생성 (1시간 유효)
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': f"recordings/{filename}"},
            ExpiresIn=3600  # 1시간
        )
        
        return {"url": url, "filename": filename, "storage": "S3", "error": "no error"}
    
    except Exception as e:
        if hasattr(e, 'response') and e.response['Error']['Code'] == '404':
            return {"error": "File not found"}
        else:
            return {"error": f"S3 error: {str(e)}"}

# 녹화 상태 확인
@app.get("/recording_status")
async def recording_status():
    return {
        "is_recording": is_recording,
        "current_file": recording_filename if is_recording else None,
        "storage_type": "S3",
        "s3_configured": USE_S3
    }

# HTTP 엔드포인트로 모든 WebSocket 연결 해제 (관리자용)
@app.post("/disconnect_ws")
async def disconnect_ws():
    """
    모든 WebSocket 연결 해제 (관리자용)
    """
    print("모든 WebSocket 연결 해제 실행")
    closed = 0
    
    for ws in list(active_websockets):
        try:
            # 종료 전 최종 통계 전송
            ws_id = id(ws)
            st = stream_stats.get(ws_id, {})
            try:
                await ws.send_text(json.dumps({
                    "type": "final_stats",
                    "detections": int(st.get("detections", 0)),
                    "auto_starts": int(st.get("auto_starts", 0)),
                    "message": "Connection will be closed by server."
                }))
            except Exception:
                pass
            await ws.close()
            closed += 1
            print(f"WebSocket 연결 해제 성공 - ID: {id(ws)}")
        except Exception as e:
            print(f"WebSocket close error: {e}")
    
    return {
        "success": True,
        "message": f"모든 WebSocket 연결이 해제되었습니다.",
        "disconnected": closed,
        "total_active_before": len(active_websockets)
    }

# 현재 연결된 사용자 상태 확인
@app.get("/verification_status")
async def get_verification_status():
    with verification_lock:
        verified_count = sum(1 for user in verified_users.values() if user["is_verified"])
        total_count = len(verified_users)
    
    return {
        "total_connections": total_count,
        "verified_connections": verified_count,
        "unverified_connections": total_count - verified_count
    }

# 모든 사용자 모자이크 강제 적용 (긴급상황용)
@app.post("/force_mosaic")
async def force_mosaic_all():
    with verification_lock:
        for websocket_id in verified_users:
            verified_users[websocket_id]["is_verified"] = False
            verified_users[websocket_id]["decryption_token"] = None
    
    # 연결된 모든 WebSocket에 모자이크 강제 적용 알림
    disconnected = 0
    for ws in list(active_websockets):
        try:
            await ws.send_text(json.dumps({
                "type": "force_mosaic",
                "message": "모든 연결에 모자이크가 강제 적용되었습니다."
            }))
        except Exception as e:
            print(f"WebSocket send error: {e}")
            disconnected += 1
    
    return {
        "message": "모든 사용자에게 모자이크가 강제 적용되었습니다.",
        "affected_connections": len(verified_users),
        "disconnected": disconnected
    }
