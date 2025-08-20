import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, Body, File, UploadFile, HTTPException
 


from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# 녹화 프레임 비동기 기록용 큐/워커
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
import shutil
import base64

# Optional: lightweight person detector via AnalyticsEngine
try:
    from analytics import AnalyticsEngine
except Exception:
    AnalyticsEngine = None  # type: ignore

# Optional: YOLOv8 (used by face detector when available)
try:
    from ultralytics import YOLO  # type: ignore
except Exception:
    YOLO = None  # type: ignore

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

# Simple build/revision marker to verify running code
SERVER_REV = "2025-08-16T13:35Z-fightfix"

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

# 최근 수신된 사용자 정보 저장용 (POST로 전달되는 userId)
last_user_info: Dict[str, Any] | None = None

# 녹화 관련 변수
is_recording = False
video_writer = None  # processed (mosaic) stream writer
video_writer_raw = None  # raw/original stream writer
recording_filename = None  # processed filename
recording_filename_raw = None  # raw filename
recording_lock = threading.Lock()
TEMP_DIR = tempfile.gettempdir()

# 현재 녹화 세션 상태(자동 녹화용)
_recording_started_at_ts: float | None = None
_recording_max_persons: int = 0
_recording_ws_id: int | None = None
_manual_start_requested: bool = False
_stop_task = None  # asyncio.Task | None, 자동 종료 시 백그라운드 태스크 핸들

# Auto recording settings
AUTO_RECORD_ENABLED = True
AUTO_RECORD_THRESHOLD = 1  # start when >= this many persons detected (user request)
AUTO_ZERO_TIMEOUT_SEC = 3.0  # stop if no person for this duration
AUTO_RECORD_DEBUG = True
# TEMP: disable auto-recording behavior at runtime without deleting code
AUTO_RECORD_TEMP_DISABLED = False  # auto-recording enabled

# 성능 튜닝용 샘플링(기본값은 보수적으로 설정)
DETECT_EVERY_N = int(os.getenv("DETECT_EVERY_N", "2"))           # 사람 수 추정은 N프레임마다 수행
MOSAIC_EVERY_N = int(os.getenv("MOSAIC_EVERY_N", "1"))            # 모자이크는 N프레임마다 새로 계산(1이면 매 프레임)
MOSAIC_PROCESS_MAX_WIDTH = int(os.getenv("MOSAIC_PROCESS_MAX_WIDTH", "0"))  # 모자이크 내부 처리용 축소 최대 폭(0이면 비활성)

# Debounce settings for presence to prevent frequent start/stop flapping
AUTO_PRESENCE_WINDOW = int(os.getenv("AUTO_PRESENCE_WINDOW", "20"))  # frames to look back
AUTO_PRESENCE_MIN_HITS = int(os.getenv("AUTO_PRESENCE_MIN_HITS", "5"))  # min frames with presence in window to consider active

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

 

# Non-blocking WebSocket send with coalescing of pending frames
async def _ws_send_coalesced(websocket: WebSocket, ws_id: int, first_bytes: bytes):
    try:
        await websocket.send_bytes(first_bytes)
        # send the latest pending if writer got ahead
        while True:
            st = stream_stats.get(ws_id)
            if not st:
                break
            pending = st.pop("pending_jpg", None)
            if not pending:
                break
            await websocket.send_bytes(pending)
    except Exception:
        pass
    finally:
        st = stream_stats.get(ws_id)
        if st is not None:
            st["send_busy"] = False

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
## (fight detection settings API removed)
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

# (Fight detection feature removed)

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

# 시간 기반 얼굴 검출 관련 설정
FACE_DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv('FACE_DETECTION_CONFIDENCE_THRESHOLD', '0.5'))
FACE_SIMILARITY_THRESHOLD = float(os.getenv('FACE_SIMILARITY_THRESHOLD', '0.95'))
PROCESSING_DURATION_SECONDS = int(os.getenv('PROCESSING_DURATION_SECONDS', '10'))

# 얼굴 검출기 초기화
face_detector = None
face_hashes = []  # 중복 제거를 위한 얼굴 해시 저장
face_bboxes = []  # 위치 기반 중복 제거를 위한 바운딩 박스 저장

def initialize_face_detector():
    """얼굴 검출기 초기화"""
    global face_detector
    if face_detector is None:
        try:
            import torch
            # PyTorch 2.6+ 호환성을 위한 설정
            torch.hub.set_dir('.')
            
            # 훈련된 얼굴 검출 모델 로드
            model_path = 'runs/face_detection/yolov8_face/weights/best.pt'
            if os.path.exists(model_path):
                face_detector = YOLO(model_path)
                print("✅ 훈련된 얼굴 검출 모델 로드 성공")
            else:
                # 훈련된 모델이 없으면 기본 모델 사용
                face_detector = YOLO('yolov8n.pt')
                print("✅ YOLOv8n 기본 모델 로드 성공")
                
        except Exception as e:
            print(f"❌ YOLO 모델 로드 실패: {e}")
            print("PyTorch 2.6+ 호환성 문제로 인한 오류입니다.")
            print("weights_only=False로 설정하여 다시 시도합니다...")
            
            try:
                import torch
                # PyTorch 2.6+ 호환성을 위한 환경 변수 설정
                os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
                
                # torch.load의 기본값을 False로 설정
                import torch.serialization
                original_load = torch.load
                
                def safe_load(*args, **kwargs):
                    kwargs['weights_only'] = False
                    return original_load(*args, **kwargs)
                
                torch.load = safe_load
                
                # 훈련된 얼굴 검출 모델 로드
                model_path = 'runs/face_detection/yolov8_face/weights/best.pt'
                if os.path.exists(model_path):
                    face_detector = YOLO(model_path)
                    print("✅ 훈련된 얼굴 검출 모델 로드 성공 (weights_only=False)")
                else:
                    # 훈련된 모델이 없으면 기본 모델 사용
                    face_detector = YOLO('yolov8n.pt')
                    print("✅ YOLOv8n 기본 모델 로드 성공 (weights_only=False)")
                
            except Exception as e2:
                print(f"❌ YOLO 모델 로드 재시도 실패: {e2}")
                raise Exception(f"YOLO 모델 초기화 실패: {e2}")
                    
    return face_detector

def calculate_image_hash(image: np.ndarray) -> str:
    """이미지 해시 계산 (중복 제거용)"""
    try:
        # 이미지를 8x8 크기로 리사이즈
        resized = cv2.resize(image, (8, 8))
        # 그레이스케일로 변환
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        # 평균값 계산
        avg = gray.mean()
        # 해시 생성
        hash_str = ''
        for i in range(8):
            for j in range(8):
                if gray[i, j] > avg:
                    hash_str += '1'
                else:
                    hash_str += '0'
        return hash_str
    except Exception as e:
        print(f"해시 계산 실패: {e}")
        return None

def calculate_hash_similarity(hash1: str, hash2: str) -> float:
    """해시 유사도 계산"""
    if len(hash1) != len(hash2):
        return 0.0
    
    # 해밍 거리 계산
    hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
    # 유사도 = 1 - (해밍 거리 / 해시 길이)
    similarity = 1 - (hamming_distance / len(hash1))
    return similarity

def is_duplicate_face(face_hash: str, bbox: list = None) -> bool:
    """얼굴 중복 검사 (해시 + 위치 기반)"""
    global face_hashes, face_bboxes
    
    if not face_hash or len(face_hashes) == 0:
        return False
    
    # 기존 얼굴들과 유사도 계산
    for existing_hash in face_hashes:
        similarity = calculate_hash_similarity(face_hash, existing_hash)
        if similarity > FACE_SIMILARITY_THRESHOLD:
            return True
    
    # 위치 기반 중복 검사 (같은 위치에 있는 얼굴은 중복으로 간주)
    if bbox:
        x1, y1, x2, y2 = bbox
        face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        face_area = (x2 - x1) * (y2 - y1)
        
        # 기존 얼굴들과 위치 비교
        for existing_face in face_bboxes:
            ex1, ey1, ex2, ey2 = existing_face
            ex_center = ((ex1 + ex2) // 2, (ey1 + ey2) // 2)
            ex_area = (ex2 - ex1) * (ey2 - ey1)
            
            # 중심점 거리 계산
            center_distance = ((face_center[0] - ex_center[0]) ** 2 + 
                             (face_center[1] - ex_center[1]) ** 2) ** 0.5
            
            # 면적 비율 계산
            area_ratio = min(face_area, ex_area) / max(face_area, ex_area)
            
            # 같은 위치에 비슷한 크기의 얼굴이면 중복으로 간주 (더 관대하게)
            if center_distance < 30 and area_ratio > 0.9:
                return True
    
    return False

# ---------------- Face validation helpers to reduce false positives ----------------
try:
    _HAAR_FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:
    _HAAR_FRONTAL = cv2.CascadeClassifier()
try:
    _HAAR_PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
except Exception:
    _HAAR_PROFILE = cv2.CascadeClassifier()

def _verify_face_region(img: np.ndarray) -> bool:
    """Return True if a face is detected in the region using Haar cascades (frontal/profile)."""
    try:
        if img is None or img.size == 0:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        found = False
        if not _HAAR_FRONTAL.empty():
            faces = _HAAR_FRONTAL.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
            if len(faces) > 0:
                found = True
        if not found and not _HAAR_PROFILE.empty():
            prof = _HAAR_PROFILE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
            if len(prof) > 0:
                found = True
            else:
                # Try mirrored for opposite profile
                gray_flip = cv2.flip(gray, 1)
                prof2 = _HAAR_PROFILE.detectMultiScale(gray_flip, scaleFactor=1.1, minNeighbors=3, minSize=(24, 24))
                if len(prof2) > 0:
                    found = True
        return found
    except Exception:
        return False

def _detect_faces_cascade(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect faces on the whole frame using Haar cascades and return boxes as (x1,y1,x2,y2)."""
    boxes: list[tuple[int, int, int, int]] = []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        if not _HAAR_FRONTAL.empty():
            faces = _HAAR_FRONTAL.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32))
            for (x, y, w, h) in faces:
                boxes.append((int(x), int(y), int(x + w), int(y + h)))
        if not _HAAR_PROFILE.empty():
            prof = _HAAR_PROFILE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32))
            for (x, y, w, h) in prof:
                boxes.append((int(x), int(y), int(x + w), int(y + h)))
            # mirrored for other side
            gray_flip = cv2.flip(gray, 1)
            prof2 = _HAAR_PROFILE.detectMultiScale(gray_flip, scaleFactor=1.1, minNeighbors=4, minSize=(32, 32))
            w_img = frame.shape[1]
            for (x, y, w, h) in prof2:
                # map back from flipped
                x1 = w_img - (x + w)
                boxes.append((int(x1), int(y), int(x1 + w), int(y + h)))
    except Exception:
        pass
    return boxes

def _iou(a: tuple[int,int,int,int], b: tuple[int,int,int,int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return (inter / union) if union > 0 else 0.0

def _nms(boxes: list[tuple[int,int,int,int]], iou_thresh: float = 0.4) -> list[tuple[int,int,int,int]]:
    if not boxes:
        return []
    keep: list[tuple[int,int,int,int]] = []
    boxes_sorted = sorted(boxes, key=lambda b: (b[2]-b[0])*(b[3]-b[1]), reverse=True)
    for b in boxes_sorted:
        if all(_iou(b, k) < iou_thresh for k in keep):
            keep.append(b)
    return keep

def upload_face_to_s3(image_path: str, s3_key: str) -> str:
    """얼굴 이미지를 S3에 업로드하고 URL 반환"""
    if not USE_S3 or not s3_client:
        return f"file://{os.path.abspath(image_path)}"
    
    try:
        # S3 업로드
        s3_client.upload_file(
            image_path,
            S3_BUCKET_NAME,
            s3_key,
            ExtraArgs={'ContentType': 'image/jpeg'}
        )
        
        # S3 URL 생성
        s3_url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
        print(f"✅ S3 업로드 성공: {s3_key}")
        return s3_url
        
    except Exception as e:
        print(f"❌ S3 업로드 실패: {e}")
        # S3 업로드 실패 시 로컬 파일 경로 반환
        return f"file://{os.path.abspath(image_path)}"

# S3 업로드 함수
def upload_to_s3(file_path, filename):
    if not USE_S3 or not s3_client:
        return False, "S3 not configured"
    
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, f"recordings/{filename}")
        return True, f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/recordings/{filename}"
    except Exception as e:
        return False, str(e)

# Spring 서버 엔드포인트 (자동 녹화 종료 시 사용자 식별자 및 URL 목록 전달)
SPRING_MAKE_ENTITY_URL = os.getenv("SPRING_MAKE_ENTITY_URL", "http://localhost:8080/api/videos/make-entity")

async def _notify_spring_make_entity(user_id: str, urls: list | None = None) -> None:
    if not user_id:
        return
    payload = {"userId": user_id, "urls": urls or []}
    try:
        _prev = payload["urls"][:2]
        _more = " ..." if len(payload["urls"]) > 2 else ""
        print(f"[SPRING] POST {SPRING_MAKE_ENTITY_URL} body={{userId:{user_id}, urls:{_prev}{_more}}}")
    except Exception:
        pass
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                SPRING_MAKE_ENTITY_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                if 200 <= resp.status < 300:
                    print(f"[SPRING] make-entity OK for userId={user_id}, urls={len(payload['urls'])}")
                else:
                    txt = await resp.text()
                    print(f"[SPRING] make-entity failed {resp.status}: {txt}")
    except Exception as e:
        print(f"[SPRING] make-entity error: {e}")


async def _auto_start_recording_from_frame(frame: np.ndarray, init_person_count: int, ws_id: int) -> None:
    """Start recording using the current frame size if auto-record is enabled and not already recording."""
    global is_recording, video_writer, video_writer_raw, recording_filename, recording_filename_raw, _recording_started_at_ts, _recording_max_persons, _recording_ws_id
    if not AUTO_RECORD_ENABLED:
        return
    h, w = frame.shape[:2]
    fps = 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def _open_writers() -> tuple[Any | None, Any | None, str, str]:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fn_proc = f"auto_recording_{ts}.mp4"
        fn_raw = f"auto_recording_{ts}_raw.mp4"
        path_proc = os.path.join(TEMP_DIR, fn_proc)
        path_raw = os.path.join(TEMP_DIR, fn_raw)
        vw_p = cv2.VideoWriter(path_proc, fourcc, fps, (w, h))
        if not vw_p.isOpened():
            try:
                fallback_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                vw_p = cv2.VideoWriter(path_proc, fallback_fourcc, fps, (w, h))
            except Exception:
                pass
        vw_r = cv2.VideoWriter(path_raw, fourcc, fps, (w, h))
        if not vw_r.isOpened():
            try:
                fallback_fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                vw_r = cv2.VideoWriter(path_raw, fallback_fourcc, fps, (w, h))
            except Exception:
                pass
        return vw_p if vw_p is not None else None, vw_r if vw_r is not None else None, fn_proc, fn_raw

    # 파일/코덱 초기화는 스레드에서 처리하여 이벤트 루프 블로킹 최소화
    vw_proc, vw_raw, fn_proc, fn_raw = await asyncio.to_thread(_open_writers)
    with recording_lock:
        if is_recording:
            # 경합 시 방금 연 것들 정리
            try:
                if vw_proc is not None:
                    vw_proc.release()
            except Exception:
                pass
            try:
                if vw_raw is not None:
                    vw_raw.release()
            except Exception:
                pass
            return
        # commit writers if at least one opened
        opened_any = False
        if vw_proc is not None and vw_proc.isOpened():
            video_writer = vw_proc
            recording_filename = fn_proc
            opened_any = True
        else:
            video_writer = None
            recording_filename = fn_proc
        if vw_raw is not None and vw_raw.isOpened():
            video_writer_raw = vw_raw
            recording_filename_raw = fn_raw
            opened_any = True
        else:
            video_writer_raw = None
            recording_filename_raw = fn_raw

    if opened_any:
        is_recording = True
        _recording_started_at_ts = time.time()
        _recording_max_persons = max(0, int(init_person_count))
        _recording_ws_id = ws_id
        print(f"Auto recording started: processed={recording_filename if video_writer else 'DISABLED'}, raw={recording_filename_raw if video_writer_raw else 'DISABLED'} ({w}x{h}@{fps}), storage={'S3' if USE_S3 else 'local'}")
    else:
        print("Failed to start auto recording (no VideoWriter opened)")


async def _stop_and_finalize_recording() -> Dict[str, Any]:
    """Stop current recording and upload/move if needed. Returns dict result.

    내부적으로 writer release는 즉시 수행하고, 업로드/파일 작업은 가능한 범위에서 스레드로 오프로딩합니다.
    """
    global is_recording, video_writer, video_writer_raw, recording_filename, recording_filename_raw

    # 1) writers release + 파일 목록 수집 (락 짧게 보유)
    with recording_lock:
        if not is_recording:
            return {"error": "Not recording"}
        # Change state and release writers while holding the lock
        is_recording = False
        try:
            if video_writer is not None:
                video_writer.release()
        except Exception:
            pass
        finally:
            video_writer = None
        try:
            if video_writer_raw is not None:
                video_writer_raw.release()
        except Exception:
            pass
        finally:
            video_writer_raw = None

        if not recording_filename and not recording_filename_raw:
            return {"status": "Recording stopped", "filenames": [], "storage": "local"}

        file_items: list[tuple[str, str]] = []
        if recording_filename:
            path_proc = os.path.join(TEMP_DIR, recording_filename)
            if os.path.exists(path_proc):
                file_items.append((recording_filename, path_proc))
            else:
                print(f"Recording stop requested but file missing: {path_proc}")
        if recording_filename_raw:
            path_raw = os.path.join(TEMP_DIR, recording_filename_raw)
            if os.path.exists(path_raw):
                file_items.append((recording_filename_raw, path_raw))
            else:
                print(f"Recording stop requested but raw file missing: {path_raw}")

    if not file_items:
        return {"error": "Recording files not found"}

    # 2) 업로드/정리(락 바깥): 오프로딩 병렬 처리
    if USE_S3 and s3_client:
        async def _upload_one(fname: str, fpath: str) -> tuple[str, bool, str]:
            success, result = await asyncio.to_thread(upload_to_s3, fpath, fname)
            try:
                os.remove(fpath)
            except Exception:
                pass
            return fname, success, result

        results = await asyncio.gather(*[_upload_one(fn, fp) for fn, fp in file_items], return_exceptions=False)
        urls: list[str] = []
        errors: list[Dict[str, Any]] = []
        for fname, success, result in results:
            if success:
                urls.append(result)
                print(f"Recording finalized - uploaded to S3: {fname} -> {result}")
            else:
                errors.append({"filename": fname, "error": result})
                print(f"Recording finalized - S3 upload failed: {fname}, error: {result}")
        status_msg = "Recording stopped and uploaded to S3" if urls and not errors else (
            "Recording stopped (partial upload to S3)" if urls and errors else "Recording stopped but S3 upload failed"
        )
        return {
            "status": status_msg,
            "filenames": [fi for fi, _ in file_items],
            "s3_urls": urls,
            "storage": "S3",
            "error": None if not errors else errors,
        }
    else:
        local_paths = [os.path.join(TEMP_DIR, fi) for fi, _ in file_items]
        print(f"Recording finalized - kept locally: {local_paths}")
        return {"status": "Recording stopped", "filenames": [fi for fi, _ in file_items], "local_paths": local_paths, "storage": "local"}

async def _finalize_and_notify(websocket: WebSocket, started_ts: float | None, max_persons: int, rec_fn: str | None, rec_fn_raw: str | None) -> None:
    """자동 종료를 백그라운드로 처리하고 WebSocket 알림 및 Spring 알림을 수행."""
    global _recording_max_persons, _recording_ws_id, _recording_started_at_ts
    res = await _stop_and_finalize_recording()
    # WebSocket에 최종 알림 구성(처리본만 노출)
    try:
        duration = None
        if started_ts is not None:
            duration = round(time.time() - started_ts, 2)
        # processed 파일 존재 여부 확인
        proc_present = False
        try:
            if isinstance(res.get("filenames"), list) and rec_fn:
                proc_present = any(fi == rec_fn for fi in res["filenames"])
        except Exception:
            proc_present = False
        payload = {
            "type": "auto_recording_finalized",
            "filename": (rec_fn if proc_present else None),
            "segment_max_persons": int(max_persons),
            "duration_sec": duration,
            "storage": res.get("storage"),
        }
        if res.get("s3_urls") and rec_fn:
            try:
                proc_url = next((u for u in res["s3_urls"] if isinstance(u, str) and u.endswith(f"/{rec_fn}")), None)
            except Exception:
                proc_url = None
            if proc_url:
                payload["s3_url"] = proc_url
        if res.get("local_paths") and rec_fn:
            try:
                proc_local = next((p for p in res["local_paths"] if isinstance(p, str) and p.endswith(f"/{rec_fn}")), None)
            except Exception:
                proc_local = None
            if proc_local:
                payload["local_paths"] = [proc_local]
            else:
                payload["local_paths"] = []
        await websocket.send_text(json.dumps(payload))
    except Exception as _se:
        print(f"WS notify error: {_se}")

    # Spring 서버 알림(가능 시)
    try:
        if last_user_info and isinstance(last_user_info, dict):
            uid = last_user_info.get("userId")
            if isinstance(uid, str) and uid:
                urls_for_spring: list[str] = []
                try:
                    if isinstance(res.get("s3_urls"), list):
                        if rec_fn:
                            proc_url = next((u for u in res["s3_urls"] if isinstance(u, str) and u.endswith(f"/{rec_fn}")), None)
                            if proc_url:
                                urls_for_spring.append(proc_url)
                        if rec_fn_raw:
                            raw_url = next((u for u in res["s3_urls"] if isinstance(u, str) and u.endswith(f"/{rec_fn_raw}")), None)
                            if raw_url:
                                urls_for_spring.append(raw_url)
                except Exception:
                    pass
                try:
                    if rec_fn and isinstance(res.get("local_paths"), list):
                        need_proc = not any(isinstance(u, str) and u.endswith(f"/{rec_fn}") for u in urls_for_spring)
                        if need_proc:
                            proc_local = next((p for p in res["local_paths"] if isinstance(p, str) and p.endswith(f"/{rec_fn}")), None)
                            if proc_local:
                                urls_for_spring.append(proc_local)
                    if rec_fn_raw and isinstance(res.get("local_paths"), list):
                        need_raw = not any(isinstance(u, str) and u.endswith(f"/{rec_fn_raw}") for u in urls_for_spring)
                        if need_raw:
                            raw_local = next((p for p in res["local_paths"] if isinstance(p, str) and p.endswith(f"/{rec_fn_raw}")), None)
                            if raw_local:
                                urls_for_spring.append(raw_local)
                except Exception:
                    pass
                if urls_for_spring:
                    _prev = urls_for_spring[:2]
                    _more = " ..." if len(urls_for_spring) > 2 else ""
                    print(f"[SPRING] notify enqueue userId={uid}, urls={len(urls_for_spring)}: {_prev}{_more}")
                    asyncio.create_task(_notify_spring_make_entity(uid, urls_for_spring))
    except Exception as _spr_err:
        print(f"[SPRING] notify error: {_spr_err}")

    # 세션 상태 리셋
    _recording_max_persons = 0
    _recording_ws_id = None
    _recording_started_at_ts = None

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
    global _recording_started_at_ts, _recording_max_persons, _recording_ws_id, _manual_start_requested, _stop_task
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

                    # 자동 녹화: 인원 수 기준으로 시작 (진폭 완화용 히스토리 적용)
                    try:
                        # Estimate current person count
                        st = stream_stats.get(websocket_id)
                        if st is None:
                            st = {}
                            stream_stats[websocket_id] = st
                        idx = int(st.get("frame_idx", 0)) + 1
                        st["frame_idx"] = idx

                        if DETECT_EVERY_N <= 1 or (idx % max(1, DETECT_EVERY_N) == 0):
                            # CPU 작업은 스레드로 오프로딩하여 이벤트 루프 블로킹 최소화
                            pcnt = await asyncio.to_thread(_estimate_person_count, frame)
                            st["last_pcnt"] = pcnt
                        else:
                            pcnt = int(st.get("last_pcnt", 0))
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
                                # Presence history for debounce
                                from collections import deque as _dq
                                hist = st.get("presence_hist")
                                if not isinstance(hist, _dq):
                                    hist = _dq(maxlen=max(1, int(AUTO_PRESENCE_WINDOW)))
                                    st["presence_hist"] = hist
                                hist.append(1 if pcnt >= 1 else 0)
                                hits = sum(hist)
                                window_len = len(hist)
                                st["presence_hits"] = hits
                                st["presence_window"] = window_len
                                if AUTO_RECORD_DEBUG:
                                    _auto_debug["presence_hits"] = hits
                                    _auto_debug["presence_window"] = window_len
                        except Exception:
                            pass
                        # auto-start when persons >= threshold
                        # Auto-start disabled guard
                        should_start = False
                        try:
                            st = stream_stats.get(websocket_id) or {}
                            hits = int(st.get("presence_hits", 0))
                            window_len = int(st.get("presence_window", 0))
                            # Require both current pcnt >= threshold and sufficient recent presence
                            if (
                                not AUTO_RECORD_TEMP_DISABLED
                                and not is_recording
                                and AUTO_RECORD_ENABLED
                                and pcnt >= AUTO_RECORD_THRESHOLD
                                and hits >= max(1, int(AUTO_PRESENCE_MIN_HITS))
                            ):
                                should_start = True
                        except Exception:
                            # fallback to previous behavior on any error
                            should_start = (
                                not AUTO_RECORD_TEMP_DISABLED and pcnt >= AUTO_RECORD_THRESHOLD and not is_recording and AUTO_RECORD_ENABLED
                            )
                        if should_start:
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
                                        # Only set filename if processed writer is active
                                        "filename": (recording_filename if (video_writer is not None and recording_filename) else None),
                                        "started_at": started_iso,
                                        "initial_persons": int(pcnt),
                                        "storage": "S3" if USE_S3 else "local",
                                    }
                                    await websocket.send_text(json.dumps(payload_start))
                                except Exception as _se2:
                                    print(f"WS start notify error: {_se2}")
                        # 수동 녹화 요청 처리 (다음 프레임 기준)
                        if _manual_start_requested and (not is_recording):
                            try:
                                await _auto_start_recording_from_frame(frame, pcnt, websocket_id)
                                # 시작 알림(WebSocket 푸시)
                                if is_recording:
                                    started_iso = None
                                    if _recording_started_at_ts is not None:
                                        started_iso = datetime.fromtimestamp(_recording_started_at_ts).isoformat(timespec='seconds')
                                    payload_start = {
                                        "type": "manual_recording_started",
                                        # Only set filename if processed writer is active
                                        "filename": (recording_filename if (video_writer is not None and recording_filename) else None),
                                        "started_at": started_iso,
                                        "initial_persons": int(pcnt),
                                        "storage": "S3" if USE_S3 else "local",
                                    }
                                    await websocket.send_text(json.dumps(payload_start))
                            except Exception as _me:
                                print(f"Manual start error: {_me}")
                            finally:
                                _manual_start_requested = False
                        # 녹화 중 최대 인원수 갱신
                        if is_recording:
                            try:
                                global _recording_max_persons
                                _recording_max_persons = max(_recording_max_persons, int(pcnt))
                            except Exception:
                                pass
                        # update zero/nonzero timers with debounce using presence history
                        try:
                            st = stream_stats.get(websocket_id) or {}
                            hits = int(st.get("presence_hits", 0))
                            window_len = int(st.get("presence_window", 0)) or 1
                        except Exception:
                            hits, window_len = (1 if pcnt >= 1 else 0), 1

                        if hits > 0:
                            _last_nonzero_person_ts = now_ts
                            _zero_since_ts = None
                            _auto_debug.pop("zero_gap_sec", None)
                        else:
                            if _zero_since_ts is None:
                                _zero_since_ts = now_ts
                            zero_gap = now_ts - _zero_since_ts
                            _auto_debug["zero_gap_sec"] = round(zero_gap, 2)
                            # auto-stop when sustained zero presence beyond timeout
                            if (
                                not AUTO_RECORD_TEMP_DISABLED
                                and is_recording
                                and zero_gap >= AUTO_ZERO_TIMEOUT_SEC
                            ):
                                # 프레임 루프를 막지 않도록 백그라운드로 종료/업로드/알림 처리
                                if _stop_task is None or _stop_task.done():
                                    start_ts_snapshot = _recording_started_at_ts
                                    max_persons_snapshot = _recording_max_persons
                                    rec_fn_snapshot = recording_filename
                                    rec_fn_raw_snapshot = recording_filename_raw
                                    _stop_task = asyncio.create_task(
                                        _finalize_and_notify(
                                            websocket,
                                            start_ts_snapshot,
                                            max_persons_snapshot,
                                            rec_fn_snapshot,
                                            rec_fn_raw_snapshot,
                                        )
                                    )
                    except Exception as e:
                        # Do not break stream on detection errors
                        print(f"Auto record check error: {e}")
                        _auto_debug["last_error"] = f"check error: {e}"
                    
                    # (fight detection removed)

                    # 사용자 검증 상태에 따라 모자이크 적용 여부 결정
                    # 원본 프레임 보존 (raw)
                    raw_frame = frame.copy()
                    with verification_lock:
                        user_status = verified_users.get(websocket_id, {"is_verified": False})
                    
                    if user_status["is_verified"]:
                        # 검증된 사용자: 원본 영상 그대로 송출
                        processed_frame = raw_frame
                    else:
                        # 미검증 사용자: 모자이크 적용 (원본 사본에 적용)
                        st = stream_stats.get(websocket_id) or {}
                        idx = int(st.get("frame_idx", 0))
                        cached = st.get("last_mosaic")
                        need_new = True
                        if MOSAIC_EVERY_N > 1 and idx % MOSAIC_EVERY_N != 0 and cached is not None:
                            need_new = False
                        if need_new:
                            try:
                                h0, w0 = raw_frame.shape[:2]
                                # 내부 처리 축소 후 원본 크기로 복원(프라이버시 보장 유지)
                                if MOSAIC_PROCESS_MAX_WIDTH and MOSAIC_PROCESS_MAX_WIDTH > 0 and w0 > MOSAIC_PROCESS_MAX_WIDTH:
                                    scale = MOSAIC_PROCESS_MAX_WIDTH / float(w0)
                                    small = cv2.resize(raw_frame, (int(w0 * scale), int(h0 * scale)))
                                    mosaic_small = await asyncio.to_thread(process_frame, small.copy(), mode="face_plate")
                                    processed_frame = cv2.resize(mosaic_small, (w0, h0))
                                else:
                                    processed_frame = await asyncio.to_thread(process_frame, raw_frame.copy(), mode="face_plate")
                                st["last_mosaic"] = processed_frame
                            except Exception:
                                processed_frame = raw_frame  # 실패 시 원본 유지(전송 전 단계에서 품질 저하로 완화)
                        else:
                            processed_frame = cached

                    # 전송 최적화: 리사이즈 + JPEG 품질 조정
                    display_frame = processed_frame
                    try:
                        max_w = int(os.getenv("STREAM_MAX_WIDTH", "960"))
                        if max_w and max_w > 0:
                            h0, w0 = display_frame.shape[:2]
                            if w0 > max_w:
                                sc = max_w / float(w0)
                                display_frame = cv2.resize(display_frame, (int(w0 * sc), int(h0 * sc)))
                    except Exception:
                        display_frame = processed_frame
                    q = int(os.getenv("STREAM_JPEG_QUALITY", "70"))
                    q = max(1, min(100, q))
                    # JPEG 인코딩도 CPU 작업이므로 오프로딩
                    def _encode_jpg(img, quality):
                        ret, arr = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
                        return arr.tobytes() if ret else b''
                    jpg_bytes = await asyncio.to_thread(_encode_jpg, display_frame, q)
                    
                    # 녹화 중이면 프레임 저장 (모자이크 및 원본 동시 저장)
                    with recording_lock:
                        if is_recording:
                            # 먼저 raw/original 프레임 기록 (모자이크 없음)
                            if video_writer_raw is not None:
                                video_writer_raw.write(raw_frame)
                            # 처리본(mosaic)은 이미 계산된 결과를 가능한 재사용
                            if video_writer is not None:
                                try:
                                    st = stream_stats.get(websocket_id) or {}
                                    idx = int(st.get("frame_idx", 0))
                                    if user_status.get("is_verified", False):
                                        # 화면은 원본이지만 저장은 모자이크 강제(캐시/N프레임 샘플 재사용)
                                        cached = st.get("last_mosaic")
                                        need_new = True
                                        if MOSAIC_EVERY_N > 1 and idx % MOSAIC_EVERY_N != 0 and cached is not None:
                                            need_new = False
                                        if need_new:
                                            h0, w0 = raw_frame.shape[:2]
                                            if MOSAIC_PROCESS_MAX_WIDTH and MOSAIC_PROCESS_MAX_WIDTH > 0 and w0 > MOSAIC_PROCESS_MAX_WIDTH:
                                                scale = MOSAIC_PROCESS_MAX_WIDTH / float(w0)
                                                small = cv2.resize(raw_frame, (int(w0 * scale), int(h0 * scale)))
                                                mosaic_small = process_frame(small.copy(), mode="face_plate")
                                                rec_proc_frame = cv2.resize(mosaic_small, (w0, h0))
                                            else:
                                                rec_proc_frame = process_frame(raw_frame.copy(), mode="face_plate")
                                            st["last_mosaic"] = rec_proc_frame
                                        else:
                                            rec_proc_frame = cached
                                    else:
                                        rec_proc_frame = processed_frame
                                except Exception:
                                    rec_proc_frame = processed_frame  # fallback
                                video_writer.write(rec_proc_frame)
                    
                    # 전송 FPS 게이팅 + 넌블로킹 코앨레스 전송
                    try:
                        st = stream_stats.get(websocket_id)
                        now_ts_gate = time.time()
                        target_fps = float(os.getenv("STREAM_TARGET_FPS", "15"))
                        can_send = True
                        if target_fps and target_fps > 0:
                            last_ts = float(st.get("last_stream_send_ts", 0)) if st else 0.0
                            min_gap = 1.0 / target_fps
                            if (now_ts_gate - last_ts) < min_gap:
                                can_send = False
                            else:
                                if st is not None:
                                    st["last_stream_send_ts"] = now_ts_gate
                        if can_send:
                            if st is not None and st.get("send_busy"):
                                st["pending_jpg"] = jpg_bytes
                            else:
                                if st is not None:
                                    st["send_busy"] = True
                                asyncio.create_task(_ws_send_coalesced(websocket, websocket_id, jpg_bytes))
                    except Exception:
                        # fallback to blocking send on error
                        try:
                            await websocket.send_bytes(jpg_bytes)
                        except Exception:
                            pass
                    
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
    # (fight detection state cleanup removed)
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

# 클라이언트 서버로부터 userId를 수신하는 API (cameraId 제거)
@app.post("/client/user")
async def receive_user_id(payload: Dict[str, Any] = Body(default={})):  # expects {"userId": string}
    global last_user_info
    # 기본 검증
    if not isinstance(payload, dict):
        return {"success": False, "message": "JSON body가 필요합니다."}
    user_id = payload.get("userId")
    if not user_id or not isinstance(user_id, str):
        return {"success": False, "message": "userId(string)은 필수입니다."}

    ts = datetime.now().isoformat(timespec='seconds')
    last_user_info = {
        "userId": user_id,
        "received_at": ts,
    }
    return {
        "success": True,
        "message": "userId 수신 완료",
    "received": {"userId": user_id},
        "storedAt": ts,
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
## (fight detection endpoints removed)

# 녹화 시작 엔드포인트
@app.post("/start_recording")
async def start_recording():
    global _manual_start_requested
    with recording_lock:
        if is_recording:
            return {"error": "Already recording"}
        _manual_start_requested = True
    # 다음 프레임에서 현재 해상도로 두 파일(처리본/원본) 동시 시작
    return {"status": "Recording will start on next frame", "pending": True}

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

# ==================== 시간 기반 얼굴 검출 API ====================

# 전역 변수
upload_dir = "uploads"
api_results_dir = "api_results"

# 디렉토리 생성
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(api_results_dir, exist_ok=True)

def validate_video_file(filename: str) -> bool:
    """비디오 파일 형식 검증"""
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)

def parse_time_input(time_input: str) -> tuple[int, int]:
    """시간 입력 파싱"""
    try:
        parts = time_input.strip().split()
        if len(parts) == 1:
            # 초만 입력된 경우 (예: "90")
            total_seconds = int(parts[0])
            minutes = total_seconds // 60
            seconds = total_seconds % 60
        elif len(parts) == 2:
            # 분 초 입력된 경우 (예: "1 30")
            minutes = int(parts[0])
            seconds = int(parts[1])
        else:
            raise ValueError("잘못된 시간 형식")
        
        if minutes < 0 or seconds < 0 or seconds >= 60:
            raise ValueError("잘못된 시간 값")
        
        return minutes, seconds
    except (ValueError, IndexError):
        raise ValueError(f"잘못된 시간 형식: {time_input}")

def detect_faces_at_time(video_path: str, start_minutes: int, start_seconds: int) -> dict:
    """특정 시간부터 얼굴 검출"""
    global face_hashes, face_bboxes
    
    try:
        detector = initialize_face_detector()
        
        # 비디오 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception("비디오 파일을 열 수 없습니다")
        
        # 비디오 정보
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # 시작 프레임 계산
        start_time_seconds = start_minutes * 60 + start_seconds
        start_frame = int(start_time_seconds * fps)
        
        # 입력 시간 검증
        if start_time_seconds > duration:
            cap.release()
            return {
                "error": f"입력한 시간({start_minutes}분 {start_seconds}초)이 비디오 길이({duration:.1f}초)를 초과합니다.",
                "video_info": {
                    "duration": duration,
                    "total_frames": total_frames,
                    "fps": fps
                }
            }
        
        # 결과 저장 디렉토리 생성
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_id = f"time_detection_{timestamp}"
        result_dir = os.path.join(api_results_dir, result_id)
        faces_dir = os.path.join(result_dir, "faces")
        os.makedirs(faces_dir, exist_ok=True)
        
        # 얼굴 검출 결과
        detected_faces = []
        unique_faces_count = 0
        total_faces_detected = 0
        
        # 얼굴 해시 및 바운딩 박스 초기화
        face_hashes = []
        face_bboxes = []
        
        # 시작 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        frame_count = start_frame
        # 모델이 얼굴 클래스 제공하는지 확인
        detector_names = None
        try:
            detector_names = getattr(detector, 'names', None)
            if detector_names is None and hasattr(detector, 'model'):
                detector_names = getattr(detector.model, 'names', None)
        except Exception:
            detector_names = None

        def _is_face_class(result_obj, cls_id: int) -> bool:
            try:
                # 우선순위: result.names -> detector names -> fallback 문자열 비교
                names_map = getattr(result_obj, 'names', None) or detector_names
                if isinstance(names_map, dict) and cls_id in names_map:
                    return 'face' in str(names_map[cls_id]).lower()
                if isinstance(names_map, (list, tuple)) and 0 <= cls_id < len(names_map):
                    return 'face' in str(names_map[cls_id]).lower()
            except Exception:
                pass
            # 이름을 알 수 없으면 보수적으로 False
            return False

        def _box_valid_shape(x1, y1, x2, y2) -> bool:
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            if w < 24 or h < 24:
                return False
            ar = (w / h) if h > 0 else 0
            return 0.6 <= ar <= 1.8

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # 1초마다 프레임 처리 (성능 최적화)
            if frame_count % int(fps) == 0:
                candidate_boxes: list[tuple[int,int,int,int]] = []
                used_cascade = False

                # 1) YOLO 얼굴 모델이면 YOLO 결과 사용 + 검증
                if hasattr(detector, 'predict'):
                    try:
                        results = detector(frame, verbose=False)
                        for result in results:
                            boxes = getattr(result, 'boxes', None)
                            if boxes is None:
                                continue
                            for box in boxes:
                                conf = float(box.conf[0]) if hasattr(box, 'conf') else 1.0
                                if conf < FACE_DETECTION_CONFIDENCE_THRESHOLD:
                                    continue
                                # 클래스 이름이 face인지 확인 (모델에 face 클래스가 없는 경우 건너뜀)
                                cls_id = int(box.cls[0]) if hasattr(box, 'cls') else -1
                                if not _is_face_class(result, cls_id):
                                    continue
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                if not _box_valid_shape(x1, y1, x2, y2):
                                    continue
                                roi = frame[y1:y2, x1:x2]
                                if not _verify_face_region(roi):
                                    continue
                                candidate_boxes.append((x1, y1, x2, y2))
                    except Exception:
                        pass

                # 2) face 클래스가 전혀 안 나왔다면 Haar cascade로 대체
                if not candidate_boxes:
                    used_cascade = True
                    candidate_boxes = _detect_faces_cascade(frame)

                # NMS로 정리
                final_boxes = _nms(candidate_boxes, iou_thresh=0.45)

                for (x1, y1, x2, y2) in final_boxes:
                    face_image = frame[y1:y2, x1:x2]
                    if face_image is None or face_image.size == 0:
                        continue
                    # cascade에서 온 경우에도 한 번 더 빠른 검증
                    if used_cascade and not _box_valid_shape(x1, y1, x2, y2):
                        continue
                    face_hash = calculate_image_hash(face_image)
                    if not face_hash:
                        continue
                    total_faces_detected += 1
                    bbox_coords = [int(x1), int(y1), int(x2), int(y2)]
                    if not is_duplicate_face(face_hash, bbox_coords):
                        face_hashes.append(face_hash)
                        face_bboxes.append(bbox_coords)
                        unique_faces_count += 1
                        face_filename = f"face_{unique_faces_count:03d}_{timestamp}.jpg"
                        face_path = os.path.join(faces_dir, face_filename)
                        cv2.imwrite(face_path, face_image)
                        s3_key = f"api_results/{result_id}/faces/{face_filename}"
                        s3_url = upload_face_to_s3(face_path, s3_key)
                        detected_faces.append({"s3_url": s3_url})
            
            frame_count += 1
            
            # 환경 변수에서 설정한 시간 후 중단 (성능 최적화)
            if frame_count - start_frame > int(fps * PROCESSING_DURATION_SECONDS):
                break
        
        cap.release()
        
        # 결과 요약 저장
        summary = {
            "faces": detected_faces
        }
        
        # 결과 JSON 저장
        summary_path = os.path.join(result_dir, "face_records.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        return summary
        
    except Exception as e:
        raise Exception(f"얼굴 검출 중 오류 발생: {str(e)}")

@app.get("/face-detection")
async def face_detection_root():
    """얼굴 검출 API 기본 정보"""
    return {
        "message": "시간 기반 얼굴 검출 API",
        "version": "1.0.0",
        "description": "영상의 특정 분,초를 입력하면 사람의 얼굴이 나오는 부분을 이미지로 저장하여 출력 (얼굴중복은 저장하지 않음)",
        "endpoints": {
            "upload_video": "/face-detection/upload-video",
            "detect_faces": "/face-detection/detect-faces",
            "video_info": "/face-detection/video-info/{filename}",
            "results": "/face-detection/results/{result_id}",
            "download_face": "/face-detection/download-face/{result_id}/{filename}"
        }
    }

@app.post("/face-detection/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """비디오 파일 업로드"""
    try:
        # 파일 검증
        if not validate_video_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail="지원하지 않는 비디오 형식입니다. MP4, AVI, MOV, MKV, WMV, FLV 파일만 지원합니다."
            )
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(upload_dir, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "message": "비디오 업로드 성공",
            "filename": filename,
            "file_path": file_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 오류: {str(e)}")

@app.post("/face-detection/detect-faces")
async def detect_faces(
    filename: str = Query(None, description="비디오 파일명 (로컬 또는 S3)"),
    time_input: str = Query(..., description="시작 시간 (예: '1 30' = 1분 30초부터, '90' = 90초부터)"),
    from_s3: bool = Query(False, description="S3에서 파일을 가져올지 여부"),
    video_url: str = Query(None, description="동영상 URL (filename 대신 사용)"),
    file: UploadFile = File(None),
):
    """특정 시간부터 얼굴 검출 (blob:/data: URL 및 직접 업로드 지원)"""
    try:
        # 시간 입력 파싱 (분,초 또는 총초)
        try:
            parts = time_input.split()
            if len(parts) == 2:
                start_minutes = int(parts[0])
                start_seconds = int(parts[1])
            elif len(parts) == 1:
                total_seconds = int(parts[0])
                start_minutes = total_seconds // 60
                start_seconds = total_seconds % 60
            else:
                raise ValueError("잘못된 시간 형식입니다. '1 30' 또는 '90' 형식으로 입력하세요.")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"시간 파싱 오류: {str(e)}")

        file_path = None
        temp_created = False

        if video_url:
            # 1) blob: URL → 서버 직접 접근 불가, 업로드 파일 필요
            if video_url.startswith("blob:"):
                if file is None:
                    raise HTTPException(status_code=400, detail=(
                        "blob: URL은 서버에서 접근할 수 없습니다. 파일을 함께 업로드하시거나 "
                        "먼저 /face-detection/upload-video로 업로드 후 filename으로 호출하세요."
                    ))
                temp_path = os.path.join(tempfile.gettempdir(), f"upload_{int(time.time())}_{file.filename or 'video'}.mp4")
                with open(temp_path, "wb") as bf:
                    shutil.copyfileobj(file.file, bf)
                file_path = temp_path
                temp_created = True
            # 2) data: URL (data:video/mp4;base64,....)
            elif video_url.startswith("data:"):
                try:
                    header, b64 = video_url.split(",", 1)
                    raw = base64.b64decode(b64)
                    temp_path = os.path.join(tempfile.gettempdir(), f"dataurl_{int(time.time())}.mp4")
                    with open(temp_path, "wb") as ftmp:
                        ftmp.write(raw)
                    file_path = temp_path
                    temp_created = True
                except Exception as de:
                    raise HTTPException(status_code=400, detail=f"data: URL 파싱 실패: {de}")
            # 3) http(s) 등 → 다운로드 후 처리
            else:
                try:
                    import requests
                    tmp_path = os.path.join(tempfile.gettempdir(), f"url_dl_{int(time.time())}.mp4")
                    with requests.get(video_url, stream=True, timeout=(5, 60)) as r:
                        r.raise_for_status()
                        with open(tmp_path, 'wb') as f:
                            for chunk in r.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                    file_path = tmp_path
                    temp_created = True
                except Exception as de:
                    raise HTTPException(status_code=400, detail=f"URL 다운로드 실패: {de}")
        elif from_s3:
            if not USE_S3 or not s3_client:
                raise HTTPException(status_code=400, detail="S3가 설정되지 않았습니다")
            try:
                s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=f"recordings/{filename}")
                temp_file = os.path.join(tempfile.gettempdir(), f"temp_{filename}")
                s3_client.download_file(S3_BUCKET_NAME, f"recordings/{filename}", temp_file)
                file_path = temp_file
            except Exception:
                raise HTTPException(status_code=404, detail=f"S3에서 파일을 찾을 수 없습니다: {filename}")
        else:
            if file is not None:
                temp_path = os.path.join(tempfile.gettempdir(), f"upload_{int(time.time())}_{file.filename or 'video'}.mp4")
                with open(temp_path, "wb") as bf:
                    shutil.copyfileobj(file.file, bf)
                file_path = temp_path
                temp_created = True
            elif not filename:
                raise HTTPException(status_code=400, detail="filename 또는 video_url 중 하나는 필수입니다")
            else:
                file_path = os.path.join(upload_dir, filename)
                if not os.path.exists(file_path):
                    raise HTTPException(status_code=404, detail="업로드된 비디오 파일을 찾을 수 없습니다")

        # 최종 처리
        result = detect_faces_at_time(file_path, start_minutes, start_seconds)

        # 임시 파일 삭제
        if temp_created and file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception:
                pass

        if "error" in result:
            return JSONResponse(content=result, status_code=400)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"얼굴 검출 오류: {str(e)}")

@app.get("/face-detection/video-info/{filename}")
async def get_video_info(filename: str):
    """비디오 정보 조회"""
    try:
        file_path = os.path.join(upload_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, 
                detail="업로드된 비디오 파일을 찾을 수 없습니다"
            )
        
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400, 
                detail="비디오 파일을 열 수 없습니다"
            )
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return {
            "filename": filename,
            "video_info": {
                "duration": duration,
                "total_frames": total_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_formatted": f"{int(duration//60)}분 {int(duration%60)}초"
            },
            "time_input_examples": [
                "0 0 - 전체 비디오",
                "0 30 - 30초부터",
                "1 0 - 1분부터",
                "1 30 - 1분 30초부터",
                "90 - 90초부터"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"비디오 정보 조회 오류: {str(e)}")

@app.get("/face-detection/results/{result_id}")
async def get_results(result_id: str):
    """검출 결과 조회"""
    try:
        result_dir = os.path.join(api_results_dir, result_id)
        if not os.path.exists(result_dir):
            raise HTTPException(
                status_code=404, 
                detail="검출 결과를 찾을 수 없습니다"
            )
        
        # 결과 JSON 파일 읽기
        summary_path = os.path.join(result_dir, "face_records.json")
        if not os.path.exists(summary_path):
            raise HTTPException(
                status_code=404, 
                detail="결과 파일을 찾을 수 없습니다"
            )
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 조회 오류: {str(e)}")

@app.get("/face-detection/download-face/{result_id}/{filename}")
async def download_face(result_id: str, filename: str):
    """얼굴 이미지 다운로드"""
    try:
        face_path = os.path.join(api_results_dir, result_id, "faces", filename)
        if not os.path.exists(face_path):
            raise HTTPException(
                status_code=404, 
                detail="얼굴 이미지를 찾을 수 없습니다"
            )
        
        return FileResponse(face_path, media_type="image/jpeg", filename=filename)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 다운로드 오류: {str(e)}")

@app.get("/face-detection/results")
async def list_all_results():
    """모든 결과 목록 조회"""
    try:
        results = []
        if os.path.exists(api_results_dir):
            for result_id in os.listdir(api_results_dir):
                result_dir = os.path.join(api_results_dir, result_id)
                if os.path.isdir(result_dir):
                    summary_path = os.path.join(result_dir, "face_records.json")
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)
                            results.append({
                                "result_id": result_id,
                                "detection_info": result_data.get("detection_info", {}),
                                "created_at": result_id.split("_")[-1] if "_" in result_id else result_id
                            })
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 목록 조회 오류: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # 환경 변수에서 호스트와 포트 가져오기
    load_dotenv()
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    print(f"통합 API 서버 시작: http://{host}:{port}")
    print(f"업로드 디렉토리: {upload_dir}")
    print(f"결과 디렉토리: {api_results_dir}")
    print(f"🔧 설정값:")
    print(f"   - 얼굴 검출 신뢰도 임계값: {FACE_DETECTION_CONFIDENCE_THRESHOLD}")
    print(f"   - 얼굴 유사도 임계값: {FACE_SIMILARITY_THRESHOLD}")
    print(f"   - 처리 시간: {PROCESSING_DURATION_SECONDS}초")
    print(f"   - S3 버킷: {S3_BUCKET_NAME if USE_S3 else '비활성화'}")
    print(f"   - S3 사용: {'활성화' if USE_S3 else '비활성화'}")
    print(f"API 엔드포인트:")
    print(f"   - WebSocket: ws://{host}:{port}/ws/video")
    print(f"   - 얼굴 검출: http://{host}:{port}/face-detection")
    print(f"   - 녹화 관리: http://{host}:{port}/recordings")
    
    uvicorn.run(app, host=host, port=port)