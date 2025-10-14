# =====================================================================
# Module: server.config
# Purpose: 환경 변수 로드 및 서버/자동녹화/스트림 관련 런타임 설정 상수 정의.
# Responsibilities:
#   - .env 로부터 설정 읽기 및 기본값 지정
#   - 외부 사용을 위한 __all__ export 목록 유지
# Design Notes:
#   - 값 변경이 필요한 경우 import 후 직접 할당(단일 프로세스 가정)
#   - 다중 프로세스/컨테이너 환경에서는 재동기화 전략 필요
# Extension Tips:
#   - Validation/변환이 복잡해지면 pydantic BaseSettings 도입 고려
#   - 동적 reload 필요 시 파일 watch + 재할당 패턴 적용
# =====================================================================
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

# Revision marker
SERVER_REV = os.getenv("SERVER_REV", "2025-08-16T13:35Z-refactor1")

# Backend / Auth
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080")
AI_API_KEY = os.getenv("AI_API_KEY")

# Auto recording settings
AUTO_RECORD_ENABLED = os.getenv("AUTO_RECORD_ENABLED", "1") in ("1","true","True")
AUTO_RECORD_THRESHOLD = int(os.getenv("AUTO_RECORD_THRESHOLD", "1"))
AUTO_ZERO_TIMEOUT_SEC = float(os.getenv("AUTO_ZERO_TIMEOUT_SEC", "3.0"))
AUTO_RECORD_DEBUG = os.getenv("AUTO_RECORD_DEBUG", "1") in ("1","true","True")
AUTO_RECORD_TEMP_DISABLED = os.getenv("AUTO_RECORD_TEMP_DISABLED", "0") in ("1","true","True")
DETECT_EVERY_N = int(os.getenv("DETECT_EVERY_N", "2"))
MOSAIC_EVERY_N = int(os.getenv("MOSAIC_EVERY_N", "2"))
MOSAIC_PROCESS_MAX_WIDTH = int(os.getenv("MOSAIC_PROCESS_MAX_WIDTH", "640"))
AUTO_PRESENCE_WINDOW = int(os.getenv("AUTO_PRESENCE_WINDOW", "20"))
AUTO_PRESENCE_MIN_HITS = int(os.getenv("AUTO_PRESENCE_MIN_HITS", "5"))
# 안정화 파라미터: 최소 녹화 지속시간과 중단 후 쿨다운(초)
MIN_RECORD_DURATION_SEC = float(os.getenv("MIN_RECORD_DURATION_SEC", "5.0"))
COOLDOWN_AFTER_STOP_SEC = float(os.getenv("COOLDOWN_AFTER_STOP_SEC", "5.0"))
STREAM_MAX_WIDTH = int(os.getenv("STREAM_MAX_WIDTH", "960"))
STREAM_JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "65"))
STREAM_TARGET_FPS = float(os.getenv("STREAM_TARGET_FPS", "12"))
RECORD_SAVE_RAW = os.getenv("RECORD_SAVE_RAW", "1") in ("1","true","True")
RECORD_BY_MOSAIC = os.getenv("RECORD_BY_MOSAIC", "1") in ("1","true","True")

# Face detection(time based) thresholds
FACE_DETECTION_CONFIDENCE_THRESHOLD = float(os.getenv('FACE_DETECTION_CONFIDENCE_THRESHOLD', '0.5'))
FACE_SIMILARITY_THRESHOLD = float(os.getenv('FACE_SIMILARITY_THRESHOLD', '0.95'))
PROCESSING_DURATION_SECONDS = int(os.getenv('PROCESSING_DURATION_SECONDS', '10'))

# S3
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
USE_S3 = all([S3_BUCKET_NAME, S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY])

# Spring endpoint
SPRING_MAKE_ENTITY_URL = os.getenv("SPRING_MAKE_ENTITY_URL", "http://localhost:8080/api/videos/make-entity")

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "uploads")
API_RESULTS_DIR = os.getenv("API_RESULTS_DIR", "api_results")

# --- Transcode settings (for S3 upload) -------------------------------------
# ffmpeg 바이너리 경로 및 H.264 인코딩 품질 설정
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
H264_TRANSCODE_BEFORE_UPLOAD = os.getenv("H264_TRANSCODE_BEFORE_UPLOAD", "1") in ("1","true","True")
# 대상별 트랜스코드 토글(기본: 처리본만 트랜스코드, 원본은 그대로 업로드)
TRANSCODE_PROCESSED_BEFORE_UPLOAD = os.getenv("TRANSCODE_PROCESSED_BEFORE_UPLOAD", os.getenv("H264_TRANSCODE_BEFORE_UPLOAD", "1")) in ("1","true","True")
TRANSCODE_RAW_BEFORE_UPLOAD = os.getenv("TRANSCODE_RAW_BEFORE_UPLOAD", "1") in ("1","true","True")
H264_CRF = int(os.getenv("H264_CRF", "23"))            # 18(고품질)~28(저품질)
H264_PRESET = os.getenv("H264_PRESET", "veryfast")      # ultrafast..placebo
H264_PIXEL_FORMAT = os.getenv("H264_PIXEL_FORMAT", "yuv420p")
# ffmpeg 내부 스레드 개수 제한(고성능 Mac에서 과부하 방지)
H264_THREADS = int(os.getenv("H264_THREADS", "2"))      # 0은 ffmpeg 기본값(자동)
# 업로드 동시성 제한(대용량 병렬 업로드로 인한 과부하 방지)
UPLOAD_MAX_CONCURRENCY = int(os.getenv("UPLOAD_MAX_CONCURRENCY", "1"))

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(API_RESULTS_DIR, exist_ok=True)

HOST = os.getenv('API_HOST', '0.0.0.0')
PORT = int(os.getenv('API_PORT', '8000'))

__all__ = [
    'SERVER_REV','BACKEND_API_URL','AI_API_KEY','AUTO_RECORD_ENABLED','AUTO_RECORD_THRESHOLD','AUTO_ZERO_TIMEOUT_SEC',
    'AUTO_RECORD_DEBUG','AUTO_RECORD_TEMP_DISABLED','DETECT_EVERY_N','MOSAIC_EVERY_N','MOSAIC_PROCESS_MAX_WIDTH',
    'AUTO_PRESENCE_WINDOW','AUTO_PRESENCE_MIN_HITS','STREAM_MAX_WIDTH','STREAM_JPEG_QUALITY','STREAM_TARGET_FPS',
    'FACE_DETECTION_CONFIDENCE_THRESHOLD','FACE_SIMILARITY_THRESHOLD','PROCESSING_DURATION_SECONDS',
    'S3_BUCKET_NAME','S3_REGION','AWS_ACCESS_KEY_ID','AWS_SECRET_ACCESS_KEY','USE_S3','SPRING_MAKE_ENTITY_URL',
    'UPLOAD_DIR','API_RESULTS_DIR','HOST','PORT','RECORD_SAVE_RAW','RECORD_BY_MOSAIC',
    'MIN_RECORD_DURATION_SEC','COOLDOWN_AFTER_STOP_SEC',
    # Transcode exports
    'FFMPEG_BIN','H264_TRANSCODE_BEFORE_UPLOAD','TRANSCODE_PROCESSED_BEFORE_UPLOAD','TRANSCODE_RAW_BEFORE_UPLOAD',
    'H264_CRF','H264_PRESET','H264_PIXEL_FORMAT','H264_THREADS','UPLOAD_MAX_CONCURRENCY'
]
