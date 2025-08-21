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
from datetime import datetime
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
MOSAIC_EVERY_N = int(os.getenv("MOSAIC_EVERY_N", "1"))
MOSAIC_PROCESS_MAX_WIDTH = int(os.getenv("MOSAIC_PROCESS_MAX_WIDTH", "0"))
AUTO_PRESENCE_WINDOW = int(os.getenv("AUTO_PRESENCE_WINDOW", "20"))
AUTO_PRESENCE_MIN_HITS = int(os.getenv("AUTO_PRESENCE_MIN_HITS", "5"))
STREAM_MAX_WIDTH = int(os.getenv("STREAM_MAX_WIDTH", "960"))
STREAM_JPEG_QUALITY = int(os.getenv("STREAM_JPEG_QUALITY", "70"))
STREAM_TARGET_FPS = float(os.getenv("STREAM_TARGET_FPS", "15"))

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
    'UPLOAD_DIR','API_RESULTS_DIR','HOST','PORT'
]
