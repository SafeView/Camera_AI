# =====================================================================
# Module: server.core.state
# Purpose: 서버 전역에서 공유되는 실시간 스트림/검증/자동녹화 상태를 보관.
# Responsibilities:
#   - WebSocket 연결 추적(active_websockets, stream_stats)
#   - 사용자 검증 상태(verified_users + lock)
#   - 녹화 세션 메타/Writer 핸들 관리(is_recording, video_writer 등)
#   - 자동 녹화 디버그/타이밍 변수(_auto_debug, *_nonzero_person_ts)
# Design Notes:
#   - 단순 전역 모듈 변수로 구성되어 FastAPI lifespan 과 동일한 생명주기.
#   - 복수 코루틴 동시 접근 구간은 recording_lock / verification_lock 으로 최소한 보호.
#   - 퍼포먼스를 위해 세부 동시성 정합(예: stream_stats 내부 dict)은 강한 일관성 대신 최종 일관성 허용.
# Concurrency:
#   - 녹화 관련 Writer 열기/닫기: recording_lock 필수
#   - 검증 업데이트/조회: verification_lock 필수
#   - 나머지 읽기는 GIL + 단일 프로세스 가정 하 경쟁 허용
# Extension Tips:
#   - 상태 추가 시 __all__ 업데이트로 외부 import 안정성 유지
#   - 고빈도 필드(프레임 카운터 등)는 atomic 필요 없으면 단순 할당 허용
# =====================================================================
from __future__ import annotations
import threading
import tempfile
from typing import Dict, Any, Set, Optional

# --- Active WebSocket Set ----------------------------------------------------
# WebSocket 객체 참조를 직접 저장 (id 기반 분리) : 종료시 discard
active_websockets: Set[object] = set()

# --- Per Connection Stream Statistics ----------------------------------------
# frame_idx, detections, auto_starts, presence_hist 등 실시간 메타
stream_stats: Dict[int, Dict[str, Any]] = {}

# --- Last Ended Stream Snapshot ----------------------------------------------
last_stream_snapshot: Optional[Dict[str, Any]] = None

# --- Verification State ------------------------------------------------------
# ws_id -> {is_verified, decryption_token, camera_id}
verified_users: Dict[int, Dict[str, Any]] = {}
verification_lock = threading.Lock()  # 검증 관련 상태 보호

# --- Recording Session State -------------------------------------------------
# 현재/직전 녹화 관련 공유 자원
is_recording: bool = False
video_writer = None          # 처리(모자이크) 영상 Writer
video_writer_raw = None      # 원본 영상 Writer
recording_filename: Optional[str] = None
recording_filename_raw: Optional[str] = None
recording_lock = threading.Lock()  # Writer/파일명 동시성 보호
TEMP_DIR = tempfile.gettempdir()

# --- Recording Meta ----------------------------------------------------------
_recording_started_at_ts: Optional[float] = None
_recording_max_persons: int = 0
_recording_ws_id: Optional[int] = None
_manual_start_requested: bool = False
_stop_task = None  # asyncio.Task | None : 비동기 finalize 작업 핸들
# 최근 녹화 시작/종료 시각 (쿨다운/최소 지속시간 판단용)
_last_record_start_ts: Optional[float] = None
_last_record_stop_ts: Optional[float] = None

# --- Auto Recording Debug Structure ------------------------------------------
_auto_debug = {
    "enabled": False,
    "threshold": 1,
    "last_check_at": None,
    "last_person_count": None,
    "engine_available": False,
    "hog_available": False,
    "attempted_start": False,
    "started": False,
    "last_error": None,
}

# --- Presence Timing (Stop 조건 계산용) --------------------------------------
_last_nonzero_person_ts: Optional[float] = None
_zero_since_ts: Optional[float] = None

# --- Last Received User Info -------------------------------------------------
last_user_info: Optional[Dict[str, Any]] = None

__all__ = [
    'active_websockets','stream_stats','last_stream_snapshot','verified_users','verification_lock',
    'is_recording','video_writer','video_writer_raw','recording_filename','recording_filename_raw','recording_lock','TEMP_DIR',
    '_recording_started_at_ts','_recording_max_persons','_recording_ws_id','_manual_start_requested','_stop_task',
    '_auto_debug','_last_nonzero_person_ts','_zero_since_ts','last_user_info',
    '_last_record_start_ts','_last_record_stop_ts'
]
