# =====================================================================
# Module: server.analytics.person_count
# Purpose: 프레임 내 사람 수 추정 (YOLO 기반 AnalyticsEngine → HOG fallback 순).
# Responsibilities:
#   - YOLO 구성/초기화 (사용자 yolo_config.json 병합)
#   - 실패 시 간단 HOGDescriptor 로 사람 수 추정
#   - 디버그 상태(state._auto_debug)에 엔진/오류 정보 기록
# Design Notes:
#   - 성능을 위해 YOLO는 한 번 초기화 후 재사용 (AUTO_ENGINE)
#   - HOG fallback 은 해상도를 640 기준으로 축소하여 속도 최적화
#   - 외부 의존성 실패시 조용히 fallback (서비스 지속성 우선)
# Extension Tips:
#   - 추가 모델(예: OpenVINO / TensorRT) 계층적 시도 가능
#   - 추후 추정 신뢰도/바운딩박스 반환하도록 인터페이스 확장 고려
# =====================================================================
from __future__ import annotations
import os
import cv2
import numpy as np
from typing import Any
from ..core import state
from ..config import AUTO_RECORD_DEBUG

# Optional AnalyticsEngine
try:
    from analytics import AnalyticsEngine  # type: ignore
except Exception:  # pragma: no cover
    AnalyticsEngine = None  # type: ignore

# Optional YOLO (used indirectly inside AnalyticsEngine)
try:
    from ultralytics import YOLO  # noqa: F401  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore

AUTO_ENGINE = None
if AnalyticsEngine is not None:
    try:
        _yolo_cfg: dict[str, Any] = {
            "yolo_model": "yolov8n.pt",
            "device": None,
            "conf": 0.35,
            "iou": 0.45,
            "sample_rate": 1,
        }
        cfg_path = os.path.join(os.getcwd(), "yolo_config.json")
        if os.path.exists(cfg_path):
            import json as _json
            with open(cfg_path, "r", encoding="utf-8") as f:
                user_cfg = _json.load(f)
                if isinstance(user_cfg, dict):
                    _yolo_cfg.update(user_cfg)
        AUTO_ENGINE = AnalyticsEngine(
            yolo_model=_yolo_cfg.get("yolo_model", "yolov8n.pt"),
            device=_yolo_cfg.get("device"),
            conf=float(_yolo_cfg.get("conf", 0.35)),
            iou=float(_yolo_cfg.get("iou", 0.45)),
            sample_rate=int(_yolo_cfg.get("sample_rate", 1))
        )
        if AUTO_RECORD_DEBUG:
            print("[AUTO] AnalyticsEngine initialized for person counting")
    except Exception as _e:  # pragma: no cover
        print(f"Auto-record AnalyticsEngine init failed: {_e}")

# Fallback HOG
_HOG = None
try:
    _HOG = cv2.HOGDescriptor()
    _HOG.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    state._auto_debug["hog_available"] = True
except Exception as _he:  # pragma: no cover
    _HOG = None
    if AUTO_RECORD_DEBUG:
        print(f"[AUTO] HOG init failed: {_he}")


def estimate_person_count(frame: np.ndarray) -> int:
    """YOLO(AnalyticsEngine) → HOG fallback."""
    count = 0
    try:
        if AUTO_ENGINE is not None and getattr(AUTO_ENGINE, 'yolo', None) is not None:
            persons = AUTO_ENGINE._detect_persons(frame) or []  # type: ignore[attr-defined]
            count = len(persons) if isinstance(persons, list) else 0
            state._auto_debug["engine_available"] = True
            return count
    except Exception as e:  # pragma: no cover
        state._auto_debug["last_error"] = f"YOLO detect error: {e}"
    try:
        if _HOG is not None:
            h, w = frame.shape[:2]
            scale = 640.0 / max(w, h)
            if scale < 1.0:
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                small = frame
            rects, _ = _HOG.detectMultiScale(small, winStride=(8, 8), padding=(8, 8), scale=1.05)
            count = len(rects) if rects is not None else 0
            return count
    except Exception as e:  # pragma: no cover
        state._auto_debug["last_error"] = f"HOG detect error: {e}"
    return 0

__all__ = ["estimate_person_count", "AUTO_ENGINE"]
