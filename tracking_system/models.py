"""
데이터 모델 및 타입 정의 모듈
"""

import numpy as np
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, Protocol, List, Dict, Any

class TrackingState(Enum):
    """추적 상태"""
    NO_TARGET = "no_target"
    TRACKING = "tracking" 
    SEARCHING = "searching"
    SUSPENDED = "suspended"

@dataclass
class Detection:
    """기본 검출 결과 클래스"""
    x: int
    y: int
    w: int
    h: int
    confidence: float = 1.0
    
    @property
    def center(self) -> Tuple[int, int]:
        """얼굴 중심점 좌표"""
        return (self.x + self.w // 2, self.y + self.h // 2)
    
    @property
    def area(self) -> int:
        """얼굴 영역 넓이"""
        return self.w * self.h
    
    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        """바운딩 박스 (x, y, w, h)"""
        return (self.x, self.y, self.w, self.h)
    
    def distance_to(self, other: 'Detection') -> float:
        """다른 검출과의 중심점 거리"""
        x1, y1 = self.center
        x2, y2 = other.center
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    
    def is_close_to_point(self, x: int, y: int, max_distance: Optional[float] = None) -> bool:
        """특정 점과의 근접성 확인"""
        if max_distance is None:
            max_distance = max(self.w, self.h)
        
        center_x, center_y = self.center
        distance = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        return distance <= max_distance

@dataclass
class FaceDetection(Detection):
    """얼굴 검출 결과"""
    pass

@dataclass 
class PersonDetection(Detection):
    """사람 전체 검출 결과"""
    face_detection: Optional[FaceDetection] = None
    score: float = 0.0  # NMS용 점수
    
    def has_face(self) -> bool:
        """얼굴이 포함되어 있는지 확인"""
        return self.face_detection is not None

@dataclass
class TrackingTarget:
    """추적 대상 정보"""
    name: str
    embedding: np.ndarray
    last_face_detection: Optional[FaceDetection] = None
    last_person_detection: Optional[PersonDetection] = None
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def update_face_detection(self, detection: FaceDetection):
        """마지막 얼굴 검출 정보 업데이트"""
        self.last_face_detection = detection
    
    def update_person_detection(self, detection: PersonDetection):
        """마지막 사람 검출 정보 업데이트"""
        self.last_person_detection = detection

@dataclass
class StoredPerson:
    """저장된 사람 정보"""
    id: str
    name: str
    embedding: np.ndarray
    face_image: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class RecognitionResult:
    """인식 결과"""
    person: StoredPerson
    similarity: float
    face_detection: FaceDetection
    confidence: float = 1.0

# 얼굴 임베딩 프로토콜 정의
class FaceEmbedder(Protocol):
    """얼굴 임베딩 인터페이스"""
    def compute_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 ROI에서 임베딩 추출"""
        ...

@dataclass
class TrackingConfig:
    """추적 관련 설정"""
    similarity_threshold: float
    strict_threshold: float
    max_movement_multiplier: float
    lost_frame_threshold: int
    
    @classmethod
    def for_insight_face(cls) -> 'TrackingConfig':
        """InsightFace용 설정"""
        from .config import Config
        return cls(
            similarity_threshold=Config.SIMILARITY_THRESHOLD_INSIGHT,
            strict_threshold=Config.STRICT_THRESHOLD_INSIGHT,
            max_movement_multiplier=Config.MAX_MOVEMENT_MULTIPLIER,
            lost_frame_threshold=Config.LOST_FRAME_THRESHOLD
        )
    
    @classmethod
    def for_histogram(cls) -> 'TrackingConfig':
        """히스토그램용 설정"""
        from .config import Config
        return cls(
            similarity_threshold=Config.SIMILARITY_THRESHOLD_HISTOGRAM,
            strict_threshold=Config.STRICT_THRESHOLD_HISTOGRAM,
            max_movement_multiplier=Config.MAX_MOVEMENT_MULTIPLIER,
            lost_frame_threshold=Config.LOST_FRAME_THRESHOLD
        )

@dataclass
class TrackingResult:
    """추적 결과"""
    state: TrackingState
    face_detection: Optional[FaceDetection] = None
    person_detection: Optional[PersonDetection] = None
    similarity: float = 0.0
    target_name: Optional[str] = None
    
    @property
    def is_tracking(self) -> bool:
        """추적 중인지 확인"""
        return self.state == TrackingState.TRACKING
    
    @property
    def is_lost(self) -> bool:
        """타겟을 잃었는지 확인"""
        return self.state in [TrackingState.SEARCHING, TrackingState.SUSPENDED]
