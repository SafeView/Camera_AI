"""
얼굴 검출 모듈
"""

import cv2
import numpy as np
from typing import List
from .models import FaceDetection
from .config import Config
from .dependencies import Dependencies

class FaceDetector:
    """얼굴 검출기 (MediaPipe + Haar Cascade 하이브리드)"""
    
    def __init__(self, dependencies: Dependencies):
        self.deps = dependencies
        
        # Haar Cascade 초기화
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # MediaPipe 초기화
        self.mediapipe_detector = None
        if self.deps.mediapipe_available:
            self._init_mediapipe()
    
    def _init_mediapipe(self):
        """MediaPipe 얼굴 검출기 초기화"""
        try:
            import mediapipe as mp
            mp_face_detection = mp.solutions.face_detection
            self.mediapipe_detector = mp_face_detection.FaceDetection(
                model_selection=0, 
                min_detection_confidence=Config.MEDIAPIPE_MIN_DETECTION_CONFIDENCE
            )
        except Exception as e:
            self.mediapipe_detector = None
    
    def detect_faces(self, frame: np.ndarray) -> List[FaceDetection]:
        """얼굴 검출 (MediaPipe + Haar Cascade 하이브리드)"""
        detections = []
        h, w = frame.shape[:2]
        
        # MediaPipe 검출
        if self.mediapipe_detector:
            detections.extend(self._detect_with_mediapipe(frame, w, h))
        
        # Haar Cascade 검출
        detections.extend(self._detect_with_haar(frame, w, h))
        
        # 중복 제거 및 필터링
        return self._filter_and_deduplicate(detections, w, h)
    
    def _detect_with_mediapipe(self, frame: np.ndarray, w: int, h: int) -> List[FaceDetection]:
        """MediaPipe로 얼굴 검출"""
        detections = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.mediapipe_detector.process(rgb_frame)
        
        if results.detections:
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                confidence = detection.score[0]
                
                detections.append(FaceDetection(x, y, width, height, confidence))
        
        return detections
    
    def _detect_with_haar(self, frame: np.ndarray, w: int, h: int) -> List[FaceDetection]:
        """Haar Cascade로 얼굴 검출"""
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        max_size = (int(w * Config.MAX_FACE_SIZE_RATIO), int(h * Config.MAX_FACE_SIZE_RATIO))
        
        faces = self.haar_cascade.detectMultiScale(
            gray, 
            scaleFactor=Config.HAAR_SCALE_FACTOR, 
            minNeighbors=Config.HAAR_MIN_NEIGHBORS,
            minSize=Config.HAAR_MIN_SIZE, 
            maxSize=max_size
        )
        
        for (x, y, w_face, h_face) in faces:
            detections.append(FaceDetection(x, y, w_face, h_face))
        
        return detections
    
    def _filter_and_deduplicate(self, detections: List[FaceDetection], w: int, h: int) -> List[FaceDetection]:
        """검출 결과 필터링 및 중복 제거"""
        # 유효성 검사
        valid_detections = [d for d in detections if self._is_valid_face(d, w, h)]
        
        # 중복 제거 (면적 기준 정렬)
        valid_detections.sort(key=lambda d: d.area, reverse=True)
        
        unique_detections = []
        for detection in valid_detections:
            if not any(self._calculate_iou(detection, existing) > Config.IOU_THRESHOLD 
                      for existing in unique_detections):
                unique_detections.append(detection)
        
        return unique_detections
    
    def _is_valid_face(self, detection: FaceDetection, frame_w: int, frame_h: int) -> bool:
        """얼굴 검출 유효성 검사"""
        x, y, w, h = detection.x, detection.y, detection.w, detection.h
        
        # 크기 검사
        if w < Config.MIN_FACE_SIZE or h < Config.MIN_FACE_SIZE:
            return False
        if w > frame_w * Config.MAX_FACE_SIZE_RATIO or h > frame_h * Config.MAX_FACE_SIZE_RATIO:
            return False
        
        # 비율 검사
        aspect_ratio = w / h
        if aspect_ratio < Config.MIN_ASPECT_RATIO or aspect_ratio > Config.MAX_ASPECT_RATIO:
            return False
        
        # 경계 검사
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            return False
        
        return True
    
    def _calculate_iou(self, detection1: FaceDetection, detection2: FaceDetection) -> float:
        """IoU(Intersection over Union) 계산"""
        x1, y1, w1, h1 = detection1.bbox
        x2, y2, w2, h2 = detection2.bbox
        
        # 교집합 계산
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def find_closest_face(self, detections: List[FaceDetection], click_point: tuple) -> FaceDetection:
        """클릭 지점에서 가장 가까운 얼굴 찾기"""
        click_x, click_y = click_point
        closest_detection = None
        min_distance = float('inf')
        
        for detection in detections:
            if detection.is_close_to_point(click_x, click_y):
                center_x, center_y = detection.center
                distance = ((click_x - center_x) ** 2 + (click_y - center_y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_detection = detection
        
        return closest_detection
