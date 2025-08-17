"""
군중 환경 최적화 모듈
"""

import cv2
import numpy as np
from typing import List, Tuple
from .models import FaceDetection, PersonDetection
from .config import Config

class CrowdOptimizer:
    """군중 환경에서의 검출 및 추적 최적화"""
    
    def __init__(self):
        self.frame_history = []
        self.detection_history = []
        self.max_history = 5
        
    def optimize_for_crowd(self, frame: np.ndarray, 
                          face_detections: List[FaceDetection], 
                          person_detections: List[PersonDetection]) -> Tuple[List[FaceDetection], List[PersonDetection]]:
        """군중 환경에 최적화된 검출 결과 반환"""
        
        # 프레임 품질 향상
        enhanced_frame = self._enhance_frame_quality(frame)
        
        # 시간적 일관성 적용
        stable_faces = self._apply_temporal_consistency(face_detections, "face")
        stable_persons = self._apply_temporal_consistency(person_detections, "person")
        
        # 군중 밀도 분석
        crowd_density = self._analyze_crowd_density(stable_persons, frame.shape[:2])
        
        # 밀도에 따른 파라미터 조정
        if crowd_density > 0.3:  # 고밀도 군중
            stable_faces = self._high_density_optimization(stable_faces, frame.shape[:2])
            stable_persons = self._high_density_optimization(stable_persons, frame.shape[:2])
        
        # 기록 업데이트
        self._update_history(frame, stable_faces, stable_persons)
        
        return stable_faces, stable_persons
    
    def _enhance_frame_quality(self, frame: np.ndarray) -> np.ndarray:
        """프레임 품질 향상"""
        # 대비 개선
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        enhanced = cv2.merge([l, a, b])
        enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        
        # 가우시안 블러로 노이즈 제거
        enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced
    
    def _apply_temporal_consistency(self, detections: List, detection_type: str) -> List:
        """시간적 일관성을 적용하여 안정적인 검출"""
        if len(self.detection_history) < 2:
            return detections
        
        stable_detections = []
        
        for detection in detections:
            # 과거 프레임에서 유사한 위치의 검출이 있었는지 확인
            consistency_score = self._calculate_consistency_score(detection, detection_type)
            
            # 일관성 점수가 높으면 유지
            if consistency_score > 0.3:
                stable_detections.append(detection)
        
        return stable_detections
    
    def _calculate_consistency_score(self, detection, detection_type: str) -> float:
        """검출의 일관성 점수 계산"""
        if len(self.detection_history) == 0:
            return 1.0
        
        max_score = 0.0
        
        for historical_detections in self.detection_history:
            if detection_type not in historical_detections:
                continue
            
            for hist_detection in historical_detections[detection_type]:
                # IoU 기반 유사도
                iou = self._calculate_detection_iou(detection, hist_detection)
                
                # 크기 유사도
                size_similarity = self._calculate_size_similarity(detection, hist_detection)
                
                # 종합 점수
                score = iou * 0.7 + size_similarity * 0.3
                max_score = max(max_score, score)
        
        return max_score
    
    def _calculate_detection_iou(self, det1, det2) -> float:
        """두 검출 간의 IoU 계산"""
        x1, y1, w1, h1 = det1.bbox
        x2, y2, w2, h2 = det2.bbox
        
        # 교집합 계산
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        union_area = w1 * h1 + w2 * h2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def _calculate_size_similarity(self, det1, det2) -> float:
        """크기 유사도 계산"""
        area1 = det1.w * det1.h
        area2 = det2.w * det2.h
        
        if area1 == 0 or area2 == 0:
            return 0.0
        
        ratio = min(area1, area2) / max(area1, area2)
        return ratio
    
    def _analyze_crowd_density(self, person_detections: List[PersonDetection], frame_shape: Tuple[int, int]) -> float:
        """군중 밀도 분석"""
        if not person_detections:
            return 0.0
        
        frame_area = frame_shape[0] * frame_shape[1]
        total_person_area = sum(det.area for det in person_detections)
        
        # 사람이 차지하는 면적 비율
        density = total_person_area / frame_area
        
        # 사람 수도 고려
        person_count_factor = min(len(person_detections) / 10.0, 1.0)
        
        return density * 0.7 + person_count_factor * 0.3
    
    def _high_density_optimization(self, detections: List, frame_shape: Tuple[int, int]) -> List:
        """고밀도 환경 최적화"""
        if not detections:
            return detections
        
        # 너무 작은 검출 제거 (고밀도에서는 부정확할 가능성 높음)
        min_area = frame_shape[0] * frame_shape[1] * 0.005  # 프레임의 0.5%
        
        filtered = []
        for detection in detections:
            if detection.area >= min_area:
                filtered.append(detection)
        
        # 신뢰도가 높은 검출만 유지
        if hasattr(detections[0], 'confidence'):
            high_confidence = [d for d in filtered if d.confidence > 0.6]
            return high_confidence if high_confidence else filtered[:5]  # 최대 5개
        
        return filtered[:8]  # 최대 8개
    
    def _update_history(self, frame: np.ndarray, faces: List[FaceDetection], persons: List[PersonDetection]):
        """기록 업데이트"""
        # 프레임 기록
        if len(self.frame_history) >= self.max_history:
            self.frame_history.pop(0)
        self.frame_history.append(frame.copy())
        
        # 검출 기록
        if len(self.detection_history) >= self.max_history:
            self.detection_history.pop(0)
        
        self.detection_history.append({
            "face": faces.copy(),
            "person": persons.copy()
        })
    
    def get_crowd_statistics(self) -> dict:
        """군중 통계 반환"""
        if not self.detection_history:
            return {"avg_faces": 0, "avg_persons": 0, "crowd_density": 0.0}
        
        recent_detections = self.detection_history[-3:]  # 최근 3프레임
        
        avg_faces = sum(len(d.get("face", [])) for d in recent_detections) / len(recent_detections)
        avg_persons = sum(len(d.get("person", [])) for d in recent_detections) / len(recent_detections)
        
        return {
            "avg_faces": round(avg_faces, 1),
            "avg_persons": round(avg_persons, 1),
            "crowd_density": round(self._analyze_crowd_density(
                recent_detections[-1].get("person", []) if recent_detections else [],
                (480, 640)  # 기본 해상도
            ), 2)
        }
