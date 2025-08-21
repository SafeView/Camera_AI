"""
메인 얼굴 추적 시스템 모듈
"""

import cv2
import numpy as np
from datetime import datetime
from typing import Optional, List, Tuple
from .models import (
    FaceDetection, PersonDetection, TrackingTarget, TrackingState, 
    TrackingResult, TrackingConfig
)
from .face_detector import FaceDetector
from .person_detector import PersonDetector
from .embedding_processor import EmbeddingProcessor
from .pinecone_manager import PineconeManager
from .server_manager import ServerManager
from .dependencies import Dependencies

class FaceTracker:
    """메인 얼굴 추적 시스템"""
    
    def __init__(self, dependencies: Dependencies):
        # 의존성 및 컴포넌트 초기화
        self.deps = dependencies
        self.face_detector = FaceDetector(dependencies)
        self.person_detector = PersonDetector(dependencies)
        self.embedding_processor = EmbeddingProcessor(dependencies)
        self.pinecone_manager = PineconeManager(dependencies)
        self.server_manager = ServerManager()
        
        # 추적 상태
        self.current_target: Optional[TrackingTarget] = None
        self.tracking_state = TrackingState.NO_TARGET
        self.person_counter = 0
        
        # 추적 설정
        self.config = self._get_tracking_config()
        
        # 내부 상태
        self._lost_frame_count = 0
        self._target_suspended = False
    
    def _get_tracking_config(self) -> TrackingConfig:
        """추적 설정 가져오기"""
        if self.deps.insight_available:
            return TrackingConfig.for_insight_face()
        else:
            return TrackingConfig.for_histogram()
    
    def start_tracking(self, frame: np.ndarray, click_point: Tuple[int, int]) -> bool:
        """클릭 지점에서 추적 시작"""
        # 사람 전체 검출
        person_detections = self.person_detector.detect_persons(frame)
        face_detections = self.face_detector.detect_faces(frame)
        
        # 가장 가까운 사람 찾기
        closest_person = self.person_detector.find_closest_person(person_detections, click_point)
        
        # 사람이 없으면 얼굴로 폴백
        if not closest_person:
            closest_face = self.face_detector.find_closest_face(face_detections, click_point)
            if not closest_face:
                return False
            
            # 얼굴 기반 임베딩 계산
            embedding = self.embedding_processor.compute_face_embedding(frame, closest_face)
            if embedding is None:
                return False
            
            # 타겟 생성 (얼굴만)
            self.person_counter += 1
            target = TrackingTarget(
                name=f"Person_{self.person_counter}",
                embedding=embedding,
                last_face_detection=closest_face
            )
            
        else:
            # 사람과 얼굴 연결
            closest_person = self.person_detector.associate_face_with_person(closest_person, face_detections)
            
            # 얼굴이 있으면 얼굴 기반 임베딩, 없으면 사람 전체 히스토그램
            if closest_person.has_face():
                embedding = self.embedding_processor.compute_face_embedding(frame, closest_person.face_detection)
            else:
                # 사람 전체 영역을 사용한 히스토그램 임베딩
                px, py, pw, ph = closest_person.bbox
                person_roi = frame[py:py+ph, px:px+pw]
                embedding = self._compute_person_histogram(person_roi)
            
            if embedding is None:
                return False
            
            # 타겟 생성 (사람 전체)
            self.person_counter += 1
            target = TrackingTarget(
                name=f"Person_{self.person_counter}",
                embedding=embedding,
                last_face_detection=closest_person.face_detection,
                last_person_detection=closest_person
            )
        
        # 추적 시작
        self.current_target = target
        self.tracking_state = TrackingState.TRACKING
        self._reset_tracking_state()
        
        # 외부 저장
        self._save_target_data(target, frame)
        
        return True
    
    def _compute_person_histogram(self, person_roi: np.ndarray) -> Optional[np.ndarray]:
        """사람 전체 영역의 개선된 임베딩 계산"""
        try:
            if person_roi.size == 0:
                print("❌ 전신 ROI가 비어있음")
                return None
            
            # ROI 크기 확인
            if person_roi.shape[0] < 10 or person_roi.shape[1] < 10:
                print(f"❌ 전신 ROI가 너무 작음: {person_roi.shape}")
                return None
            
            print(f"🔍 전신 ROI 크기: {person_roi.shape}")
            
            # 개선된 사람 임베딩 계산
            result = self.embedding_processor.compute_person_embedding(person_roi)
            
            if result is not None:
                print(f"✅ 전신 임베딩 생성 성공: {result.shape}")
            else:
                print("❌ 전신 임베딩 생성 실패")
            
            return result
            
        except Exception as e:
            print(f"❌ 전신 임베딩 생성 오류: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def update_tracking(self, frame: np.ndarray) -> TrackingResult:
        """추적 업데이트"""
        if not self.current_target:
            return TrackingResult(TrackingState.NO_TARGET)
        
        # 얼굴과 사람 검출
        face_detections = self.face_detector.detect_faces(frame)
        person_detections = self.person_detector.detect_persons(frame)
        
        # 타겟 매칭 시도 (얼굴 우선, 사람 전체 폴백)
        best_face, face_similarity = self._find_best_face_match(face_detections, frame)
        best_person, person_similarity = self._find_best_person_match(person_detections, frame)
        
        # 가장 좋은 매칭 선택
        if best_face and face_similarity > person_similarity:
            if self._is_face_tracking_valid(best_face, face_similarity):
                self._handle_successful_face_tracking(best_face)
                return TrackingResult(
                    state=TrackingState.TRACKING,
                    face_detection=best_face,
                    similarity=face_similarity,
                    target_name=self.current_target.name
                )
        elif best_person and self._is_person_tracking_valid(best_person, person_similarity):
            # 사람과 얼굴 연결
            best_person = self.person_detector.associate_face_with_person(best_person, face_detections)
            self._handle_successful_person_tracking(best_person)
            return TrackingResult(
                state=TrackingState.TRACKING,
                face_detection=best_person.face_detection,
                person_detection=best_person,
                similarity=person_similarity,
                target_name=self.current_target.name
            )
        
        # 추적 실패
        state = self._handle_tracking_loss()
        return TrackingResult(
            state=state,
            face_detection=self.current_target.last_face_detection,
            person_detection=self.current_target.last_person_detection,
            similarity=max(face_similarity, person_similarity),
            target_name=self.current_target.name
        )
    
    def _find_best_face_match(self, face_detections: List[FaceDetection], frame: np.ndarray) -> Tuple[Optional[FaceDetection], float]:
        """현재 타겟과 가장 유사한 얼굴 찾기"""
        if not face_detections or not self.current_target:
            return None, 0.0
        
        best_detection = None
        best_similarity = 0.0
        
        for detection in face_detections:
            # 임베딩 계산
            embedding = self.embedding_processor.compute_face_embedding(frame, detection)
            if embedding is None:
                continue
            
            # 유사도 계산
            similarity = EmbeddingProcessor.cosine_similarity(
                self.current_target.embedding, embedding
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_detection = detection
        
        return best_detection, best_similarity
    
    def _find_best_person_match(self, person_detections: List[PersonDetection], frame: np.ndarray) -> Tuple[Optional[PersonDetection], float]:
        """현재 타겟과 가장 유사한 사람 전체 찾기"""
        if not person_detections or not self.current_target:
            return None, 0.0
        
        best_detection = None
        best_similarity = 0.0
        
        for detection in person_detections:
            # 사람 전체 히스토그램 계산
            px, py, pw, ph = detection.bbox
            person_roi = frame[py:py+ph, px:px+pw]
            embedding = self._compute_person_histogram(person_roi)
            
            if embedding is None:
                continue
            
            # 유사도 계산
            similarity = EmbeddingProcessor.cosine_similarity(
                self.current_target.embedding, embedding
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_detection = detection
        
        return best_detection, best_similarity
    
    def _is_face_tracking_valid(self, detection: FaceDetection, similarity: float) -> bool:
        """얼굴 추적 유효성 검사"""
        # 유사도 검사
        if similarity < self.config.strict_threshold:
            return False
        
        # 위치 검증
        if self.current_target.last_face_detection:
            if not self._is_position_valid(detection, self.current_target.last_face_detection):
                print(f"⚠️ 얼굴 타겟이 너무 멀리 이동했습니다.")
                return False
        
        return True
    
    def _is_person_tracking_valid(self, detection: PersonDetection, similarity: float) -> bool:
        """사람 전체 추적 유효성 검사"""
        # 유사도 검사 (사람 전체는 조금 낮은 임계값 사용)
        threshold = self.config.strict_threshold * 0.8
        if similarity < threshold:
            return False
        
        # 위치 검증
        if self.current_target.last_person_detection:
            if not self._is_position_valid(detection, self.current_target.last_person_detection):
                return False
        
        return True
    
    def _is_position_valid(self, current: FaceDetection, previous: FaceDetection) -> bool:
        """위치 유효성 검사"""
        distance = current.distance_to(previous)
        max_movement = max(previous.w, previous.h) * self.config.max_movement_multiplier
        return distance <= max_movement
    
    def _handle_successful_face_tracking(self, detection: FaceDetection):
        """얼굴 추적 성공 처리"""
        self.current_target.update_face_detection(detection)
        self.tracking_state = TrackingState.TRACKING
        self._reset_tracking_state()
    
    def _handle_successful_person_tracking(self, detection: PersonDetection):
        """사람 전체 추적 성공 처리"""
        self.current_target.update_person_detection(detection)
        if detection.has_face():
            self.current_target.update_face_detection(detection.face_detection)
        self.tracking_state = TrackingState.TRACKING
        self._reset_tracking_state()
    
    def _handle_tracking_loss(self) -> TrackingState:
        """추적 실패 처리"""
        self._lost_frame_count += 1
        
        if self._lost_frame_count > self.config.lost_frame_threshold:
            if not self._target_suspended:
                self._target_suspended = True
            return TrackingState.SUSPENDED
        else:
            return TrackingState.SEARCHING
    
    def _reset_tracking_state(self):
        """추적 상태 리셋"""
        self._lost_frame_count = 0
        if self._target_suspended:
            self._target_suspended = False
    
    def _save_target_data(self, target: TrackingTarget, frame: np.ndarray):
        """타겟 데이터 저장"""
        # 얼굴과 사람 전체 이미지 모두 저장
        if target.last_face_detection:
            # 얼굴 이미지 저장
            face_detection = target.last_face_detection
            if self.pinecone_manager.is_available():
                self.pinecone_manager.save_embedding(target, face_detection, frame.shape[:2])
            
            fx, fy, fw, fh = face_detection.bbox
            face_image = frame[fy:fy+fh, fx:fx+fw]
            self.server_manager.add_target(target, face_image)
        
        # 사람 전체 이미지도 저장 (있는 경우)
        if target.last_person_detection:
            px, py, pw, ph = target.last_person_detection.bbox
            person_image = frame[py:py+ph, px:px+pw]
            
            # 사람 전체 이미지를 별도로 저장
            try:
                import tempfile
                import os
                temp_filename = f"temp_person_{target.name}_{datetime.now().strftime('%H%M%S')}.jpg"
                cv2.imwrite(temp_filename, person_image)
                
                # 임시 파일 정리
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
            except Exception as e:
                pass
    
    def clear_target(self):
        """타겟 완전 해제"""
        self.current_target = None
        self.tracking_state = TrackingState.NO_TARGET
        self._reset_tracking_state()
        self._target_suspended = False
    
    def suspend_tracking(self):
        """추적 일시 중지"""
        if self.current_target:
            self._target_suspended = True
            self.tracking_state = TrackingState.SUSPENDED
    
    def get_current_target(self) -> Optional[TrackingTarget]:
        """현재 추적 중인 타겟 반환"""
        return self.current_target
    
    def get_tracking_state(self) -> TrackingState:
        """현재 추적 상태 반환"""
        return self.tracking_state
    
    def is_tracking_active(self) -> bool:
        """추적이 활성 상태인지 확인"""
        return self.current_target is not None
    
    def get_statistics(self) -> dict:
        """추적 통계 반환"""
        return {
            "current_target": self.current_target.name if self.current_target else None,
            "tracking_state": self.tracking_state.value,
            "total_targets_created": self.person_counter,
            "lost_frame_count": self._lost_frame_count,
            "target_suspended": self._target_suspended
        }
