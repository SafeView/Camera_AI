"""
녹화된 영상 처리 및 저장된 사람 인식 모듈
"""

import cv2
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
from .models import FaceDetection, PersonDetection, StoredPerson, RecognitionResult
from .face_detector import FaceDetector
from .person_detector import PersonDetector
from .embedding_processor import EmbeddingProcessor
from .dependencies import Dependencies


class VideoProcessor:
    """녹화된 영상 처리 및 저장된 사람 인식"""
    
    def __init__(self):
        self.deps = Dependencies()
        self.face_detector = FaceDetector(self.deps)
        self.person_detector = PersonDetector(self.deps)
        self.embedding_processor = EmbeddingProcessor(self.deps)
        
        # 인식 설정
        self.recognition_threshold = 0.8
        self.stored_persons: List[StoredPerson] = []
        
        # 결과 저장
        self.recognition_results: List[RecognitionResult] = []
        self.frame_results: Dict[int, List[RecognitionResult]] = {}
    
    def load_stored_persons(self, stored_persons: List[StoredPerson]):
        """저장된 사람 목록 로드"""
        self.stored_persons = stored_persons
        print(f"📁 {len(stored_persons)}명의 저장된 사람을 로드했습니다.")
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """녹화된 영상 처리 및 저장된 사람 인식"""
        print(f"🎬 영상 처리 시작: {video_path}")
        
        # 비디오 캡처 열기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 영상을 열 수 없습니다: {video_path}")
            return {}
        
        # 비디오 정보 가져오기
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 영상 정보: {total_frames} 프레임, {fps:.2f} FPS, {width}x{height}")
        
        # 출력 비디오 설정
        output_writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            output_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # 프레임별 처리
        frame_count = 0
        recognition_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 진행률 표시
            if frame_count % 30 == 0:  # 30프레임마다
                progress = (frame_count / total_frames) * 100
                print(f"📈 진행률: {progress:.1f}% ({frame_count}/{total_frames})")
            
            # 프레임 처리
            processed_frame, frame_results = self._process_frame(frame, frame_count)
            
            # 결과 저장
            if frame_results:
                self.frame_results[frame_count] = frame_results
                recognition_count += len(frame_results)
            
            # 출력 비디오에 프레임 저장
            if output_writer:
                output_writer.write(processed_frame)
        
        # 정리
        cap.release()
        if output_writer:
            output_writer.release()
        
        # 결과 요약
        summary = self._generate_summary(frame_count, recognition_count, fps)
        
        print(f"✅ 영상 처리 완료!")
        print(f"📊 총 {frame_count} 프레임 처리")
        print(f"👥 총 {recognition_count}회 인식")
        
        return summary
    
    def _process_frame(self, frame: np.ndarray, frame_number: int) -> Tuple[np.ndarray, List[RecognitionResult]]:
        """단일 프레임 처리"""
        # 얼굴과 사람 검출
        face_detections = self.face_detector.detect_faces(frame)
        person_detections = self.person_detector.detect_persons(frame)
        
        frame_results = []
        
        # 모든 얼굴에 대해 저장된 사람 인식 시도
        for face_detection in face_detections:
            recognition_result = self._recognize_person(face_detection, frame)
            if recognition_result:
                frame_results.append(recognition_result)
                self.recognition_results.append(recognition_result)
        
        # 처리된 프레임 그리기
        processed_frame = self._draw_recognition_results(frame, face_detections, person_detections, frame_results)
        
        return processed_frame, frame_results
    
    def _recognize_person(self, face_detection: FaceDetection, frame: np.ndarray) -> Optional[RecognitionResult]:
        """개별 사람 인식"""
        try:
            # 얼굴 ROI 추출
            x, y, w, h = face_detection.bbox
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return None
            
            # 얼굴 임베딩 계산
            embedding = self.embedding_processor.compute_face_embedding(frame, face_detection)
            if embedding is None:
                return None
            
            # 모든 저장된 사람들과 비교
            best_match = None
            best_similarity = 0.0
            
            for person in self.stored_persons:
                similarity = self.embedding_processor.cosine_similarity(embedding, person.embedding)
                
                if similarity > best_similarity and similarity >= self.recognition_threshold:
                    best_similarity = similarity
                    best_match = person
            
            if best_match:
                return RecognitionResult(
                    person=best_match,
                    similarity=best_similarity,
                    face_detection=face_detection
                )
            
            return None
            
        except Exception as e:
            return None
    
    def _draw_recognition_results(self, frame: np.ndarray, face_detections: List[FaceDetection], 
                                 person_detections: List[PersonDetection], 
                                 recognition_results: List[RecognitionResult]) -> np.ndarray:
        """인식 결과를 프레임에 그리기"""
        # 모든 검출된 얼굴에 회색 박스
        for detection in face_detections:
            x, y, w, h = detection.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (128, 128, 128), 1)
        
        # 모든 검출된 사람에 연한 파란색 박스
        for detection in person_detections:
            x, y, w, h = detection.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 128, 64), 1)
        
        # 인식된 사람들 표시
        for result in recognition_results:
            detection = result.face_detection
            x, y, w, h = detection.bbox
            
            # 인식된 사람은 초록색 박스
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            
            # 이름과 유사도 표시
            label = f"👤 {result.person.name} ({result.similarity:.3f})"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # 상태 정보 표시
        self._draw_status_info(frame, len(face_detections), len(person_detections), len(recognition_results))
        
        return frame
    
    def _draw_status_info(self, frame: np.ndarray, face_count: int, person_count: int, recognition_count: int):
        """상태 정보 표시"""
        # 검출 수
        cv2.putText(frame, f"Faces: {face_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Persons: {person_count}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 64), 2)
        
        # 인식 수
        cv2.putText(frame, f"Recognized: {recognition_count}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 저장된 사람 수
        cv2.putText(frame, f"Stored: {len(self.stored_persons)}", (10, 120),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def _generate_summary(self, total_frames: int, total_recognitions: int, fps: float) -> Dict[str, Any]:
        """처리 결과 요약 생성"""
        # 인식된 사람별 통계
        person_stats = {}
        for result in self.recognition_results:
            person_name = result.person.name
            if person_name not in person_stats:
                person_stats[person_name] = {
                    'count': 0,
                    'avg_similarity': 0.0,
                    'max_similarity': 0.0,
                    'frames': []
                }
            
            stats = person_stats[person_name]
            stats['count'] += 1
            stats['avg_similarity'] += result.similarity
            stats['max_similarity'] = max(stats['max_similarity'], result.similarity)
        
        # 평균 유사도 계산
        for person_name, stats in person_stats.items():
            stats['avg_similarity'] /= stats['count']
        
        return {
            'total_frames': total_frames,
            'total_recognitions': total_recognitions,
            'fps': fps,
            'duration_seconds': total_frames / fps if fps > 0 else 0,
            'person_statistics': person_stats,
            'recognition_rate': total_recognitions / total_frames if total_frames > 0 else 0
        }
    
    def get_recognition_results(self) -> List[RecognitionResult]:
        """인식 결과 반환"""
        return self.recognition_results
    
    def get_frame_results(self) -> Dict[int, List[RecognitionResult]]:
        """프레임별 인식 결과 반환"""
        return self.frame_results
    
    def set_recognition_threshold(self, threshold: float):
        """인식 임계값 설정"""
        self.recognition_threshold = threshold

