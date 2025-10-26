"""
특정 사람 얼굴 인식 기반 시간 분석 모듈
얼굴 사진을 입력받아 해당 사람이 나오는 시간대만 추출
"""

import cv2
import numpy as np
import mediapipe as mp
import threading
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import tempfile
import os
from scipy.spatial.distance import cosine


class FaceRecognitionTimingAnalyzer:
    """특정 사람 얼굴 인식 기반 시간 분석 클래스"""
    
    def __init__(self):
        """초기화"""
        self._init_detectors()
        self.target_embedding = None
        
    def _init_detectors(self):
        """감지기 초기화"""
        try:
            # MediaPipe 얼굴 검출기
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.5
            )
            self._mp_face_lock = threading.Lock()
            
            # Haar Cascade 얼굴 검출기
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
            
            # HOG 사람 검출기
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            print("얼굴 인식 시간 분석 감지기 초기화 완료")
            
        except Exception as e:
            print(f"감지기 초기화 실패: {e}")
            self.face_detection = None
            self.face_cascade = None
            self.hog = None
    
    def set_target_face(self, face_image: np.ndarray) -> bool:
        """타겟 얼굴 설정 및 임베딩 생성"""
        try:
            # 얼굴 검출
            face_roi = self._extract_face_roi(face_image)
            if face_roi is None:
                return False
            
            # 임베딩 생성
            embedding = self._compute_face_embedding(face_roi)
            if embedding is None:
                return False
            
            self.target_embedding = embedding
            print("타겟 얼굴 임베딩 생성 완료")
            return True
            
        except Exception as e:
            print(f"타겟 얼굴 설정 실패: {e}")
            return False
    
    def analyze_video(self, video_path: str, similarity_threshold: float = 0.6) -> Dict[str, Any]:
        """동영상에서 타겟 사람이 나오는 시간대 분석"""
        if self.target_embedding is None:
            return {"error": "타겟 얼굴이 설정되지 않았습니다"}
        
        print(f"얼굴 인식 시간 분석 시작: {video_path}")
        
        # 비디오 정보 가져오기
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": f"비디오를 열 수 없습니다: {video_path}"}
        
        # 비디오 정보
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"비디오 정보: {total_frames} 프레임, {fps:.2f} FPS, {duration:.2f}초")
        
        # 분석 결과 저장
        person_timings = []
        current_segment = None
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # 성능을 위해 3프레임마다 1프레임만 분석
            if frame_count % 3 != 0:
                continue
            
            # 현재 시간 계산
            current_time = frame_count / fps
            
            # 타겟 사람 감지
            has_target_person = self._detect_target_person(frame, similarity_threshold)
            
            # 시간대 세그먼트 관리
            if has_target_person:
                if current_segment is None:
                    # 새로운 타겟 사람 출현 세그먼트 시작
                    current_segment = {
                        "start_time": current_time,
                        "end_time": current_time
                    }
                else:
                    # 기존 세그먼트 연장
                    current_segment["end_time"] = current_time
            else:
                if current_segment is not None:
                    # 타겟 사람이 사라짐 - 세그먼트 완료
                    person_timings.append(current_segment.copy())
                    current_segment = None
        
        # 마지막 세그먼트 처리
        if current_segment is not None:
            person_timings.append(current_segment)
        
        cap.release()
        
        # 연속된 시간대 통합
        merged_timings = self._merge_continuous_segments(person_timings)
        
        # 시간대 문자열로 변환
        time_ranges = []
        for segment in merged_timings:
            start_time = self._format_time(segment["start_time"])
            end_time = self._format_time(segment["end_time"])
            time_ranges.append(f"{start_time}~{end_time}")
        
        print(f"얼굴 인식 시간 분석 완료: {len(time_ranges)}개 시간대 (통합 전: {len(person_timings)}개)")
        
        return {
            "person_timings": time_ranges,
            "total_segments": len(time_ranges),
            "similarity_threshold": similarity_threshold
        }
    
    def _extract_face_roi(self, image: np.ndarray) -> Optional[np.ndarray]:
        """이미지에서 얼굴 ROI 추출"""
        try:
            # MediaPipe로 얼굴 검출
            with self._mp_face_lock:
                results = self.face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            if results.detections:
                for detection in results.detections:
                    if hasattr(detection, 'score') and len(detection.score) > 0:
                        confidence = detection.score[0]
                        if confidence > 0.5:
                            # 바운딩 박스 추출
                            bbox = detection.location_data.relative_bounding_box
                            h, w, _ = image.shape
                            
                            x = int(bbox.xmin * w)
                            y = int(bbox.ymin * h)
                            width = int(bbox.width * w)
                            height = int(bbox.height * h)
                            
                            # ROI 추출
                            roi = image[y:y+height, x:x+width]
                            if roi.size > 0:
                                return roi
            
            # Haar Cascade로 보완
            if self.face_cascade is not None:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30)
                )
                if len(faces) > 0:
                    x, y, w, h = faces[0]
                    roi = image[y:y+h, x:x+w]
                    if roi.size > 0:
                        return roi
            
            return None
            
        except Exception as e:
            return None
    
    def _compute_face_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 ROI에서 임베딩 생성 (간단한 히스토그램 기반)"""
        try:
            # 얼굴 크기 정규화
            face_roi = cv2.resize(face_roi, (128, 128))
            
            # 히스토그램 기반 임베딩 생성
            # RGB 각 채널의 히스토그램을 결합
            hist_r = cv2.calcHist([face_roi], [0], None, [32], [0, 256])
            hist_g = cv2.calcHist([face_roi], [1], None, [32], [0, 256])
            hist_b = cv2.calcHist([face_roi], [2], None, [32], [0, 256])
            
            # 히스토그램 정규화
            hist_r = cv2.normalize(hist_r, hist_r).flatten()
            hist_g = cv2.normalize(hist_g, hist_g).flatten()
            hist_b = cv2.normalize(hist_b, hist_b).flatten()
            
            # 결합
            embedding = np.concatenate([hist_r, hist_g, hist_b])
            
            # L2 정규화
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding.astype(np.float32)
            
        except Exception as e:
            return None
    
    def _detect_target_person(self, frame: np.ndarray, similarity_threshold: float) -> bool:
        """프레임에서 타겟 사람 감지"""
        if self.target_embedding is None:
            return False
        
        try:
            # 얼굴 검출
            face_roi = self._extract_face_roi(frame)
            if face_roi is None:
                return False
            
            # 임베딩 생성
            current_embedding = self._compute_face_embedding(face_roi)
            if current_embedding is None:
                return False
            
            # 유사도 계산
            similarity = 1 - cosine(self.target_embedding, current_embedding)
            
            return similarity >= similarity_threshold
            
        except Exception as e:
            return False
    
    def _merge_continuous_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """연속된 시간대 세그먼트를 통합"""
        if not segments:
            return []
        
        # 시작 시간으로 정렬
        sorted_segments = sorted(segments, key=lambda x: x["start_time"])
        merged = [sorted_segments[0]]
        
        for current in sorted_segments[1:]:
            last = merged[-1]
            
            # 현재 세그먼트가 이전 세그먼트와 연속되거나 겹치는 경우
            if current["start_time"] <= last["end_time"] + 1.0:  # 1초 이내 간격 허용
                # 끝 시간을 더 큰 값으로 업데이트
                last["end_time"] = max(last["end_time"], current["end_time"])
            else:
                # 연속되지 않으면 새로운 세그먼트로 추가
                merged.append(current)
        
        return merged
    
    def _format_time(self, seconds: float) -> str:
        """초를 MM:SS 형태로 변환"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


# 전역 분석기 인스턴스
face_recognition_analyzer = FaceRecognitionTimingAnalyzer()

# FastAPI 라우터 생성
router = APIRouter(prefix="/face-recognition-timing", tags=["Face Recognition Timing"])


@router.post("/analyze")
async def analyze_video_with_face(
    face_image: UploadFile = File(..., description="찾고자 하는 사람의 얼굴 사진"),
    video_file: UploadFile = File(..., description="분석할 동영상 파일")
):
    """얼굴 사진과 동영상을 입력받아 해당 사람이 나오는 시간대 분석"""
    
    # 파일 확장자 검증
    if not face_image.filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        raise HTTPException(
            status_code=400, 
            detail="얼굴 이미지는 JPG, JPEG, PNG, BMP 파일만 지원됩니다."
        )
    
    if not video_file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400, 
            detail="동영상은 MP4, AVI, MOV, MKV 파일만 지원됩니다."
        )
    
    # 고정된 유사도 임계값 설정
    similarity_threshold = 0.8
    
    # 임시 파일로 저장
    face_temp_path = None
    video_temp_path = None
    
    try:
        # 얼굴 이미지 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(face_image.filename)[1]) as temp_file:
            face_content = await face_image.read()
            temp_file.write(face_content)
            face_temp_path = temp_file.name
        
        # 동영상 파일 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(video_file.filename)[1]) as temp_file:
            video_content = await video_file.read()
            temp_file.write(video_content)
            video_temp_path = temp_file.name
        
        # 얼굴 이미지 로드 및 타겟 설정
        face_image_cv = cv2.imread(face_temp_path)
        if face_image_cv is None:
            raise HTTPException(status_code=400, detail="얼굴 이미지를 읽을 수 없습니다.")
        
        if not face_recognition_analyzer.set_target_face(face_image_cv):
            raise HTTPException(status_code=400, detail="얼굴 이미지에서 얼굴을 찾을 수 없습니다.")
        
        # 동영상 분석
        result = face_recognition_analyzer.analyze_video(video_temp_path, similarity_threshold)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        import traceback
        error_detail = f"분석 중 오류가 발생했습니다: {str(e)}"
        if not str(e):
            error_detail = f"분석 중 오류가 발생했습니다: {type(e).__name__}"
        print(f"API 오류: {error_detail}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=error_detail)
        
    finally:
        # 임시 파일 삭제
        if face_temp_path and os.path.exists(face_temp_path):
            os.unlink(face_temp_path)
        if video_temp_path and os.path.exists(video_temp_path):
            os.unlink(video_temp_path)

