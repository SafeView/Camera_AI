"""
사람 출현 시간 분석 모듈
기존 서버에 통합된 버전
"""

import cv2
import numpy as np
import mediapipe as mp
import threading
from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
import tempfile
import os


class PersonTimingAnalyzer:
    """사람 출현 시간 분석 클래스"""
    
    def __init__(self):
        """초기화"""
        self._init_detectors()
        
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
            
            print("사람 출현 시간 분석 감지기 초기화 완료")
            
        except Exception as e:
            print(f"감지기 초기화 실패: {e}")
            self.face_detection = None
            self.face_cascade = None
            self.hog = None
    
    def analyze_video(self, video_path: str) -> Dict[str, Any]:
        """동영상에서 사람이 나오는 시간대 분석"""
        print(f"사람 출현 시간 분석 시작: {video_path}")
        
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
            
            # 사람 감지
            has_person = self._detect_person(frame)
            
            # 시간대 세그먼트 관리
            if has_person:
                if current_segment is None:
                    # 새로운 사람 출현 세그먼트 시작
                    current_segment = {
                        "start_time": current_time,
                        "end_time": current_time
                    }
                else:
                    # 기존 세그먼트 연장
                    current_segment["end_time"] = current_time
            else:
                if current_segment is not None:
                    # 사람이 사라짐 - 세그먼트 완료
                    person_timings.append(current_segment.copy())
                    current_segment = None
        
        # 마지막 세그먼트 처리
        if current_segment is not None:
            person_timings.append(current_segment)
        
        cap.release()
        
        # 시간대 문자열로 변환
        time_ranges = []
        for segment in person_timings:
            start_time = self._format_time(segment["start_time"])
            end_time = self._format_time(segment["end_time"])
            time_ranges.append(f"{start_time}~{end_time}")
        
        print(f"사람 출현 시간 분석 완료: {len(time_ranges)}개 시간대")
        
        return {
            "person_timings": time_ranges,
            "total_segments": len(time_ranges)
        }
    
    def _detect_person(self, frame: np.ndarray) -> bool:
        """사람 감지 (간단한 방식)"""
        if self.face_detection is None:
            return False
        
        try:
            h, w, _ = frame.shape
            
            # 성능 최적화: 큰 영상은 축소
            if w > 640:
                scale = 640 / float(w)
                small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                small_frame = frame
            
            # MediaPipe 얼굴 검출
            with self._mp_face_lock:
                results = self.face_detection.process(cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB))
            
            if results.detections:
                for detection in results.detections:
                    if hasattr(detection, 'score') and len(detection.score) > 0:
                        confidence = detection.score[0]
                        if confidence > 0.5:
                            return True
            
            # Haar Cascade 얼굴 검출 (보완)
            if self.face_cascade is not None:
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30)
                )
                if len(faces) > 0:
                    return True
            
            # HOG 사람 검출 (보완)
            if self.hog is not None:
                (rects, weights) = self.hog.detectMultiScale(
                    small_frame,
                    winStride=(8, 8),
                    padding=(8, 8),
                    scale=1.05,
                    hitThreshold=0.3
                )
                if len(rects) > 0:
                    return True
            
            return False
            
        except Exception as e:
            return False
    
    def _format_time(self, seconds: float) -> str:
        """초를 MM:SS 형태로 변환"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


# 전역 분석기 인스턴스
person_timing_analyzer = PersonTimingAnalyzer()

# FastAPI 라우터 생성
router = APIRouter(prefix="/person-timing", tags=["Person Timing"])


@router.post("/analyze")
async def analyze_video(file: UploadFile = File(...)):
    """동영상 파일 업로드 및 사람 출현 시간 분석"""
    
    # 파일 확장자 검증
    if not file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        raise HTTPException(
            status_code=400, 
            detail="지원하지 않는 파일 형식입니다. MP4, AVI, MOV, MKV 파일만 지원됩니다."
        )
    
    # 임시 파일로 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    
    try:
        # 분석 실행
        result = person_timing_analyzer.analyze_video(temp_file_path)
        
        # 임시 파일 삭제
        os.unlink(temp_file_path)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return JSONResponse(content=result)
        
    except Exception as e:
        # 임시 파일 삭제
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
        
        raise HTTPException(status_code=500, detail=f"분석 중 오류가 발생했습니다: {str(e)}")


