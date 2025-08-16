#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시간 기반 얼굴 검출 API
영상의 특정 분,초를 입력하면 사람의 얼굴이 나오는 부분을 이미지로 저장하여 출력 (얼굴중복은 저장하지 않음)
"""

import cv2
import numpy as np
import os
import json
import uuid
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import shutil
from ultralytics import YOLO
import hashlib
from PIL import Image
import io
import boto3
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

class S3Uploader:
    """S3 업로드 클래스"""
    
    def __init__(self):
        # .env 파일에서 환경 변수 로드
        load_dotenv()
        
        # 환경 변수에서 AWS 자격 증명 가져오기
        aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        aws_region = os.getenv('AWS_DEFAULT_REGION', 'ap-northeast-2')
        bucket_name = os.getenv('AWS_S3_BUCKET', 'gitpolio-images')
        
        if not aws_access_key_id or not aws_secret_access_key:
            raise ValueError("AWS 자격 증명이 .env 파일에 설정되지 않았습니다.")
        
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=aws_region
        )
        self.bucket_name = bucket_name
    
    def upload_image(self, image_path: str, s3_key: str) -> str:
        """이미지를 S3에 업로드하고 URL 반환"""
        try:
            # S3 업로드
            self.s3_client.upload_file(
                image_path,
                self.bucket_name,
                s3_key,
                ExtraArgs={'ContentType': 'image/jpeg'}
            )
            
            # S3 URL 생성
            s3_url = f"https://{self.bucket_name}.s3.ap-northeast-2.amazonaws.com/{s3_key}"
            print(f"✅ S3 업로드 성공: {s3_key}")
            return s3_url
            
        except Exception as e:
            print(f"❌ S3 업로드 실패: {e}")
            # S3 업로드 실패 시 로컬 파일 경로 반환
            return f"file://{os.path.abspath(image_path)}"

class TimeBasedFaceDetector:
    """시간 기반 얼굴 검출기"""
    
    def __init__(self):
        self.face_detector = None
        self.face_hashes = []  # 중복 제거를 위한 얼굴 해시 저장
        
        # 환경 변수에서 설정 가져오기
        load_dotenv()
        self.similarity_threshold = float(os.getenv('FACE_SIMILARITY_THRESHOLD', '0.95'))
        self.confidence_threshold = float(os.getenv('FACE_DETECTION_CONFIDENCE_THRESHOLD', '0.5'))
        self.processing_duration = int(os.getenv('PROCESSING_DURATION_SECONDS', '10'))
        
        self.s3_uploader = S3Uploader()
        
    def initialize_detector(self):
        """얼굴 검출기 초기화 - YOLO 모델 강제 사용"""
        if self.face_detector is None:
            try:
                import torch
                # PyTorch 2.6+ 호환성을 위한 설정
                torch.hub.set_dir('.')
                
                # 훈련된 얼굴 검출 모델 로드
                model_path = '../runs/face_detection/yolov8_face/weights/best.pt'
                if os.path.exists(model_path):
                    self.face_detector = YOLO(model_path)
                    print("✅ 훈련된 얼굴 검출 모델 로드 성공")
                else:
                    # 훈련된 모델이 없으면 기본 모델 사용
                    self.face_detector = YOLO('yolov8n.pt')
                    print("✅ YOLOv8n 기본 모델 로드 성공")
                
            except Exception as e:
                print(f"❌ YOLO 모델 로드 실패: {e}")
                print("PyTorch 2.6+ 호환성 문제로 인한 오류입니다.")
                print("weights_only=False로 설정하여 다시 시도합니다...")
                
                try:
                    import torch
                    # PyTorch 2.6+ 호환성을 위한 환경 변수 설정
                    import os
                    os.environ['TORCH_WEIGHTS_ONLY'] = 'False'
                    
                    # torch.load의 기본값을 False로 설정
                    import torch.serialization
                    original_load = torch.load
                    
                    def safe_load(*args, **kwargs):
                        kwargs['weights_only'] = False
                        return original_load(*args, **kwargs)
                    
                    torch.load = safe_load
                    
                    # 훈련된 얼굴 검출 모델 로드
                    model_path = '../runs/face_detection/yolov8_face/weights/best.pt'
                    if os.path.exists(model_path):
                        self.face_detector = YOLO(model_path)
                        print("✅ 훈련된 얼굴 검출 모델 로드 성공 (weights_only=False)")
                    else:
                        # 훈련된 모델이 없으면 기본 모델 사용
                        self.face_detector = YOLO('yolov8n.pt')
                        print("✅ YOLOv8n 기본 모델 로드 성공 (weights_only=False)")
                    
                except Exception as e2:
                    print(f"❌ YOLO 모델 로드 재시도 실패: {e2}")
                    raise Exception(f"YOLO 모델 초기화 실패: {e2}")
                    
        return self.face_detector
    
    def calculate_image_hash(self, image: np.ndarray) -> str:
        """이미지 해시 계산 (중복 제거용)"""
        try:
            # 이미지를 8x8 크기로 리사이즈
            resized = cv2.resize(image, (8, 8))
            # 그레이스케일로 변환
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            # 평균값 계산
            avg = gray.mean()
            # 해시 생성
            hash_str = ''
            for i in range(8):
                for j in range(8):
                    if gray[i, j] > avg:
                        hash_str += '1'
                    else:
                        hash_str += '0'
            return hash_str
        except Exception as e:
            print(f"해시 계산 실패: {e}")
            return None
    
    def calculate_hash_similarity(self, hash1: str, hash2: str) -> float:
        """해시 유사도 계산"""
        if len(hash1) != len(hash2):
            return 0.0
        
        # 해밍 거리 계산
        hamming_distance = sum(c1 != c2 for c1, c2 in zip(hash1, hash2))
        # 유사도 = 1 - (해밍 거리 / 해시 길이)
        similarity = 1 - (hamming_distance / len(hash1))
        return similarity
    
    def is_duplicate_face(self, face_hash: str, bbox: list = None) -> bool:
        """얼굴 중복 검사 (해시 + 위치 기반)"""
        if not face_hash or len(self.face_hashes) == 0:
            return False
        
        # 기존 얼굴들과 유사도 계산
        for existing_hash in self.face_hashes:
            similarity = self.calculate_hash_similarity(face_hash, existing_hash)
            if similarity > self.similarity_threshold:
                return True
        
        # 위치 기반 중복 검사 (같은 위치에 있는 얼굴은 중복으로 간주)
        if bbox:
            x1, y1, x2, y2 = bbox
            face_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            face_area = (x2 - x1) * (y2 - y1)
            
            # 기존 얼굴들과 위치 비교
            for existing_face in getattr(self, 'face_bboxes', []):
                ex1, ey1, ex2, ey2 = existing_face
                ex_center = ((ex1 + ex2) // 2, (ey1 + ey2) // 2)
                ex_area = (ex2 - ex1) * (ey2 - ey1)
                
                # 중심점 거리 계산
                center_distance = ((face_center[0] - ex_center[0]) ** 2 + 
                                 (face_center[1] - ex_center[1]) ** 2) ** 0.5
                
                # 면적 비율 계산
                area_ratio = min(face_area, ex_area) / max(face_area, ex_area)
                
                # 같은 위치에 비슷한 크기의 얼굴이면 중복으로 간주 (더 관대하게)
                if center_distance < 30 and area_ratio > 0.9:
                    return True
        
        return False
    
    def detect_faces_at_time(self, video_path: str, start_minutes: int, start_seconds: int) -> Dict:
        """특정 시간부터 얼굴 검출"""
        try:
            detector = self.initialize_detector()
            
            # 비디오 열기
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("비디오 파일을 열 수 없습니다")
            
            # 비디오 정보
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            # 시작 프레임 계산
            start_time_seconds = start_minutes * 60 + start_seconds
            start_frame = int(start_time_seconds * fps)
            
            # 입력 시간 검증
            if start_time_seconds > duration:
                cap.release()
                return {
                    "error": f"입력한 시간({start_minutes}분 {start_seconds}초)이 비디오 길이({duration:.1f}초)를 초과합니다.",
                    "video_info": {
                        "duration": duration,
                        "total_frames": total_frames,
                        "fps": fps
                    }
                }
            
            # 결과 저장 디렉토리 생성
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_id = f"time_detection_{timestamp}"
            result_dir = os.path.join("api_results", result_id)
            faces_dir = os.path.join(result_dir, "faces")
            os.makedirs(faces_dir, exist_ok=True)
            
            # 얼굴 검출 결과
            detected_faces = []
            unique_faces_count = 0
            total_faces_detected = 0
            self.face_bboxes = []  # 위치 기반 중복 제거를 위한 바운딩 박스 저장
            
            # 시작 프레임으로 이동
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frame_count = start_frame
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 1초마다 프레임 처리 (성능 최적화)
                if frame_count % int(fps) == 0:
                    # 얼굴 검출 (YOLOv8 또는 OpenCV Haar Cascade)
                    if hasattr(detector, 'predict'):  # YOLOv8 모델
                        results = detector(frame, verbose=False)
                        
                        for result in results:
                            boxes = result.boxes
                            if boxes is not None:
                                for box in boxes:
                                    # 신뢰도 확인
                                    confidence = float(box.conf[0])
                                    if confidence < self.confidence_threshold:  # 환경 변수에서 가져온 신뢰도 임계값
                                        continue
                                    
                                    # 바운딩 박스 좌표
                                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                                    
                                    # 얼굴 영역 추출
                                    face_image = frame[y1:y2, x1:x2]
                                    if face_image.size == 0:
                                        continue
                                    
                                    # 얼굴 해시 계산
                                    face_hash = self.calculate_image_hash(face_image)
                                    if face_hash is None:
                                        continue
                                    
                                    total_faces_detected += 1
                                    
                                    # 중복 검사
                                    bbox_coords = [int(x1), int(y1), int(x2), int(y2)]
                                    if not self.is_duplicate_face(face_hash, bbox_coords):
                                        # 새로운 얼굴이면 저장
                                        self.face_hashes.append(face_hash)
                                        self.face_bboxes.append(bbox_coords)
                                        unique_faces_count += 1
                                        
                                        # 얼굴 이미지 저장
                                        face_filename = f"face_{unique_faces_count:03d}_{timestamp}.jpg"
                                        face_path = os.path.join(faces_dir, face_filename)
                                        cv2.imwrite(face_path, face_image)
                                        
                                        # S3 업로드
                                        s3_key = f"api_results/{result_id}/faces/{face_filename}"
                                        s3_url = self.s3_uploader.upload_image(face_path, s3_key)
                                        
                                        # 검출 정보 저장
                                        current_time = frame_count / fps
                                        minutes = int(current_time // 60)
                                        seconds = int(current_time % 60)
                                        
                                        detected_faces.append({
                                            "face_id": unique_faces_count,
                                            "filename": face_filename,
                                            "detection_time": f"{minutes:02d}:{seconds:02d}",
                                            "confidence": float(confidence),
                                            "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                            "file_path": face_path,
                                            "s3_url": s3_url,
                                            "face_hash": face_hash[:16] + "..."  # 해시 일부만 표시
                                        })
                    
                    else:  # OpenCV Haar Cascade 모델
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        faces = detector.detectMultiScale(gray, 1.1, 4)
                        
                        for (x, y, w, h) in faces:
                            # 바운딩 박스 좌표
                            x1, y1, x2, y2 = x, y, x + w, y + h
                            
                            # 얼굴 영역 추출
                            face_image = frame[y1:y2, x1:x2]
                            if face_image.size == 0:
                                continue
                            
                            # 얼굴 해시 계산
                            face_hash = self.calculate_image_hash(face_image)
                            if face_hash is None:
                                continue
                            
                            total_faces_detected += 1
                            
                            # 중복 검사
                            bbox_coords = [int(x1), int(y1), int(x2), int(y2)]
                            if not self.is_duplicate_face(face_hash, bbox_coords):
                                # 새로운 얼굴이면 저장
                                self.face_hashes.append(face_hash)
                                self.face_bboxes.append(bbox_coords)
                                unique_faces_count += 1
                                
                                # 얼굴 이미지 저장
                                face_filename = f"face_{unique_faces_count:03d}_{timestamp}.jpg"
                                face_path = os.path.join(faces_dir, face_filename)
                                cv2.imwrite(face_path, face_image)
                                
                                # S3 업로드
                                s3_key = f"api_results/{result_id}/faces/{face_filename}"
                                s3_url = self.s3_uploader.upload_image(face_path, s3_key)
                                
                                # 검출 정보 저장
                                current_time = frame_count / fps
                                minutes = int(current_time // 60)
                                seconds = int(current_time % 60)
                                
                                detected_faces.append({
                                    "face_id": unique_faces_count,
                                    "filename": face_filename,
                                    "detection_time": f"{minutes:02d}:{seconds:02d}",
                                    "confidence": 0.8,  # Haar Cascade는 기본 신뢰도
                                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                                    "file_path": face_path,
                                    "s3_url": s3_url,
                                    "face_hash": face_hash[:16] + "..."  # 해시 일부만 표시
                                })
                
                frame_count += 1
                
                # 환경 변수에서 설정한 시간 후 중단 (성능 최적화)
                if frame_count - start_frame > int(fps * self.processing_duration):
                    break
            
            cap.release()
            
            # 결과 요약 저장
            summary = {
                "result_id": result_id,
                "detection_info": {
                    "start_time": f"{start_minutes:02d}:{start_seconds:02d}",
                    "total_faces_detected": total_faces_detected,
                    "unique_faces_saved": unique_faces_count,
                    "processing_duration": f"{self.processing_duration}초",
                    "faces_directory": faces_dir,
                    "duplicate_removal_method": "Image Hash Similarity"
                },
                "faces": detected_faces,
                "video_info": {
                    "duration": duration,
                    "total_frames": total_frames,
                    "fps": fps,
                    "start_frame": start_frame
                }
            }
            
            # 결과 JSON 저장
            summary_path = os.path.join(result_dir, "face_records.json")
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            return summary
            
        except Exception as e:
            raise Exception(f"얼굴 검출 중 오류 발생: {str(e)}")

# FastAPI 앱 생성
app = FastAPI(
    title="시간 기반 얼굴 검출 API",
    description="영상의 특정 분,초를 입력하면 사람의 얼굴이 나오는 부분을 이미지로 저장하여 출력 (얼굴중복은 저장하지 않음)",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 전역 변수
upload_dir = "uploads"
api_results_dir = "api_results"
face_detector = TimeBasedFaceDetector()

# 디렉토리 생성
os.makedirs(upload_dir, exist_ok=True)
os.makedirs(api_results_dir, exist_ok=True)

def validate_video_file(filename: str) -> bool:
    """비디오 파일 형식 검증"""
    allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv'}
    return any(filename.lower().endswith(ext) for ext in allowed_extensions)

def parse_time_input(time_input: str) -> Tuple[int, int]:
    """시간 입력 파싱"""
    try:
        parts = time_input.strip().split()
        if len(parts) == 1:
            # 초만 입력된 경우 (예: "90")
            total_seconds = int(parts[0])
            minutes = total_seconds // 60
            seconds = total_seconds % 60
        elif len(parts) == 2:
            # 분 초 입력된 경우 (예: "1 30")
            minutes = int(parts[0])
            seconds = int(parts[1])
        else:
            raise ValueError("잘못된 시간 형식")
        
        if minutes < 0 or seconds < 0 or seconds >= 60:
            raise ValueError("잘못된 시간 값")
        
        return minutes, seconds
    except (ValueError, IndexError):
        raise ValueError(f"잘못된 시간 형식: {time_input}")

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "message": "시간 기반 얼굴 검출 API",
        "version": "1.0.0",
        "description": "영상의 특정 분,초를 입력하면 사람의 얼굴이 나오는 부분을 이미지로 저장하여 출력 (얼굴중복은 저장하지 않음)",
        "endpoints": {
            "upload_video": "/upload-video",
            "detect_faces": "/detect-faces",
            "video_info": "/video-info/{filename}",
            "results": "/results/{result_id}",
            "download_face": "/download-face/{result_id}/{filename}"
        }
    }

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    """비디오 파일 업로드"""
    try:
        # 파일 검증
        if not validate_video_file(file.filename):
            raise HTTPException(
                status_code=400, 
                detail="지원하지 않는 비디오 형식입니다. MP4, AVI, MOV, MKV, WMV, FLV 파일만 지원합니다."
            )
        
        # 파일 저장
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}_{file.filename}"
        file_path = os.path.join(upload_dir, filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return {
            "message": "비디오 업로드 성공",
            "filename": filename,
            "file_path": file_path
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"업로드 오류: {str(e)}")

@app.post("/detect-faces")
async def detect_faces(
    filename: str = Query(..., description="업로드된 비디오 파일명"),
    time_input: str = Query(..., description="시작 시간 (예: '1 30' = 1분 30초부터, '90' = 90초부터)")
):
    """특정 시간부터 얼굴 검출"""
    try:
        file_path = os.path.join(upload_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, 
                detail="업로드된 비디오 파일을 찾을 수 없습니다"
            )
        
        # 시간 입력 파싱
        start_minutes, start_seconds = parse_time_input(time_input)
        
        # 얼굴 검출 실행
        result = face_detector.detect_faces_at_time(file_path, start_minutes, start_seconds)
        
        # 오류가 있는 경우
        if "error" in result:
            return JSONResponse(content=result, status_code=400)
        
        return result
        
    except ValueError as e:
        return JSONResponse(
            content={
                "error": f"잘못된 시간 형식입니다: {time_input}",
                "format_example": "올바른 형식: '1 30' (1분 30초) 또는 '90' (90초)"
            },
            status_code=400
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"얼굴 검출 오류: {str(e)}")

@app.get("/video-info/{filename}")
async def get_video_info(filename: str):
    """비디오 정보 조회"""
    try:
        file_path = os.path.join(upload_dir, filename)
        if not os.path.exists(file_path):
            raise HTTPException(
                status_code=404, 
                detail="업로드된 비디오 파일을 찾을 수 없습니다"
            )
        
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise HTTPException(
                status_code=400, 
                detail="비디오 파일을 열 수 없습니다"
            )
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        return {
            "filename": filename,
            "video_info": {
                "duration": duration,
                "total_frames": total_frames,
                "fps": fps,
                "width": width,
                "height": height,
                "duration_formatted": f"{int(duration//60)}분 {int(duration%60)}초"
            },
            "time_input_examples": [
                "0 0 - 전체 비디오",
                "0 30 - 30초부터",
                "1 0 - 1분부터",
                "1 30 - 1분 30초부터",
                "90 - 90초부터"
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"비디오 정보 조회 오류: {str(e)}")

@app.get("/results/{result_id}")
async def get_results(result_id: str):
    """검출 결과 조회"""
    try:
        result_dir = os.path.join(api_results_dir, result_id)
        if not os.path.exists(result_dir):
            raise HTTPException(
                status_code=404, 
                detail="검출 결과를 찾을 수 없습니다"
            )
        
        # 결과 JSON 파일 읽기
        summary_path = os.path.join(result_dir, "face_records.json")
        if not os.path.exists(summary_path):
            raise HTTPException(
                status_code=404, 
                detail="결과 파일을 찾을 수 없습니다"
            )
        
        with open(summary_path, 'r', encoding='utf-8') as f:
            result = json.load(f)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 조회 오류: {str(e)}")

@app.get("/download-face/{result_id}/{filename}")
async def download_face(result_id: str, filename: str):
    """얼굴 이미지 다운로드"""
    try:
        face_path = os.path.join(api_results_dir, result_id, "faces", filename)
        if not os.path.exists(face_path):
            raise HTTPException(
                status_code=404, 
                detail="얼굴 이미지를 찾을 수 없습니다"
            )
        
        return FileResponse(face_path, media_type="image/jpeg", filename=filename)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"이미지 다운로드 오류: {str(e)}")

@app.get("/results")
async def list_all_results():
    """모든 결과 목록 조회"""
    try:
        results = []
        if os.path.exists(api_results_dir):
            for result_id in os.listdir(api_results_dir):
                result_dir = os.path.join(api_results_dir, result_id)
                if os.path.isdir(result_dir):
                    summary_path = os.path.join(result_dir, "face_records.json")
                    if os.path.exists(summary_path):
                        with open(summary_path, 'r', encoding='utf-8') as f:
                            result_data = json.load(f)
                            results.append({
                                "result_id": result_id,
                                "detection_info": result_data.get("detection_info", {}),
                                "created_at": result_id.split("_")[-1] if "_" in result_id else result_id
                            })
        
        return {"results": results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"결과 목록 조회 오류: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # 환경 변수에서 호스트와 포트 가져오기
    load_dotenv()
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    print(f"🚀 API 서버 시작: http://{host}:{port}")
    print(f"📁 업로드 디렉토리: {upload_dir}")
    print(f"📊 결과 디렉토리: {api_results_dir}")
    print(f"🔧 설정값:")
    print(f"   - 얼굴 검출 신뢰도 임계값: {face_detector.confidence_threshold}")
    print(f"   - 얼굴 유사도 임계값: {face_detector.similarity_threshold}")
    print(f"   - 처리 시간: {face_detector.processing_duration}초")
    print(f"   - S3 버킷: {face_detector.s3_uploader.bucket_name}")
    
    uvicorn.run(app, host=host, port=port)
