from __future__ import annotations
import os, cv2, json, shutil, base64, tempfile, time
from datetime import datetime
from typing import Dict, Any, List
from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from .config import (
    FACE_DETECTION_CONFIDENCE_THRESHOLD, FACE_SIMILARITY_THRESHOLD, PROCESSING_DURATION_SECONDS,
    API_RESULTS_DIR, UPLOAD_DIR
)
from .storage.s3 import s3_client
from .config import USE_S3, S3_BUCKET_NAME, S3_REGION

router = APIRouter(prefix="/face-detection", tags=["face-detection"])

# 상태
face_detector = None
face_hashes: List[str] = []
face_bboxes: List[List[int]] = []

# 디렉토리 보장
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(API_RESULTS_DIR, exist_ok=True)

# --- 안전한 파일명 유틸 (길이/문자 제한) --------------------------------------
import re

def _safe_filename(original: str | None, *, prefix: str = "", default_ext: str = "", max_len: int = 120) -> str:
    """원본 파일명을 기반으로 안전하고 짧은 파일명을 생성.
    - 허용 문자만 남기고 불허 문자는 '_'로 치환
    - 확장자 유지(없으면 default_ext 사용)
    - prefix 포함 전체 길이를 max_len 이하로 절단
    """
    name = (original or "").strip()
    name = os.path.basename(name)
    stem, ext = os.path.splitext(name)
    if not ext and default_ext:
        ext = default_ext if default_ext.startswith(".") else f".{default_ext}"
    # 확장자 최대 10자, 허용 문자만
    ext = re.sub(r"[^A-Za-z0-9\.]+", "", ext)[:10]
    # 스템 정규화
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-") or "file"
    # 전체 길이 제한
    budget = max_len - len(prefix) - len(ext)
    if budget < 8:  # 너무 작은 경우 최소 확보
        budget = 8
    if len(stem) > budget:
        # 앞부분 대부분 + 끝쪽 일부 유지
        keep_head = max(4, int(budget * 0.75))
        keep_tail = budget - keep_head
        stem = f"{stem[:keep_head]}_{stem[-keep_tail:] if keep_tail>0 else ''}".rstrip("_")
    return f"{prefix}{stem}{ext}"

# --------------------------------------------------------------------------

def initialize_face_detector():  # pragma: no cover (heavy)
    global face_detector
    if face_detector is None:
        try:
            from ultralytics import YOLO  # type: ignore
            model_path = 'runs/face_detection/yolov8_face/weights/best.pt'
            if os.path.exists(model_path):
                face_detector = YOLO(model_path)
            else:
                face_detector = YOLO('yolov8n.pt')
        except Exception as e:
            raise RuntimeError(f"YOLO 모델 초기화 실패: {e}")
    return face_detector

def validate_video_file(filename: str) -> bool:
    return any(filename.lower().endswith(ext) for ext in ('.mp4','.avi','.mov','.mkv','.wmv','.flv'))

def calculate_image_hash(image):
    try:
        resized = cv2.resize(image, (8,8))
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        avg = gray.mean()
        return ''.join('1' if gray[i,j] > avg else '0' for i in range(8) for j in range(8))
    except Exception:
        return ''

def calculate_hash_similarity(h1: str, h2: str) -> float:
    if len(h1) != len(h2) or not h1:
        return 0.0
    dist = sum(c1!=c2 for c1,c2 in zip(h1,h2))
    return 1 - dist/len(h1)

def is_duplicate_face(face_hash: str, bbox: List[int]):
    if not face_hash or not face_hashes:
        return False
    for h in face_hashes:
        if calculate_hash_similarity(face_hash, h) > FACE_SIMILARITY_THRESHOLD:
            return True
    # 위치 기반 (단순 중심거리+면적비)
    if bbox:
        x1,y1,x2,y2 = bbox
        cx,cy = (x1+x2)//2,(y1+y2)//2
        area = max(1,(x2-x1)*(y2-y1))
        for ex1,ey1,ex2,ey2 in face_bboxes:
            ecx, ecy = (ex1+ex2)//2, (ey1+ey2)//2
            earea = max(1,(ex2-ex1)*(ey2-ey1))
            dist = ((cx-ecx)**2 + (cy-ecy)**2)**0.5
            ratio = min(area,earea)/max(area,earea)
            if dist < 30 and ratio > 0.9:
                return True
    return False

try:
    _HAAR_FRONTAL = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
except Exception:  # pragma: no cover
    _HAAR_FRONTAL = cv2.CascadeClassifier()
try:
    _HAAR_PROFILE = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
except Exception:  # pragma: no cover
    _HAAR_PROFILE = cv2.CascadeClassifier()

def _verify_face_region(img):
    try:
        if img is None or img.size == 0:
            return False
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        if not _HAAR_FRONTAL.empty():
            if len(_HAAR_FRONTAL.detectMultiScale(gray,1.1,3,minSize=(24,24))):
                return True
        if not _HAAR_PROFILE.empty():
            prof = _HAAR_PROFILE.detectMultiScale(gray,1.1,3,minSize=(24,24))
            if len(prof):
                return True
            gray_flip = cv2.flip(gray,1)
            if len(_HAAR_PROFILE.detectMultiScale(gray_flip,1.1,3,minSize=(24,24))):
                return True
        return False
    except Exception:
        return False

def _detect_faces_cascade(frame):
    boxes = []
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        if not _HAAR_FRONTAL.empty():
            for (x,y,w,h) in _HAAR_FRONTAL.detectMultiScale(gray,1.1,4,minSize=(32,32)):
                boxes.append((x,y,x+w,y+h))
        if not _HAAR_PROFILE.empty():
            for (x,y,w,h) in _HAAR_PROFILE.detectMultiScale(gray,1.1,4,minSize=(32,32)):
                boxes.append((x,y,x+w,y+h))
            gray_flip = cv2.flip(gray,1)
            w_img = frame.shape[1]
            for (x,y,w,h) in _HAAR_PROFILE.detectMultiScale(gray_flip,1.1,4,minSize=(32,32)):
                x1 = w_img - (x+w)
                boxes.append((x1,y,x1+w,y+h))
    except Exception:
        pass
    return boxes

def _iou(a,b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    if inter<=0: return 0.0
    area_a = max(0,ax2-ax1)*max(0,ay2-ay1)
    area_b = max(0,bx2-bx1)*max(0,by2-by1)
    return inter / (area_a + area_b - inter + 1e-9)

def _nms(boxes, thr=0.45):
    if not boxes: return []
    keep=[]
    boxes_sorted=sorted(boxes,key=lambda b:(b[2]-b[0])*(b[3]-b[1]),reverse=True)
    for b in boxes_sorted:
        if all(_iou(b,k) < thr for k in keep):
            keep.append(b)
    return keep

def upload_face_to_s3(image_path: str, s3_key: str) -> str:
    if not USE_S3 or not s3_client:
        return f"file://{os.path.abspath(image_path)}"
    try:
        s3_client.upload_file(image_path, S3_BUCKET_NAME, s3_key, ExtraArgs={'ContentType':'image/jpeg'})
        return f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{s3_key}"
    except Exception:
        return f"file://{os.path.abspath(image_path)}"

def detect_faces_at_time(video_path: str, start_minutes: int, start_seconds: int):
    global face_hashes, face_bboxes
    detector = initialize_face_detector()
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "비디오 파일을 열 수 없습니다"}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps>0 else 0
    start_time_seconds = start_minutes*60 + start_seconds
    start_frame = int(start_time_seconds * fps) if fps>0 else 0
    if start_time_seconds > duration:
        cap.release()
        return {"error": "입력 시간이 비디오 길이를 초과", "video_info": {"duration": duration, "total_frames": total_frames, "fps": fps}}
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_id = f"time_detection_{ts}"
    result_dir = os.path.join(API_RESULTS_DIR, result_id)
    faces_dir = os.path.join(result_dir, 'faces')
    os.makedirs(faces_dir, exist_ok=True)
    detected_faces = []
    face_hashes = []
    face_bboxes = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame
    # attempt to derive names
    try:
        detector_names = getattr(detector,'names', None)
    except Exception:
        detector_names = None
    def _is_face_class(result_obj, cls_id):
        try:
            names_map = getattr(result_obj,'names', None) or detector_names
            if isinstance(names_map, dict) and cls_id in names_map:
                return 'face' in str(names_map[cls_id]).lower()
            if isinstance(names_map,(list,tuple)) and 0 <= cls_id < len(names_map):
                return 'face' in str(names_map[cls_id]).lower()
        except Exception:
            pass
        return False
    def _box_valid_shape(x1,y1,x2,y2):
        w = max(0,x2-x1); h=max(0,y2-y1)
        if w < 24 or h < 24: return False
        ar = (w/h) if h>0 else 0
        return 0.6 <= ar <= 1.8
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            if fps>0 and frame_count % int(fps) == 0:
                candidate=[]; used_cascade=False
                if hasattr(detector,'predict'):
                    try:
                        results = detector(frame, verbose=False)
                        for result in results:
                            boxes = getattr(result,'boxes', None)
                            if boxes is None: continue
                            for box in boxes:
                                conf = float(box.conf[0]) if hasattr(box,'conf') else 1.0
                                if conf < FACE_DETECTION_CONFIDENCE_THRESHOLD: continue
                                cls_id = int(box.cls[0]) if hasattr(box,'cls') else -1
                                if not _is_face_class(result, cls_id):
                                    continue
                                x1,y1,x2,y2 = map(int, box.xyxy[0])
                                if not _box_valid_shape(x1,y1,x2,y2):
                                    continue
                                roi = frame[y1:y2, x1:x2]
                                if not _verify_face_region(roi):
                                    continue
                                candidate.append((x1,y1,x2,y2))
                    except Exception:
                        pass
                if not candidate:
                    used_cascade=True
                    candidate = _detect_faces_cascade(frame)
                final_boxes = _nms(candidate)
                for (x1,y1,x2,y2) in final_boxes:
                    face_img = frame[y1:y2, x1:x2]
                    if face_img is None or face_img.size==0: continue
                    if used_cascade and not _box_valid_shape(x1,y1,x2,y2):
                        continue
                    face_hash = calculate_image_hash(face_img)
                    if not face_hash: continue
                    bbox_coords=[x1,y1,x2,y2]
                    if not is_duplicate_face(face_hash, bbox_coords):
                        face_hashes.append(face_hash)
                        face_bboxes.append(bbox_coords)
                        idx = len(face_hashes)
                        face_filename = f"face_{idx:03d}_{ts}.jpg"
                        face_path = os.path.join(faces_dir, face_filename)
                        cv2.imwrite(face_path, face_img)
                        s3_key = f"api_results/{result_id}/faces/{face_filename}"
                        s3_url = upload_face_to_s3(face_path, s3_key)
                        detected_faces.append({"s3_url": s3_url})
            frame_count += 1
            if fps>0 and frame_count - start_frame > int(fps * PROCESSING_DURATION_SECONDS):
                break
    finally:
        cap.release()
    summary = {"faces": detected_faces}
    with open(os.path.join(result_dir, 'face_records.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    return summary

@router.get("")
async def root():
    return {"message": "시간 기반 얼굴 검출 API", "version": "1.0.0"}

@router.post('/upload-video')
async def upload_video(file: UploadFile = File(...)):
    if not validate_video_file(file.filename):
        raise HTTPException(status_code=400, detail='지원하지 않는 비디오 형식')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    # 원본 이름이 길어도 안전하게 축약
    safe_name = _safe_filename(file.filename, prefix=f"{ts}_", default_ext=".mp4", max_len=120)
    path = os.path.join(UPLOAD_DIR, safe_name)
    with open(path, 'wb') as bf:
        shutil.copyfileobj(file.file, bf)
    return {"message": "비디오 업로드 성공", "filename": safe_name, "file_path": path}

@router.post('/detect-faces')
async def detect_faces(
    filename: str = Query(None),
    time_input: str = Query(...),
    from_s3: bool = Query(False),
    video_url: str = Query(None),
    file: UploadFile = File(None)
):
    try:
        parts = time_input.split()
        if len(parts)==2:
            start_minutes, start_seconds = int(parts[0]), int(parts[1])
        elif len(parts)==1:
            total = int(parts[0]); start_minutes = total//60; start_seconds = total%60
        else:
            raise ValueError
    except ValueError:
        raise HTTPException(status_code=400, detail='시간 형식 오류')
    file_path=None; temp_created=False
    if video_url:
        if video_url.startswith('blob:'):
            if file is None:
                raise HTTPException(status_code=400, detail='blob: URL은 직접 업로드 필요')
            tmpdir = tempfile.gettempdir()
            ts = int(time.time())
            # 업로드된 파일 원본명을 안전하게 축약하여 임시 파일명 생성
            safe_tmp = _safe_filename(getattr(file, 'filename', None) or 'video', prefix=f"upload_{ts}_", default_ext=".mp4", max_len=120)
            temp_path = os.path.join(tmpdir, safe_tmp)
            with open(temp_path,'wb') as bf: shutil.copyfileobj(file.file, bf)
            file_path=temp_path; temp_created=True
        elif video_url.startswith('data:'):
            try:
                header,b64 = video_url.split(',',1)
                raw = base64.b64decode(b64)
                tmpdir = tempfile.gettempdir()
                temp_path = os.path.join(tmpdir, f"dataurl_{int(time.time())}.mp4")
                with open(temp_path,'wb') as ftmp: ftmp.write(raw)
                file_path=temp_path; temp_created=True
            except Exception as de:
                raise HTTPException(status_code=400, detail=f'data URL 파싱 실패: {de}')
        else:
            try:
                import requests
                tmp = os.path.join(tempfile.gettempdir(), f"url_dl_{int(time.time())}.mp4")
                with requests.get(video_url, stream=True, timeout=(5,60)) as r:
                    r.raise_for_status()
                    with open(tmp,'wb') as f:
                        for chunk in r.iter_content(chunk_size=1024*1024):
                            if chunk: f.write(chunk)
                file_path=tmp; temp_created=True
            except Exception as de:
                raise HTTPException(status_code=400, detail=f'URL 다운로드 실패: {de}')
    elif from_s3:
        if not USE_S3 or not s3_client:
            raise HTTPException(status_code=400, detail='S3 미설정')
        try:
            s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=f'recordings/{filename}')
            tmpdir = tempfile.gettempdir()
            safe_tmp = _safe_filename(filename or 'video', prefix="temp_", default_ext=".mp4", max_len=120)
            temp_file = os.path.join(tmpdir, safe_tmp)
            s3_client.download_file(S3_BUCKET_NAME, f'recordings/{filename}', temp_file)
            file_path=temp_file
        except Exception:
            raise HTTPException(status_code=404, detail='S3에서 파일 없음')
    else:
        if file is not None:
            tmpdir = tempfile.gettempdir()
            ts = int(time.time())
            safe_tmp = _safe_filename(getattr(file, 'filename', None) or 'video', prefix=f"upload_{ts}_", default_ext=".mp4", max_len=120)
            temp_path = os.path.join(tmpdir, safe_tmp)
            with open(temp_path,'wb') as bf: shutil.copyfileobj(file.file, bf)
            file_path=temp_path; temp_created=True
        elif not filename:
            raise HTTPException(status_code=400, detail='filename 또는 video_url 필요')
        else:
            path = os.path.join(UPLOAD_DIR, filename)
            if not os.path.exists(path):
                raise HTTPException(status_code=404, detail='업로드 비디오 없음')
            file_path=path
    result = detect_faces_at_time(file_path, start_minutes, start_seconds)
    if temp_created and file_path and os.path.exists(file_path):
        try: os.remove(file_path)
        except Exception: pass
    if 'error' in result:
        return JSONResponse(content=result, status_code=400)
    return result

@router.get('/video-info/{filename}')
async def video_info(filename: str):
    path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail='비디오 없음')
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail='비디오 열기 실패')
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    dur = total / fps if fps>0 else 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return {"filename": filename, "video_info": {"duration": dur, "total_frames": total, "fps": fps, "width": w, "height": h}}

@router.get('/results/{result_id}')
async def get_results(result_id: str):
    result_dir = os.path.join(API_RESULTS_DIR, result_id)
    summary_path = os.path.join(result_dir, 'face_records.json')
    if not os.path.exists(summary_path):
        raise HTTPException(status_code=404, detail='결과 없음')
    with open(summary_path,'r',encoding='utf-8') as f:
        return json.load(f)

@router.get('/download-face/{result_id}/{filename}')
async def download_face(result_id: str, filename: str):
    face_path = os.path.join(API_RESULTS_DIR, result_id, 'faces', filename)
    if not os.path.exists(face_path):
        raise HTTPException(status_code=404, detail='이미지 없음')
    return FileResponse(face_path, media_type='image/jpeg', filename=filename)

@router.get('/results')
async def list_results():
    out=[]
    if os.path.exists(API_RESULTS_DIR):
        for rid in os.listdir(API_RESULTS_DIR):
            summary_path = os.path.join(API_RESULTS_DIR, rid, 'face_records.json')
            if os.path.exists(summary_path):
                try:
                    with open(summary_path,'r',encoding='utf-8') as f: data=json.load(f)
                except Exception:
                    data={}
                out.append({"result_id": rid, "count": len(data.get('faces', []))})
    return {"results": out}

__all__ = ['router']
