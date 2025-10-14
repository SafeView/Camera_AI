# =====================================================================
# Module: AI_processor
# Purpose: 얼굴/머리 비식별(모자이크) 처리 및 다양한 검출 파이프라인 유틸리티 제공.
# Responsibilities:
#   - MediaPipe + Haar + Profile + HOG fallback 결합을 통한 얼굴/머리 영역 결정
#   - 프레임별 샘플링/캐싱을 통한 성능 최적화 (_frame_counter, *_last_*)
#   - 추적/영속 모자이크(head tracks) 유지로 깜빡임 감소
# Design Notes:
#   - 환경 변수 기반 튜닝(샘플 간격, 최소 픽셀, margin 등) -> 운영 중 동적 조정 용이
#   - 2차 검증(Haar/Profile)로 잘못된 MediaPipe 결과 필터링
#   - 성능 이슈 발생 시: (1) 해상도 축소 (2) 샘플 주기 증가 (3) Haar 검증 비활성화
# Extension Tips:
#   - GPU 가속 필요 시 MediaPipe 대신 ONNX/TensorRT 모델 교체
#   - 추적 품질 향상을 위해 Kalman Filter / ByteTrack 등 통합 가능
#   - 번호판 인식 후 민감정보 탐지 파이프라인 연계 가능
# =====================================================================

import os
import cv2
import numpy as np
import mediapipe as mp
import threading

# 얼굴 탐지 모델
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# 옆모습 검증용 프로파일 얼굴 분류기
try:
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
except Exception:
    profile_cascade = cv2.CascadeClassifier()

mp_face_detection = mp.solutions.face_detection
# 신뢰도 임계값을 0.7로 상향 조정 (더 엄격한 얼굴 검출)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)
_mp_face_lock = threading.Lock()

# --- Performance tuning (env-configurable) ---
AI_DETECT_MAX_WIDTH = int(os.getenv("AI_DETECT_MAX_WIDTH", "640"))   # downscale width for detection; 0 to disable
AI_FACE_SAMPLE_N = int(os.getenv("AI_FACE_SAMPLE_N", "3"))           # run face detection every N frames
# 기본값을 활성화로 변경하여 잘못된 검출을 줄임
AI_FACE_HAAR_VALIDATE = os.getenv("AI_FACE_HAAR_VALIDATE", "1") in ("1", "true", "True")
AI_PIXELATE_SCALE = int(os.getenv("AI_PIXELATE_SCALE", "12"))        # larger -> bigger blocks
AI_BOX_MARGIN = float(os.getenv("AI_BOX_MARGIN", "0.15"))            # expand boxes by ratio for safety
AI_FACE_MIN_SIZE_PX = int(os.getenv("AI_FACE_MIN_SIZE_PX", "16"))     # allow smaller faces
AI_FACE_CONF = float(os.getenv("AI_FACE_CONF", "0.6"))                # lower to catch small faces
AI_FACE_DETECT_AT_FULLRES_WHEN_EMPTY = os.getenv("AI_FACE_DETECT_AT_FULLRES_WHEN_EMPTY", "1") in ("1", "true", "True")
AI_FACE_SECOND_PASS_HAAR = os.getenv("AI_FACE_SECOND_PASS_HAAR", "1") in ("1", "true", "True")
AI_HAAR_MIN_NEIGHBORS = int(os.getenv("AI_HAAR_MIN_NEIGHBORS", "4"))
AI_HAAR_MIN_SIZE_PX = int(os.getenv("AI_HAAR_MIN_SIZE_PX", "20"))
AI_USE_HOG_PERSON_FALLBACK = os.getenv("AI_USE_HOG_PERSON_FALLBACK", "1") in ("1", "true", "True")
AI_HOG_HEAD_RATIO_TOP = float(os.getenv("AI_HOG_HEAD_RATIO_TOP", "0.4"))
AI_HEAD_STICKY_FRAMES = int(os.getenv("AI_HEAD_STICKY_FRAMES", "30"))  # persist head mosaic this many frames
AI_HEAD_IOU_MATCH = float(os.getenv("AI_HEAD_IOU_MATCH", "0.2"))       # IoU to associate detections to tracks
AI_HEAD_EXPAND_TOP_RATIO = float(os.getenv("AI_HEAD_EXPAND_TOP_RATIO", "0.25"))  # expand upward to include hair
AI_HEAD_EXPAND_X_RATIO = float(os.getenv("AI_HEAD_EXPAND_X_RATIO", "0.1"))   # expand left/right for side faces
# 광학 흐름 기반 경량 추적 옵션
AI_FLOW_ENABLED = os.getenv("AI_FLOW_ENABLED", "1") in ("1", "true", "True")
AI_FLOW_PROCESS_MAX_WIDTH = int(os.getenv("AI_FLOW_PROCESS_MAX_WIDTH", "480"))  # 흐름 계산용 축소 폭
AI_FLOW_FB_PYR_SCALE = float(os.getenv("AI_FLOW_FB_PYR_SCALE", "0.5"))
AI_FLOW_FB_LEVELS = int(os.getenv("AI_FLOW_FB_LEVELS", "3"))
AI_FLOW_FB_WINSIZE = int(os.getenv("AI_FLOW_FB_WINSIZE", "15"))
AI_FLOW_FB_ITERS = int(os.getenv("AI_FLOW_FB_ITERS", "3"))
AI_FLOW_FB_POLY_N = int(os.getenv("AI_FLOW_FB_POLY_N", "5"))
AI_FLOW_FB_POLY_SIGMA = float(os.getenv("AI_FLOW_FB_POLY_SIGMA", "1.2"))
AI_FLOW_BOX_SAMPLE_STEP = int(os.getenv("AI_FLOW_BOX_SAMPLE_STEP", "8"))  # ROI 내 샘플 간격(px, 축소 공간)

# Simple frame counter and cached boxes
_frame_counter = 0
_last_faces = []  # list of (x1,y1,x2,y2)
_hog = None  # lazy init for person fallback
_head_tracks = []  # list of {"box":(x1,y1,x2,y2), "ttl":int}
# 흐름 상태 (축소 그레이 프레임 캐시)
_prev_gray_small = None
_prev_gray_small_shape = None
_prev_flow_scale = 1.0

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(union) if union > 0 else 0.0

def _update_head_tracks(detected_boxes, decay: int = 1):
    """Update head tracks with new detections; maintain sticky TTL for persistence.
    decay: 프레임당 TTL 감소량. 탐지 수행 프레임에서는 1, 비탐지 프레임에서는 0으로 주어 트랙이 쉽게 사라지지 않게 한다.
    """
    global _head_tracks
    # Decrement existing TTL (조건부)
    if decay > 0:
        for tr in _head_tracks:
            tr["ttl"] -= decay
    # Associate detections to existing tracks via IoU
    assigned = set()
    for i, box in enumerate(detected_boxes):
        # find best match
        best_j, best_iou = -1, 0.0
        for j, tr in enumerate(_head_tracks):
            iou = _iou_xyxy(box, tr["box"])
            if iou > best_iou:
                best_iou, best_j = iou, j
        if best_iou >= AI_HEAD_IOU_MATCH and best_j >= 0:
            _head_tracks[best_j]["box"] = box
            _head_tracks[best_j]["ttl"] = AI_HEAD_STICKY_FRAMES
            assigned.add(i)
    # Add new tracks for unmatched detections
    for i, box in enumerate(detected_boxes):
        if i in assigned:
            continue
        _head_tracks.append({"box": box, "ttl": AI_HEAD_STICKY_FRAMES})
    # Remove expired
    _head_tracks = [tr for tr in _head_tracks if tr["ttl"] > 0]

def _nms_boxes(boxes, thr=0.5):
    if not boxes:
        return []
    # Simple greedy NMS by area (largest first)
    areas = [max(1, (x2-x1)) * max(1, (y2-y1)) for (x1,y1,x2,y2) in boxes]
    order = sorted(range(len(boxes)), key=lambda i: areas[i], reverse=True)
    keep = []
    used = [False]*len(boxes)
    for i in order:
        if used[i]:
            continue
        keep.append(boxes[i])
        used[i] = True
        for j in order:
            if used[j]:
                continue
            if _iou_xyxy(boxes[i], boxes[j]) > thr:
                used[j] = True
    return keep

def fast_pixelate(image, scale=AI_PIXELATE_SCALE):
    """Fast blocky pixelation by downscale-then-upscale."""
    (h, w) = image.shape[:2]
    if h == 0 or w == 0:
        return image
    sx = max(1, w // max(1, scale))
    sy = max(1, h // max(1, scale))
    small = cv2.resize(image, (sx, sy), interpolation=cv2.INTER_LINEAR)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

def is_valid_face_detection(bbox, frame_width, frame_height):
    """얼굴 검출 결과가 유효한지 검증하는 함수"""
    width = int(bbox.width * frame_width)
    height = int(bbox.height * frame_height)
    
    # 1. 최소/최대 크기 필터링 (너무 작거나 큰 검출 결과 제거)
    min_face_size = max(1, AI_FACE_MIN_SIZE_PX)
    max_face_size = min(frame_width, frame_height) * 0.8  # 프레임의 80% 이하
    
    if width < min_face_size or height < min_face_size:
        return False
    if width > max_face_size or height > max_face_size:
        return False
    
    # 2. 종횡비 필터링 (완화: 0.5~2.0)
    aspect_ratio = width / height if height > 0 else 0
    if aspect_ratio < 0.5 or aspect_ratio > 2.0:
        return False
    
    # 3. 프레임 경계 검사
    x1 = int(bbox.xmin * frame_width)
    y1 = int(bbox.ymin * frame_height)
    x2 = int((bbox.xmin + bbox.width) * frame_width)
    y2 = int((bbox.ymin + bbox.height) * frame_height)
    
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        return False
    
    return True

def _verify_face_roi(roi_bgr: np.ndarray) -> bool:
    """ROI에서 Haar(정면) 또는 Profile(옆면)로 얼굴이 실제 존재하는지 검증.
    좌우 반전(profile)까지 확인하여 옆모습도 인정한다.
    """
    try:
        if roi_bgr is None or roi_bgr.size == 0:
            return False
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        # 정면
        if not face_cascade.empty():
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
            if len(faces) > 0:
                return True
        # 옆면
        if not profile_cascade.empty():
            prof = profile_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
            if len(prof) > 0:
                return True
            # 반전해서 반대쪽 옆모습도 검사
            gray_flip = cv2.flip(gray, 1)
            prof2 = profile_cascade.detectMultiScale(gray_flip, scaleFactor=1.1, minNeighbors=4, minSize=(20, 20))
            if len(prof2) > 0:
                return True
    except Exception:
        return False
    return False

def _compute_small_gray(frame):
    """프레임을 광학 흐름 계산용으로 축소하고 그레이스케일 반환."""
    h, w = frame.shape[:2]
    if AI_FLOW_PROCESS_MAX_WIDTH and w > AI_FLOW_PROCESS_MAX_WIDTH:
        scale = AI_FLOW_PROCESS_MAX_WIDTH / float(w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale)))
    else:
        scale = 1.0
        small = frame
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray, scale

def _flow_update_tracks(curr_gray_small, scale_small):
    """Farneback 광학 흐름으로 기존 트랙 박스를 미세 이동시켜 유지.
    ROI 내 평균 이동 벡터를 구해 박스를 이동한다. 유효 이동이면 TTL을 소폭 회복시켜 깜빡임을 줄인다.
    """
    global _prev_gray_small, _prev_flow_scale, _head_tracks
    if _prev_gray_small is None or _prev_gray_small.shape != curr_gray_small.shape:
        _prev_gray_small = curr_gray_small.copy()
        _prev_flow_scale = scale_small
        return
    try:
        flow = cv2.calcOpticalFlowFarneback(
            _prev_gray_small, curr_gray_small,
            None,
            pyr_scale=AI_FLOW_FB_PYR_SCALE,
            levels=AI_FLOW_FB_LEVELS,
            winsize=AI_FLOW_FB_WINSIZE,
            iterations=AI_FLOW_FB_ITERS,
            poly_n=AI_FLOW_FB_POLY_N,
            poly_sigma=AI_FLOW_FB_POLY_SIGMA,
            flags=0
        )  # shape: (H,W,2)
        H, W = flow.shape[:2]
        moved = False
        for tr in _head_tracks:
            x1, y1, x2, y2 = tr["box"]
            # 현재 축소 스케일 기준 좌표로 변환 (이전 프레임과 스케일 동일 가정)
            sx1 = int(x1 * scale_small)
            sy1 = int(y1 * scale_small)
            sx2 = int(x2 * scale_small)
            sy2 = int(y2 * scale_small)
            # 유효 ROI 검사
            sx1 = max(0, min(W - 1, sx1)); sy1 = max(0, min(H - 1, sy1))
            sx2 = max(0, min(W, sx2)); sy2 = max(0, min(H, sy2))
            if sx2 - sx1 < 4 or sy2 - sy1 < 4:
                continue
            # 샘플 간격으로 그리드 포인트 선택 후 평균 이동
            step = max(2, AI_FLOW_BOX_SAMPLE_STEP)
            region = flow[sy1:sy2:step, sx1:sx2:step]
            if region.size == 0:
                continue
            dx = float(np.mean(region[..., 0]))
            dy = float(np.mean(region[..., 1]))
            # 너무 큰 점프는 노이즈로 간주하고 무시
            max_jump = 0.1 * max(W, H)
            if abs(dx) > max_jump or abs(dy) > max_jump:
                continue
            # 원본 좌표계로 환산하여 이동 적용
            if scale_small > 0:
                mx = int(round(dx / scale_small))
                my = int(round(dy / scale_small))
            else:
                mx = int(round(dx))
                my = int(round(dy))
            nx1 = max(0, min(curr_gray_small.shape[1] if scale_small==1.0 else int(W/scale_small), x1 + mx))
            ny1 = max(0, min(curr_gray_small.shape[0] if scale_small==1.0 else int(H/scale_small), y1 + my))
            nx2 = max(nx1 + 1, min(curr_gray_small.shape[1] if scale_small==1.0 else int(W/scale_small), x2 + mx))
            ny2 = max(ny1 + 1, min(curr_gray_small.shape[0] if scale_small==1.0 else int(H/scale_small), y2 + my))
            tr["box"] = (nx1, ny1, nx2, ny2)
            # 유효 이동으로 판단되면 TTL을 소폭 회복 (최대치 제한)
            tr["ttl"] = min(AI_HEAD_STICKY_FRAMES, tr["ttl"] + 1)
            moved = True
        # 상태 업데이트
        _prev_gray_small = curr_gray_small.copy()
        _prev_flow_scale = scale_small
    except Exception:
        _prev_gray_small = curr_gray_small.copy()
        _prev_flow_scale = scale_small
        return

def detect_and_blur(frame, blur_face=True):
    global _frame_counter, _last_faces
    _frame_counter += 1
    h, w = frame.shape[:2]
    # Downscale for detection
    det_frame = frame
    scale_det = 1.0
    if AI_DETECT_MAX_WIDTH and AI_DETECT_MAX_WIDTH > 0 and w > AI_DETECT_MAX_WIDTH:
        scale_det = AI_DETECT_MAX_WIDTH / float(w)
        det_frame = cv2.resize(frame, (int(w * scale_det), int(h * scale_det)))
    # Flow prep (경량 추적용)
    flow_gray_small, flow_scale = (None, 1.0)
    if AI_FLOW_ENABLED:
        flow_gray_small, flow_scale = _compute_small_gray(frame)

    # Face/head detection sampling
    do_face = blur_face and ((_frame_counter % max(1, AI_FACE_SAMPLE_N)) == 0)
    faces_to_use = []
    detected_boxes = []  # fresh detections this cycle
    if blur_face:
        if do_face:
            with _mp_face_lock:
                results = face_detection.process(cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB))
            new_faces = []
            if results.detections:
                dh, dw = det_frame.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    if not is_valid_face_detection(bbox, dw, dh):
                        continue
                    if hasattr(detection, 'score') and len(detection.score) > 0:
                        confidence = detection.score[0]
                        if confidence < AI_FACE_CONF:
                            continue
                    x1 = max(0, int(bbox.xmin * dw))
                    y1 = max(0, int(bbox.ymin * dh))
                    x2 = min(dw, int((bbox.xmin + bbox.width) * dw))
                    y2 = min(dh, int((bbox.ymin + bbox.height) * dh))
                    # ROI Haar/Profile 검증 (잘못된 검출 제거)
                    if AI_FACE_HAAR_VALIDATE:
                        roi = det_frame[y1:y2, x1:x2]
                        if roi.size == 0 or not _verify_face_roi(roi):
                            continue
                    # scale back to original
                    if scale_det != 1.0:
                        x1 = int(x1 / scale_det)
                        y1 = int(y1 / scale_det)
                        x2 = int(x2 / scale_det)
                        y2 = int(y2 / scale_det)
                    # expand margin
                    mx = int((x2 - x1) * AI_BOX_MARGIN)
                    my = int((y2 - y1) * AI_BOX_MARGIN)
                    x1 = max(0, x1 - mx)
                    y1 = max(0, y1 - my)
                    x2 = min(w, x2 + mx)
                    y2 = min(h, y2 + my)
                    new_faces.append((x1, y1, x2, y2))
            # optional Haar validation (costly)
            if AI_FACE_HAAR_VALIDATE and new_faces:
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                haar_faces = face_cascade.detectMultiScale(
                    gray_full, scaleFactor=1.1, minNeighbors=AI_HAAR_MIN_NEIGHBORS, minSize=(AI_HAAR_MIN_SIZE_PX, AI_HAAR_MIN_SIZE_PX), maxSize=(int(w*0.6), int(h*0.6))
                )
                validated = []
                for (x1, y1, x2, y2) in new_faces:
                    for (hx, hy, hw, hh) in haar_faces:
                        overlap_x1 = max(x1, hx)
                        overlap_y1 = max(y1, hy)
                        overlap_x2 = min(x2, hx + hw)
                        overlap_y2 = min(y2, hy + hh)
                        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                            validated.append((x1, y1, x2, y2))
                            break
                _last_faces = validated if validated else new_faces
            else:
                _last_faces = new_faces
            # If nothing found and allowed, try full-res second pass
            if not _last_faces and AI_FACE_DETECT_AT_FULLRES_WHEN_EMPTY and scale_det != 1.0:
                with _mp_face_lock:
                    results_full = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                new_faces2 = []
                if results_full.detections:
                    for detection in results_full.detections:
                        bbox = detection.location_data.relative_bounding_box
                        if not is_valid_face_detection(bbox, w, h):
                            continue
                        if hasattr(detection, 'score') and len(detection.score) > 0:
                            confidence = detection.score[0]
                            if confidence < AI_FACE_CONF:
                                continue
                        x1 = max(0, int(bbox.xmin * w))
                        y1 = max(0, int(bbox.ymin * h))
                        x2 = min(w, int((bbox.xmin + bbox.width) * w))
                        y2 = min(h, int((bbox.ymin + bbox.height) * h))
                        if AI_FACE_HAAR_VALIDATE:
                            roi = frame[y1:y2, x1:x2]
                            if roi.size == 0 or not _verify_face_roi(roi):
                                continue
                        mx = int((x2 - x1) * AI_BOX_MARGIN)
                        my = int((y2 - y1) * AI_BOX_MARGIN)
                        x1 = max(0, x1 - mx)
                        y1 = max(0, y1 - my)
                        x2 = min(w, x2 + mx)
                        y2 = min(h, y2 + my)
                        new_faces2.append((x1, y1, x2, y2))
                if new_faces2:
                    _last_faces = new_faces2
            # If still nothing, try Haar-only fallback
            if not _last_faces and AI_FACE_SECOND_PASS_HAAR:
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                haar_faces = face_cascade.detectMultiScale(
                    gray_full, scaleFactor=1.1, minNeighbors=AI_HAAR_MIN_NEIGHBORS, minSize=(AI_HAAR_MIN_SIZE_PX, AI_HAAR_MIN_SIZE_PX), maxSize=(int(w*0.8), int(h*0.8))
                )
                new_faces3 = []
                for (hx, hy, hw, hh) in haar_faces:
                    mx = int(hw * AI_BOX_MARGIN)
                    my = int(hh * AI_BOX_MARGIN)
                    x1 = max(0, hx - mx)
                    y1 = max(0, hy - my)
                    x2 = min(w, hx + hw + mx)
                    y2 = min(h, hy + hh + my)
                    new_faces3.append((x1, y1, x2, y2))
                if new_faces3:
                    _last_faces = new_faces3
        # HOG person fallback (mosaic head region)
        if not _last_faces and AI_USE_HOG_PERSON_FALLBACK:
            global _hog
            if _hog is None:
                _hog = cv2.HOGDescriptor()
                _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            rects, _ = _hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            head_boxes = []
            for (px, py, pw, ph) in rects:
                head_h = int(ph * AI_HOG_HEAD_RATIO_TOP)
                x1 = max(0, px)
                y1 = max(0, py)
                x2 = min(w, px + pw)
                y2 = min(h, py + head_h)
                if (x2 - x1) >= AI_FACE_MIN_SIZE_PX and (y2 - y1) >= AI_FACE_MIN_SIZE_PX:
                    head_boxes.append((x1, y1, x2, y2))
            if head_boxes:
                _last_faces = head_boxes
        # collect detections (if any) and update sticky tracks
        detected_boxes = list(_last_faces) if do_face else []
        _update_head_tracks(detected_boxes, decay=(1 if do_face else 0))
        # 광학 흐름으로 트랙 유지/추적
        if AI_FLOW_ENABLED and flow_gray_small is not None:
            _flow_update_tracks(flow_gray_small, flow_scale)
        faces_to_use = [tr["box"] for tr in _head_tracks]
        faces_to_use = _nms_boxes(faces_to_use, thr=0.5)

    # Apply mosaic
    if faces_to_use:
        for (x1, y1, x2, y2) in faces_to_use:
            # expand upward to cover hair/side head
            h_box = y2 - y1
            expand_top = int(h_box * AI_HEAD_EXPAND_TOP_RATIO)
            y1e = max(0, y1 - expand_top)
            # expand left/right a bit to catch side profile drift
            w_box = x2 - x1
            expand_x = int(w_box * AI_HEAD_EXPAND_X_RATIO)
            x1e = max(0, x1 - expand_x)
            x2e = min(frame.shape[1], x2 + expand_x)
            roi = frame[y1e:y2, x1e:x2e]
            if roi.size > 0:
                frame[y1e:y2, x1e:x2e] = fast_pixelate(roi.copy(), scale=AI_PIXELATE_SCALE)
    return frame

def enhanced_face_detection(frame):
    """향상된 얼굴 검출 - MediaPipe와 Haar Cascade 조합"""
    faces_to_blur = []
    h, w, _ = frame.shape
    
    # 1. MediaPipe 얼굴 검출
    with _mp_face_lock:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.detections:
        for detection in results.detections:
            bbox = detection.location_data.relative_bounding_box
            
            # 검출 결과 유효성 검증
            if not is_valid_face_detection(bbox, w, h):
                continue
            
            # 신뢰도 검증
            if hasattr(detection, 'score') and len(detection.score) > 0:
                confidence = detection.score[0]
                if confidence < 0.8:
                    continue
            
            x1 = max(0, int(bbox.xmin * w))
            y1 = max(0, int(bbox.ymin * h))
            x2 = min(w, int((bbox.xmin + bbox.width) * w))
            y2 = min(h, int((bbox.ymin + bbox.height) * h))
            
            faces_to_blur.append((x1, y1, x2, y2, 'mediapipe'))
    
    # 2. Haar Cascade로 추가 검증 (더 엄격한 파라미터)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    haar_faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=6,  # 더 엄격하게 (기본 3에서 6으로)
        minSize=(40, 40),  # 최소 크기
        maxSize=(int(w*0.6), int(h*0.6))  # 최대 크기 제한
    )
    
    # 두 검출 결과가 겹치는 영역만 최종 승인
    validated_faces = []
    for (x1, y1, x2, y2, source) in faces_to_blur:
        if source == 'mediapipe':
            # MediaPipe 결과를 Haar Cascade로 검증
            face_found = False
            for (hx, hy, hw, hh) in haar_faces:
                # 겹치는 영역이 있는지 확인 (IoU 방식)
                overlap_x1 = max(x1, hx)
                overlap_y1 = max(y1, hy)
                overlap_x2 = min(x2, hx + hw)
                overlap_y2 = min(y2, hy + hh)
                
                if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                    overlap_area = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
                    mediapipe_area = (x2 - x1) * (y2 - y1)
                    haar_area = hw * hh
                    
                    # 겹치는 영역이 충분히 크면 유효한 얼굴로 인정
                    if overlap_area > 0.3 * min(mediapipe_area, haar_area):
                        face_found = True
                        break
            
            if face_found:
                validated_faces.append((x1, y1, x2, y2))
    
    return validated_faces

# 다양한 모자이크 옵션을 위한 함수 예시
def process_frame(frame, mode="face"):
    # plate 모드는 제거됨. 어떤 모드든 얼굴 모자이크만 수행.
    return detect_and_blur(frame, blur_face=True)

def detect_and_blur_meta(frame, blur_face=True):
    """detect_and_blur의 메타 반환 버전. (processed_frame, {faces, faces_fresh})를 반환한다."""
    global _frame_counter, _last_faces
    _frame_counter += 1
    h, w = frame.shape[:2]
    det_frame = frame
    scale_det = 1.0
    if AI_DETECT_MAX_WIDTH and AI_DETECT_MAX_WIDTH > 0 and w > AI_DETECT_MAX_WIDTH:
        scale_det = AI_DETECT_MAX_WIDTH / float(w)
        det_frame = cv2.resize(frame, (int(w * scale_det), int(h * scale_det)))
    flow_gray_small, flow_scale = (None, 1.0)
    if AI_FLOW_ENABLED:
        flow_gray_small, flow_scale = _compute_small_gray(frame)

    do_face = blur_face and ((_frame_counter % max(1, AI_FACE_SAMPLE_N)) == 0)
    faces_to_use = []
    detected_boxes = []
    if blur_face:
        if do_face:
            with _mp_face_lock:
                results = face_detection.process(cv2.cvtColor(det_frame, cv2.COLOR_BGR2RGB))
            new_faces = []
            if results.detections:
                dh, dw = det_frame.shape[:2]
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    if not is_valid_face_detection(bbox, dw, dh):
                        continue
                    if hasattr(detection, 'score') and len(detection.score) > 0:
                        confidence = detection.score[0]
                        if confidence < AI_FACE_CONF:
                            continue
                    x1 = max(0, int(bbox.xmin * dw)); y1 = max(0, int(bbox.ymin * dh))
                    x2 = min(dw, int((bbox.xmin + bbox.width) * dw)); y2 = min(dh, int((bbox.ymin + bbox.height) * dh))
                    # ROI Haar/Profile 검증 (잘못된 검출 제거)
                    if AI_FACE_HAAR_VALIDATE:
                        roi = det_frame[y1:y2, x1:x2]
                        if roi.size == 0 or not _verify_face_roi(roi):
                            continue
                    # scale back to original
                    if scale_det != 1.0:
                        x1 = int(x1 / scale_det); y1 = int(y1 / scale_det)
                        x2 = int(x2 / scale_det); y2 = int(y2 / scale_det)
                    # expand margin
                    mx = int((x2 - x1) * AI_BOX_MARGIN); my = int((y2 - y1) * AI_BOX_MARGIN)
                    x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
                    x2 = min(w, x2 + mx); y2 = min(h, y2 + my)
                    new_faces.append((x1, y1, x2, y2))
            # optional Haar validation (costly)
            if AI_FACE_HAAR_VALIDATE and new_faces:
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                haar_faces = face_cascade.detectMultiScale(
                    gray_full, scaleFactor=1.1, minNeighbors=AI_HAAR_MIN_NEIGHBORS, minSize=(AI_HAAR_MIN_SIZE_PX, AI_HAAR_MIN_SIZE_PX), maxSize=(int(w*0.6), int(h*0.6))
                )
                validated = []
                for (x1, y1, x2, y2) in new_faces:
                    for (hx, hy, hw, hh) in haar_faces:
                        overlap_x1 = max(x1, hx); overlap_y1 = max(y1, hy)
                        overlap_x2 = min(x2, hx + hw); overlap_y2 = min(y2, hy + hh)
                        if overlap_x1 < overlap_x2 and overlap_y1 < overlap_y2:
                            validated.append((x1, y1, x2, y2)); break
                _last_faces = validated if validated else new_faces
            else:
                _last_faces = new_faces
            # If nothing found and allowed, try full-res second pass
            if not _last_faces and AI_FACE_DETECT_AT_FULLRES_WHEN_EMPTY and scale_det != 1.0:
                with _mp_face_lock:
                    results_full = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                new_faces2 = []
                if results_full.detections:
                    for detection in results_full.detections:
                        bbox = detection.location_data.relative_bounding_box
                        if not is_valid_face_detection(bbox, w, h):
                            continue
                        if hasattr(detection, 'score') and len(detection.score) > 0:
                            confidence = detection.score[0]
                            if confidence < AI_FACE_CONF:
                                continue
                        x1 = max(0, int(bbox.xmin * w)); y1 = max(0, int(bbox.ymin * h))
                        x2 = min(w, int((bbox.xmin + bbox.width) * w)); y2 = min(h, int((bbox.ymin + bbox.height) * h))
                        if AI_FACE_HAAR_VALIDATE:
                            roi = frame[y1:y2, x1:x2]
                            if roi.size == 0 or not _verify_face_roi(roi):
                                continue
                        mx = int((x2 - x1) * AI_BOX_MARGIN); my = int((y2 - y1) * AI_BOX_MARGIN)
                        x1 = max(0, x1 - mx); y1 = max(0, y1 - my)
                        x2 = min(w, x2 + mx); y2 = min(h, y2 + my)
                        new_faces2.append((x1, y1, x2, y2))
                if new_faces2:
                    _last_faces = new_faces2
            # If still nothing, try Haar-only fallback
            if not _last_faces and AI_FACE_SECOND_PASS_HAAR:
                gray_full = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                haar_faces = face_cascade.detectMultiScale(
                    gray_full, scaleFactor=1.1, minNeighbors=AI_HAAR_MIN_NEIGHBORS, minSize=(AI_HAAR_MIN_SIZE_PX, AI_HAAR_MIN_SIZE_PX), maxSize=(int(w*0.8), int(h*0.8))
                )
                new_faces3 = []
                for (hx, hy, hw, hh) in haar_faces:
                    mx = int(hw * AI_BOX_MARGIN); my = int(hh * AI_BOX_MARGIN)
                    x1 = max(0, hx - mx); y1 = max(0, hy - my)
                    x2 = min(w, hx + hw + mx); y2 = min(h, hy + hh + my)
                    new_faces3.append((x1, y1, x2, y2))
                if new_faces3:
                    _last_faces = new_faces3
        if not _last_faces and AI_USE_HOG_PERSON_FALLBACK:
            global _hog
            if _hog is None:
                _hog = cv2.HOGDescriptor(); _hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            rects, _ = _hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
            head_boxes = []
            for (px, py, pw, ph) in rects:
                head_h = int(ph * AI_HOG_HEAD_RATIO_TOP)
                x1 = max(0, px); y1 = max(0, py)
                x2 = min(w, px + pw); y2 = min(h, py + head_h)
                if (x2 - x1) >= AI_FACE_MIN_SIZE_PX and (y2 - y1) >= AI_FACE_MIN_SIZE_PX:
                    head_boxes.append((x1, y1, x2, y2))
            if head_boxes:
                _last_faces = head_boxes
        detected_boxes = list(_last_faces) if do_face else []
        _update_head_tracks(detected_boxes, decay=(1 if do_face else 0))
        if AI_FLOW_ENABLED and flow_gray_small is not None:
            _flow_update_tracks(flow_gray_small, flow_scale)
        faces_to_use = [tr["box"] for tr in _head_tracks]
        faces_to_use = _nms_boxes(faces_to_use, thr=0.5)

    if faces_to_use:
        for (x1, y1, x2, y2) in faces_to_use:
            h_box = y2 - y1
            expand_top = int(h_box * AI_HEAD_EXPAND_TOP_RATIO)
            y1e = max(0, y1 - expand_top)
            w_box = x2 - x1
            expand_x = int(w_box * AI_HEAD_EXPAND_X_RATIO)
            x1e = max(0, x1 - expand_x)
            x2e = min(frame.shape[1], x2 + expand_x)
            roi = frame[y1e:y2, x1e:x2e]
            if roi.size > 0:
                frame[y1e:y2, x1e:x2e] = fast_pixelate(roi.copy(), scale=AI_PIXELATE_SCALE)
    faces_count = len(faces_to_use)
    faces_fresh = len(detected_boxes) if do_face else 0
    return frame, {"faces": faces_count, "faces_fresh": faces_fresh}


def process_frame_with_meta(frame, mode="face"):
    """모드에 관계없이 얼굴 모자이크 + 메타(얼굴 수, 신선 검출 수)만 반환한다.
    반환: (processed_frame, {faces, faces_fresh})
    """
    return detect_and_blur_meta(frame, blur_face=True)
