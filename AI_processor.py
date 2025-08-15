import cv2
import numpy as np
import mediapipe as mp

# 얼굴 탐지 모델
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# 번호판 탐지 모델 (예시: haarcascade_russian_plate_number.xml)
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

mp_face_detection = mp.solutions.face_detection
# 신뢰도 임계값을 0.7로 상향 조정 (더 엄격한 얼굴 검출)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.7)

def anonymize_pixelate(image, blocks=15):
    (h, w) = image.shape[:2]
    xSteps = np.linspace(0, w, blocks + 1, dtype="int")
    ySteps = np.linspace(0, h, blocks + 1, dtype="int")
    for i in range(1, len(ySteps)):
        for j in range(1, len(xSteps)):
            startX = xSteps[j - 1]
            startY = ySteps[i - 1]
            endX = xSteps[j]
            endY = ySteps[i]
            roi = image[startY:endY, startX:endX]
            (B, G, R) = [int(x) for x in cv2.mean(roi)[:3]]
            cv2.rectangle(image, (startX, startY), (endX, endY), (B, G, R), -1)
    return image

def is_valid_face_detection(bbox, frame_width, frame_height):
    """얼굴 검출 결과가 유효한지 검증하는 함수"""
    width = int(bbox.width * frame_width)
    height = int(bbox.height * frame_height)
    
    # 1. 최소/최대 크기 필터링 (너무 작거나 큰 검출 결과 제거)
    min_face_size = 30  # 최소 30픽셀
    max_face_size = min(frame_width, frame_height) * 0.7  # 프레임의 70% 이하
    
    if width < min_face_size or height < min_face_size:
        return False
    if width > max_face_size or height > max_face_size:
        return False
    
    # 2. 종횡비 필터링 (얼굴의 일반적인 비율: 0.7~1.5)
    aspect_ratio = width / height if height > 0 else 0
    if aspect_ratio < 0.6 or aspect_ratio > 1.6:
        return False
    
    # 3. 프레임 경계 검사
    x1 = int(bbox.xmin * frame_width)
    y1 = int(bbox.ymin * frame_height)
    x2 = int((bbox.xmin + bbox.width) * frame_width)
    y2 = int((bbox.ymin + bbox.height) * frame_height)
    
    if x1 < 0 or y1 < 0 or x2 > frame_width or y2 > frame_height:
        return False
    
    return True

def detect_and_blur(frame, blur_face=True, blur_plate=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_face:
        # MediaPipe 얼굴 검출 적용
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                
                # 검출 결과 유효성 검증
                if not is_valid_face_detection(bbox, w, h):
                    continue
                
                # 신뢰도 추가 검증 (MediaPipe score 활용)
                if hasattr(detection, 'score') and len(detection.score) > 0:
                    confidence = detection.score[0]
                    if confidence < 0.8:  # 높은 신뢰도만 허용
                        continue
                
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                
                # 경계 보정
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    frame[y1:y2, x1:x2] = anonymize_pixelate(roi.copy(), blocks=15)
    if blur_plate:
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 20))
        for (x, y, w, h) in plates:
            roi = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = anonymize_pixelate(roi.copy(), blocks=15)
    return frame

def enhanced_face_detection(frame):
    """향상된 얼굴 검출 - MediaPipe와 Haar Cascade 조합"""
    faces_to_blur = []
    h, w, _ = frame.shape
    
    # 1. MediaPipe 얼굴 검출
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
def process_frame(frame, mode="face_plate"):
    if mode == "face":
        return detect_and_blur_enhanced(frame, blur_face=True, blur_plate=False)
    elif mode == "plate":
        return detect_and_blur(frame, blur_face=False, blur_plate=True)
    elif mode == "face_plate":
        return detect_and_blur_enhanced(frame, blur_face=True, blur_plate=True)
    else:
        return frame

def detect_and_blur_enhanced(frame, blur_face=True, blur_plate=True):
    """향상된 검출 로직을 사용한 블러 처리"""
    if blur_face:
        # 향상된 얼굴 검출 사용
        validated_faces = enhanced_face_detection(frame)
        for (x1, y1, x2, y2) in validated_faces:
            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                frame[y1:y2, x1:x2] = anonymize_pixelate(roi.copy(), blocks=15)
    
    if blur_plate:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 20))
        for (x, y, w, h) in plates:
            roi = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = anonymize_pixelate(roi.copy(), blocks=15)
    
    return frame

