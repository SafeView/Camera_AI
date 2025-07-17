import cv2
import numpy as np
import mediapipe as mp

# 얼굴 탐지 모델
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# 번호판 탐지 모델 (예시: haarcascade_russian_plate_number.xml)
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")

mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

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

def detect_and_blur(frame, blur_face=True, blur_plate=True):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if blur_face:
        # MediaPipe 얼굴 검출 적용
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                x2 = int((bbox.xmin + bbox.width) * w)
                y2 = int((bbox.ymin + bbox.height) * h)
                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    frame[y1:y2, x1:x2] = anonymize_pixelate(roi.copy(), blocks=15)
    if blur_plate:
        plates = plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 20))
        for (x, y, w, h) in plates:
            roi = frame[y:y+h, x:x+w]
            frame[y:y+h, x:x+w] = anonymize_pixelate(roi.copy(), blocks=15)
    return frame

# 다양한 모자이크 옵션을 위한 함수 예시
def process_frame(frame, mode="face_plate"):
    if mode == "face":
        return detect_and_blur(frame, blur_face=True, blur_plate=False)
    elif mode == "plate":
        return detect_and_blur(frame, blur_face=False, blur_plate=True)
    elif mode == "face_plate":
        return detect_and_blur(frame, blur_face=True, blur_plate=True)
    else:
        return frame
