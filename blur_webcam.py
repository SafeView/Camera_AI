import cv2
import numpy as np
import time

# 모자이크 토글 변수
enable_blur = True

# OpenCV 내장 얼굴 탐지 모델 경로
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

# 웹캠 시작
cap = cv2.VideoCapture(0)
time.sleep(2.0)  # 카메라 워밍업

def anonymize_face_pixelate(image, blocks=20):
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

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # 얼굴 탐지 반복
    for (startX, startY, w, h) in faces:
        endX = startX + w
        endY = startY + h
        face = frame[startY:endY, startX:endX]
        if enable_blur:
            face = anonymize_face_pixelate(face, blocks=15)
        frame[startY:endY, startX:endX] = face

    # 모자이크 상태 표시 텍스트
    status_text = "Blur ON" if enable_blur else "Blur OFF"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0) if enable_blur else (0, 0, 255), 2)

    # 탐지된 얼굴 수 표시
    face_count = len(faces)
    cv2.putText(frame, f"Faces: {face_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (255, 255, 255), 2)

    cv2.imshow("SafeView", frame)

    key = cv2.waitKey(1) & 0xFF

    # 'b' 키를 누르면 모자이크 토글
    if key == ord('b'):
        enable_blur = not enable_blur

    # 'q' 키로 종료
    if key == ord('q'):
        break

# 종료
cap.release()
cv2.destroyAllWindows()