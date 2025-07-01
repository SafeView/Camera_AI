import cv2
import numpy as np
import time

# 모자이크 토글 변수
enable_blur = True

# 얼굴 탐지 모델 경로
prototxt = "deploy.prototxt"
weights = "res10_300x300_ssd_iter_140000.caffemodel"

# 얼굴 인식 네트워크 로딩
net = cv2.dnn.readNetFromCaffe(prototxt, weights)

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

    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    detections = net.forward()

    # 얼굴 탐지 반복
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.3:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # ROI 추출 및 픽셀화
        face = frame[startY:endY, startX:endX]
        if enable_blur:
            face = anonymize_face_pixelate(face, blocks=15)
        frame[startY:endY, startX:endX] = face

    # 모자이크 상태 표시 텍스트
    status_text = "Blur ON" if enable_blur else "Blur OFF"
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 255, 0) if enable_blur else (0, 0, 255), 2)

    # 탐지된 얼굴 수 표시
    face_count = 0
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence >= 0.5:
            face_count += 1
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