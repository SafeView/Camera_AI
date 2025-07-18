import cv2
import numpy as np
import subprocess

# 입력(프론트 서버에서 송출하는 RTMP/RTSP 등)과 출력(프론트 서버로 송출) 주소
input_url = "rtmp://localhost/live/input"  # 프론트 서버에서 송출하는 주소
output_url = "rtmp://localhost/live/output"  # 프론트 서버로 다시 송출할 주소
fps = 30

# 모자이크 토글 변수
enable_blur = True


# AI_processor 모듈 import
from AI_processor import process_frame

import time
# 입력 스트림이 연결될 때까지 대기
while True:
    cap = cv2.VideoCapture(input_url)
    ret, frame = cap.read()
    if ret:
        print("입력 스트림 연결 성공.")
        break
    print("입력 스트림 대기 중... (1초 후 재시도)")
    cap.release()
    time.sleep(1)
height, width = frame.shape[:2]

# ffmpeg 송출 프로세스 준비
ffmpeg_cmd = [
    'ffmpeg',
    '-y',
    '-f', 'rawvideo',
    '-vcodec', 'rawvideo',
    '-pix_fmt', 'bgr24',
    '-s', f'{width}x{height}',
    '-r', str(int(fps)),
    '-i', '-',
    '-c:v', 'libx264',
    '-preset', 'veryfast',
    '-f', 'flv',
    output_url
]
proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)

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
    # frame = cv2.flip(frame, 1)  # 필요시 사용
    if enable_blur:
        frame = process_frame(frame, mode="face_plate")
    # 프레임을 프론트 서버로 송출
    proc.stdin.write(frame.tobytes())

cap.release()
proc.stdin.close()
proc.wait()