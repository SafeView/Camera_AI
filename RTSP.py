import cv2
import subprocess
from AI_processor import process_frame

streaming_proc = None  # ffmpeg 프로세스 핸들 전역 변수

def stream_rtsp_and_process(rtsp_url=None, mosaic_mode="face_plate", output_url = "rtmp://localhost/live/mosaic"):
    global streaming_proc
    # rtsp_url이 None 또는 'webcam'이면 로컬 웹캠 사용
    if rtsp_url is None or rtsp_url == "webcam":
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        print("스트림을 열 수 없습니다.")
        return
    # 프레임 크기 정보
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25
    # ffmpeg 송출 명령어 (RTMP 송출)
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
        '-f', 'flv',  # RTMP는 flv 포맷
        output_url
    ]
    streaming_proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed_frame = process_frame(frame, mode=mosaic_mode)
        streaming_proc.stdin.write(processed_frame.tobytes())
    cap.release()
    streaming_proc.stdin.close()
    streaming_proc.wait()
    streaming_proc = None

# 스트리밍 종료 API용 함수
def stream_stop():
    global streaming_proc
    if streaming_proc is not None:
        try:
            streaming_proc.terminate()
            streaming_proc.wait()
        except Exception as e:
            print(f"스트리밍 종료 중 오류: {e}")
        streaming_proc = None
        return True
    return False