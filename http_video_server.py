import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from AI_processor import process_frame
import io
import os
from datetime import datetime
import threading
import boto3
from botocore.exceptions import ClientError
import tempfile
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

app = FastAPI()

# CORS 설정 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용으로 모든 도메인 허용, 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# 연결된 WebSocket 추적용
active_websockets = set()

# 녹화 관련 변수
is_recording = False
video_writer = None
recording_filename = None
recording_lock = threading.Lock()
TEMP_DIR = tempfile.gettempdir()

# S3 설정 (환경변수로 설정 권장)
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
S3_REGION = os.getenv("S3_REGION")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# S3 사용 가능 여부 확인
USE_S3 = all([S3_BUCKET_NAME, S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY])

# S3 클라이언트 초기화
s3_client = None
if USE_S3:
    try:
        s3_client = boto3.client(
            's3',
            region_name=S3_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        print("S3 클라이언트 초기화 성공")
    except Exception as e:
        print(f"S3 클라이언트 초기화 실패: {e}")
        USE_S3 = False
else:
    print("S3 환경변수가 설정되지 않음")

# 인증된 공인(모자이크 해제) 연결 엔드포인트
@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    global is_recording, video_writer, recording_filename
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            # 모자이크 적용하지 않음
            _, jpg = cv2.imencode('.jpg', frame)
            
            # 녹화 중이면 프레임 저장
            with recording_lock:
                if is_recording and video_writer is not None:
                    video_writer.write(frame)
            
            await websocket.send_bytes(jpg.tobytes())
    except Exception as e:
        print("WebSocket closed:", e)
    finally:
        active_websockets.discard(websocket)

# 일반 사용자(모자이크 적용) 연결 엔드포인트
@app.websocket("/ws/video/mo")
async def video_mosaic_ws(websocket: WebSocket):
    global is_recording, video_writer, recording_filename
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            processed_frame = process_frame(frame, mode="face_plate")
            _, jpg = cv2.imencode('.jpg', processed_frame)
            
            # 녹화 중이면 처리된 프레임 저장
            with recording_lock:
                if is_recording and video_writer is not None:
                    video_writer.write(processed_frame)
            
            await websocket.send_bytes(jpg.tobytes())
    except Exception as e:
        print("WebSocket closed:", e)
    finally:
        active_websockets.discard(websocket)

# 녹화 시작 엔드포인트
@app.post("/start_recording")
async def start_recording():
    global is_recording, video_writer, recording_filename
    
    if not USE_S3 or not s3_client:
        return {"error": "S3 연결 오류: 환경변수를 확인하세요"}
    
    with recording_lock:
        if is_recording:
            return {"error": "Already recording"}
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_filename = f"recording_{timestamp}.mp4"
        filepath = os.path.join(TEMP_DIR, recording_filename)
        
        # VideoWriter 초기화 (640x480, 30fps 기본값)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(filepath, fourcc, 30.0, (640, 480))
        
        if video_writer.isOpened():
            is_recording = True
            return {
                "status": "Recording started (저장: S3)", 
                "filename": recording_filename,
                "storage": "S3"
            }
        else:
            return {"error": "Failed to start recording"}

# S3 업로드 함수
def upload_to_s3(file_path, filename):
    if not USE_S3 or not s3_client:
        return False, "S3 not configured"
    
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, f"recordings/{filename}")
        return True, f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/recordings/{filename}"
    except Exception as e:
        return False, str(e)

# 녹화 중단 엔드포인트
@app.post("/stop_recording")
async def stop_recording():
    global is_recording, video_writer, recording_filename
    
    if not USE_S3 or not s3_client:
        return {"error": "S3 연결 오류: 환경변수를 확인하세요"}
    
    with recording_lock:
        if not is_recording:
            return {"error": "Not recording"}
        
        is_recording = False
        if video_writer is not None:
            video_writer.release()
            video_writer = None
        
        if recording_filename:
            temp_filepath = os.path.join(TEMP_DIR, recording_filename)
            if os.path.exists(temp_filepath):
                success, result = upload_to_s3(temp_filepath, recording_filename)
                
                # 임시 파일 삭제
                try:
                    os.remove(temp_filepath)
                except Exception as e:
                    print(f"임시 파일 삭제 실패: {e}")
                
                if success:
                    return {
                        "status": "Recording stopped and uploaded to S3", 
                        "filename": recording_filename,
                        "s3_url": result,
                        "storage": "S3"
                    }
                else:
                    return {
                        "status": "Recording stopped but S3 upload failed", 
                        "filename": recording_filename,
                        "error": result,
                        "storage": "S3 (실패)"
                    }
            else:
                return {"error": "Recording file not found"}
        
        return {"status": "Recording stopped", "filename": recording_filename}

# S3에서 녹화 파일 목록 조회
@app.get("/recordings")
async def list_recordings():
    if not USE_S3 or not s3_client:
        return {"error": "S3 연결 오류: 환경변수를 확인하세요"}
    
    try:
        response = s3_client.list_objects_v2(
            Bucket=S3_BUCKET_NAME,
            Prefix="recordings/"
        )
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                filename = obj['Key'].replace('recordings/', '')
                if filename.endswith('.mp4') and filename:  # 빈 파일명 제외
                    files.append({
                        "filename": filename,
                        "size": obj['Size'],
                        "last_modified": obj['LastModified'].isoformat(),
                        "url": f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{obj['Key']}",
                        "storage": "S3"
                    })
        
        files.sort(key=lambda x: x['last_modified'], reverse=True)  # 최신 파일 먼저
        return {"recordings": files, "storage": "S3"}
    
    except Exception as e:
        return {"error": f"S3 error: {str(e)}"}

# S3에서 녹화 파일 직접 접근 URL 생성
@app.get("/recordings/{filename}")
async def get_recording_url(filename: str):
    if not USE_S3 or not s3_client:
        return {"error": "S3 연결 오류: 환경변수를 확인하세요"}
    
    try:
        # S3에서 파일 존재 확인
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=f"recordings/{filename}")
        
        # Pre-signed URL 생성 (1시간 유효)
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': f"recordings/{filename}"},
            ExpiresIn=3600  # 1시간
        )
        
        return {"url": url, "filename": filename, "storage": "S3"}
    
    except Exception as e:
        if hasattr(e, 'response') and e.response['Error']['Code'] == '404':
            return {"error": "File not found"}
        else:
            return {"error": f"S3 error: {str(e)}"}

# 녹화 상태 확인
@app.get("/recording_status")
async def recording_status():
    return {
        "is_recording": is_recording,
        "current_file": recording_filename if is_recording else None,
        "storage_type": "S3",
        "s3_configured": USE_S3
    }

# HTTP 엔드포인트로 모든 WebSocket 연결 해제
@app.post("/disconnect_ws")
async def disconnect_ws():
    closed = 0
    for ws in list(active_websockets):
        try:
            await ws.close()
            closed += 1
        except Exception as e:
            print(f"WebSocket close error: {e}")
    return {"disconnected": closed}
