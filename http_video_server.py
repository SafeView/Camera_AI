import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, File, UploadFile, Form
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from AI_processor import process_frame
import io
import os
from datetime import datetime
import threading
import boto3
from botocore.exceptions import ClientError
import tempfile
from dotenv import load_dotenv
import json
import aiohttp
import asyncio

# 트래킹 시스템 import
from tracking_system.face_tracker import FaceTracker
from tracking_system.dependencies import Dependencies
from tracking_system.models import TrackingState, TrackingResult

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

# 키 검증 관련 설정
BACKEND_API_URL = os.getenv("BACKEND_API_URL", "http://localhost:8080")  # 백엔드 API URL
AI_API_KEY = os.getenv("AI_API_KEY")  # AI API 키

# 검증된 사용자별 모자이크 해제 상태 관리
verified_users = {}  # {websocket_id: {"is_verified": bool, "decryption_token": str, "camera_id": str}}
verification_lock = threading.Lock()

# 녹화 관련 변수
is_recording = False
video_writer = None
recording_filename = None
recording_lock = threading.Lock()
TEMP_DIR = tempfile.gettempdir()

# 트래킹 시스템 초기화
try:
    tracking_dependencies = Dependencies()
    tracking_system = FaceTracker(tracking_dependencies)
    print("✅ 트래킹 시스템 초기화 성공")
except Exception as e:
    print(f"❌ 트래킹 시스템 초기화 실패: {e}")
    tracking_system = None

# 클릭 기반 트래킹을 위한 변수
pending_click_coordinates = None
click_lock = threading.Lock()

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

# S3 업로드 함수
def upload_to_s3(file_path, filename):
    if not USE_S3 or not s3_client:
        return False, "S3 not configured"
    
    try:
        s3_client.upload_file(file_path, S3_BUCKET_NAME, f"recordings/{filename}")
        return True, f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/recordings/{filename}"
    except Exception as e:
        return False, str(e)

# 키 검증 함수
async def verify_key_with_backend(access_token: str, camera_id: str):
    """백엔드 API로 키 검증 요청"""
    if not AI_API_KEY:
        return {"success": False, "message": "AI API 키가 설정되지 않았습니다."}
    
    url = f"{BACKEND_API_URL}/api/decryption/keys/verify/ai"
    headers = {
        "AiApiKey": AI_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "accessToken": access_token,
        "cameraId": camera_id
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, json=data) as response:
                if response.status == 200:
                    result = await response.json()
                    return {"success": True, "data": result}
                else:
                    error_text = await response.text()
                    return {"success": False, "message": f"키 검증 실패: {error_text}"}
    except Exception as e:
        return {"success": False, "message": f"백엔드 연결 오류: {str(e)}"}

# 통합 비디오 WebSocket 엔드포인트
@app.websocket("/ws/video")
async def unified_video_ws(websocket: WebSocket):
    global is_recording, video_writer, recording_filename, pending_click_coordinates
    await websocket.accept()
    websocket_id = id(websocket)
    active_websockets.add(websocket)
    
    # WebSocket 초기화
    websocket._last_tracking_send = datetime.now()
    
    # 초기 상태: 모자이크 적용 (미검증 상태)
    with verification_lock:
        verified_users[websocket_id] = {
            "is_verified": False,
            "decryption_token": None,
            "camera_id": None
        }
    
    try:
        while True:
            try:
                # 메시지 수신 (JSON 또는 바이너리)
                message = await websocket.receive()
                
                if "text" in message:
                    # JSON 메시지 처리 (키 검증 요청)
                    try:
                        data = json.loads(message["text"])
                        if data.get("type") == "key_verification":
                            access_token = data.get("accessToken")
                            camera_id = data.get("cameraId")
                            
                            if not access_token or not camera_id:
                                await websocket.send_text(json.dumps({
                                    "type": "verification_result",
                                    "success": False,
                                    "message": "accessToken과 cameraId는 필수입니다."
                                }))
                                continue
                            
                            # 백엔드로 키 검증 요청
                            verification_result = await verify_key_with_backend(access_token, camera_id)
                            
                            if verification_result["success"]:
                                # 백엔드 응답 구조: ApiResponse<KeyVerificationResponseDto>
                                backend_response = verification_result["data"]
                                
                                # ApiResponse에서 data 필드 추출
                                if backend_response.get("isSuccess") and "data" in backend_response:
                                    response_data = backend_response["data"]
                                    
                                    # KeyVerificationResponseDto 필드 확인
                                    is_valid = response_data.get("valid", False)  # 'isValid' 대신 'valid' 사용
                                    can_decrypt = response_data.get("canDecrypt", False)
                                    
                                    if is_valid and can_decrypt:
                                        with verification_lock:
                                            verified_users[websocket_id] = {
                                                "is_verified": True,
                                                "decryption_token": response_data.get("decryptionToken"),
                                                "camera_id": camera_id
                                            }
                                        
                                        await websocket.send_text(json.dumps({
                                            "type": "verification_result",
                                            "success": True,
                                            "message": "키 검증 성공 - 모자이크가 해제됩니다.",
                                            "canDecrypt": True,
                                            "isValid": True,
                                            "expiresAt": response_data.get("expiresAt"),
                                            "remainingUses": response_data.get("remainingUses"),
                                            "decryptionToken": response_data.get("decryptionToken"),
                                            "verifiedAt": response_data.get("verifiedAt"),
                                            "blockchainVerified": response_data.get("blockchainVerified")
                                        }))
                                    else:
                                        await websocket.send_text(json.dumps({
                                            "type": "verification_result",
                                            "success": False,
                                            "message": response_data.get("message", f"키 검증 실패 - 권한이 없습니다."),
                                            "canDecrypt": False,
                                            "isValid": is_valid
                                        }))
                                else:
                                    # API 응답이 실패인 경우
                                    error_message = backend_response.get("message", "API 응답 오류")
                                    await websocket.send_text(json.dumps({
                                        "type": "verification_result",
                                        "success": False,
                                        "message": f"백엔드 API 오류: {error_message}",
                                        "canDecrypt": False
                                    }))
                            else:
                                await websocket.send_text(json.dumps({
                                    "type": "verification_result",
                                    "success": False,
                                    "message": verification_result["message"],
                                    "canDecrypt": False
                                }))
                        
                        elif data.get("type") == "disconnect":
                            # 개별 사용자 연결 해제 요청
                            await websocket.send_text(json.dumps({
                                "type": "disconnect_result",
                                "success": True,
                                "message": "연결이 해제됩니다."
                            }))
                            # WebSocket 연결 종료
                            await websocket.close()
                            break
                        
                        elif data.get("type") == "tracking_click":
                            # 클릭 기반 트래킹 시작
                            x = data.get("x")
                            y = data.get("y")
                            
                            if x is not None and y is not None:
                                # 클릭 좌표 저장
                                global pending_click_coordinates
                                with click_lock:
                                    pending_click_coordinates = (x, y)
                                
                                await websocket.send_text(json.dumps({
                                    "type": "tracking_click_result",
                                    "success": True,
                                    "message": f"클릭 좌표 ({x}, {y})가 기록되었습니다. 다음 프레임에서 트래킹을 시도합니다.",
                                    "coordinates": {"x": x, "y": y}
                                }))
                            else:
                                await websocket.send_text(json.dumps({
                                    "type": "tracking_click_result",
                                    "success": False,
                                    "message": "클릭 좌표가 제공되지 않았습니다."
                                }))
                        
                        elif data.get("type") == "tracking_control":
                            # 트래킹 제어 명령
                            action = data.get("action")
                            
                            if action == "clear":
                                if tracking_system:
                                    tracking_system.clear_target()
                                    await websocket.send_text(json.dumps({
                                        "type": "tracking_control_result",
                                        "success": True,
                                        "action": "clear",
                                        "message": "추적 타겟이 해제되었습니다."
                                    }))
                                else:
                                    await websocket.send_text(json.dumps({
                                        "type": "tracking_control_result",
                                        "success": False,
                                        "action": "clear",
                                        "message": "트래킹 시스템이 초기화되지 않았습니다."
                                    }))
                            elif action == "suspend":
                                if tracking_system:
                                    tracking_system.suspend_tracking()
                                    await websocket.send_text(json.dumps({
                                        "type": "tracking_control_result",
                                        "success": True,
                                        "action": "suspend",
                                        "message": "추적이 일시 중지되었습니다."
                                    }))
                                else:
                                    await websocket.send_text(json.dumps({
                                        "type": "tracking_control_result",
                                        "success": False,
                                        "action": "suspend",
                                        "message": "트래킹 시스템이 초기화되지 않았습니다."
                                    }))
                            else:
                                await websocket.send_text(json.dumps({
                                    "type": "tracking_control_result",
                                    "success": False,
                                    "message": f"알 수 없는 액션: {action}"
                                }))
                    
                    except json.JSONDecodeError:
                        await websocket.send_text(json.dumps({
                            "type": "error",
                            "message": "잘못된 JSON 형식입니다."
                        }))
                
                elif "bytes" in message:
                    # 비디오 프레임 처리
                    data = message["bytes"]
                    nparr = np.frombuffer(data, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    # 클릭 좌표가 있으면 트래킹 시작
                    if tracking_system:
                        with click_lock:
                            if pending_click_coordinates:
                                x, y = pending_click_coordinates
                                pending_click_coordinates = None
                                
                                # 클릭 좌표로 트래킹 시작
                                success = tracking_system.start_tracking(frame, (x, y))
                                if success:
                                    print(f"✅ 클릭 좌표 ({x}, {y})에서 트래킹 시작 성공")
                                else:
                                    print(f"❌ 클릭 좌표 ({x}, {y})에서 트래킹 시작 실패")
                    
                    # 트래킹 업데이트 (트래킹 시스템이 활성화된 경우)
                    tracking_result = None
                    if tracking_system and tracking_system.is_tracking_active():
                        tracking_result = tracking_system.update_tracking(frame)
                    
                    # 사용자 검증 상태에 따라 모자이크 적용 여부 결정
                    with verification_lock:
                        user_status = verified_users.get(websocket_id, {"is_verified": False})
                    
                    if user_status["is_verified"]:
                        # 검증된 사용자: 원본 영상
                        processed_frame = frame
                    else:
                        # 미검증 사용자: 모자이크 적용
                        processed_frame = process_frame(frame, mode="face_plate")
                    
                    # 트래킹 결과가 있으면 프레임에 표시
                    if tracking_result and tracking_result.is_tracking:
                        processed_frame = _draw_tracking_on_frame(processed_frame, tracking_result)
                    
                    # 트래킹 상태를 JSON으로 전송 (주기적으로)
                    if tracking_result and hasattr(websocket, '_last_tracking_send'):
                        current_time = datetime.now()
                        if not hasattr(websocket, '_last_tracking_send') or \
                           (current_time - websocket._last_tracking_send).total_seconds() > 1.0:  # 1초마다
                            
                            tracking_status = {
                                "type": "tracking_status",
                                "is_tracking": tracking_result.is_tracking,
                                "target_name": tracking_result.target_name,
                                "tracking_state": tracking_result.state.value,
                                "similarity": tracking_result.similarity,
                                "face_detection": tracking_result.face_detection.bbox if tracking_result.face_detection else None,
                                "person_detection": tracking_result.person_detection.bbox if tracking_result.person_detection else None
                            }
                            
                            try:
                                await websocket.send_text(json.dumps(tracking_status))
                                websocket._last_tracking_send = current_time
                            except Exception as e:
                                print(f"트래킹 상태 전송 오류: {e}")
                    elif not tracking_result and hasattr(websocket, '_last_tracking_send'):
                        # 트래킹이 중지된 경우 상태 전송
                        tracking_status = {
                            "type": "tracking_status",
                            "is_tracking": False,
                            "target_name": None,
                            "tracking_state": "no_target",
                            "similarity": 0.0,
                            "face_detection": None,
                            "person_detection": None
                        }
                        
                        try:
                            await websocket.send_text(json.dumps(tracking_status))
                            websocket._last_tracking_send = datetime.now()
                        except Exception as e:
                            print(f"트래킹 상태 전송 오류: {e}")
                    
                    _, jpg = cv2.imencode('.jpg', processed_frame)
                    
                    # 녹화 중이면 프레임 저장 (원본 또는 처리된 프레임)
                    with recording_lock:
                        if is_recording and video_writer is not None:
                            video_writer.write(processed_frame)
                    
                    await websocket.send_bytes(jpg.tobytes())
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"WebSocket message processing error: {e}")
                break
                
    except Exception as e:
        print("WebSocket connection error:", e)
    finally:
        active_websockets.discard(websocket)
        # 사용자 상태 정리
        with verification_lock:
            if websocket_id in verified_users:
                del verified_users[websocket_id]

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
                "storage": "S3",
                "error": "no error"
            }
        else:
            return {"error": "Failed to start recording"}

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
                        "storage": "S3",
                        "error" : "no error"
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
        
        return {"url": url, "filename": filename, "storage": "S3", "error": "no error"}
    
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

# HTTP 엔드포인트로 모든 WebSocket 연결 해제 (관리자용)
@app.post("/disconnect_ws")
async def disconnect_ws():
    """
    모든 WebSocket 연결 해제 (관리자용)
    """
    print("모든 WebSocket 연결 해제 실행")
    closed = 0
    
    for ws in list(active_websockets):
        try:
            await ws.close()
            closed += 1
            print(f"WebSocket 연결 해제 성공 - ID: {id(ws)}")
        except Exception as e:
            print(f"WebSocket close error: {e}")
    
    return {
        "success": True,
        "message": f"모든 WebSocket 연결이 해제되었습니다.",
        "disconnected": closed,
        "total_active_before": len(active_websockets)
    }

# 현재 연결된 사용자 상태 확인
@app.get("/verification_status")
async def get_verification_status():
    with verification_lock:
        verified_count = sum(1 for user in verified_users.values() if user["is_verified"])
        total_count = len(verified_users)
    
    return {
        "total_connections": total_count,
        "verified_connections": verified_count,
        "unverified_connections": total_count - verified_count
    }

# 모든 사용자 모자이크 강제 적용 (긴급상황용)
@app.post("/force_mosaic")
async def force_mosaic_all():
    with verification_lock:
        for websocket_id in verified_users:
            verified_users[websocket_id]["is_verified"] = False
            verified_users[websocket_id]["decryption_token"] = None
    
    # 연결된 모든 WebSocket에 모자이크 강제 적용 알림
    disconnected = 0
    for ws in list(active_websockets):
        try:
            await ws.send_text(json.dumps({
                "type": "force_mosaic",
                "message": "모든 연결에 모자이크가 강제 적용되었습니다."
            }))
        except Exception as e:
            print(f"WebSocket send error: {e}")
            disconnected += 1
    
    return {
        "message": "모든 사용자에게 모자이크가 강제 적용되었습니다.",
        "affected_connections": len(verified_users),
        "disconnected": disconnected
    }

# 추적 세션 관리 변수
tracking_sessions = {}
current_session_id = None

# 추적 세션 시작
@app.post("/tracking/start")
async def start_tracking(session_name: str = "default"):
    """추적 세션 시작"""
    global current_session_id
    
            # 트래킹 시스템 서버 세션 시작
    if tracking_system:
        tracking_system.server_manager.start_session()
    
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_name}"
    tracking_sessions[session_id] = {
        "start_time": datetime.now(),
        "session_name": session_name,
        "targets": {},
        "events": []
    }
    
    current_session_id = session_id
    
    return {
        "session_id": session_id,
        "status": "started",
        "message": f"추적 세션이 시작되었습니다: {session_name}",
        "tracking_system": "FaceTracker initialized"
    }

# 추적 세션 중지
@app.post("/tracking/stop")
async def stop_tracking():
    """추적 세션 중지"""
    global current_session_id
    
    if current_session_id is None:
        return {"error": "활성 세션이 없습니다."}
    
    # 트래킹 시스템 서버 세션 종료
    tracking_stats = {}
    if tracking_system:
        tracking_system.server_manager.stop_session()
        tracking_stats = tracking_system.get_statistics()
    
    session = tracking_sessions[current_session_id]
    session["end_time"] = datetime.now()
    
    # 간단한 레포트 생성
    duration = (session["end_time"] - session["start_time"]).total_seconds()
    report = {
        "session_id": current_session_id,
        "session_name": session["session_name"],
        "start_time": session["start_time"].isoformat(),
        "end_time": session["end_time"].isoformat(),
        "duration_seconds": duration,
        "targets": session["targets"],
        "events": session["events"],
        "tracking_statistics": tracking_stats,
        "summary": {
            "total_events": len(session["events"]),
            "total_targets": len(session["targets"]),
            "tracking_targets_created": tracking_stats.get("total_targets_created", 0)
        }
    }
    
    # 레포트 저장
    report_filename = f"tracking_report_{current_session_id}.json"
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    result = {
        "session_id": current_session_id,
        "status": "stopped",
        "report_file": report_filename,
        "summary": report["summary"],
        "tracking_stats": tracking_stats
    }
    
    current_session_id = None
    return result

def _draw_tracking_on_frame(frame: np.ndarray, tracking_result) -> np.ndarray:
    """프레임에 트래킹 결과 그리기"""
    if not tracking_result or not tracking_result.is_tracking:
        return frame
    
    # 얼굴 박스 그리기 (노란색)
    if tracking_result.face_detection:
        detection = tracking_result.face_detection
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
        
        # 타겟 이름과 유사도 표시
        label = f"{tracking_result.target_name}: {tracking_result.similarity:.3f}"
        cv2.putText(frame, label, (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # 사람 전체 박스 그리기 (초록색)
    if tracking_result.person_detection:
        detection = tracking_result.person_detection
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # 전신 라벨
        person_label = f"{tracking_result.target_name} (Full Body)"
        cv2.putText(frame, person_label, (x, y + h + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # 상태 정보 표시
    status_text = f"Tracking: {tracking_result.target_name}"
    cv2.putText(frame, status_text, (10, frame.shape[0] - 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame

# 타겟 관리 변수
target_persons = {}

# 타겟 추가
@app.post("/targets/add")
async def add_target_person(image: UploadFile = File(...), name: str = Form(...)):
    """특정 사람을 타겟으로 추가"""
    try:
        # 이미지 파일 읽기
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return {"error": "유효하지 않은 이미지입니다."}
        
        # 트래킹 시스템에 타겟 추가
        if not tracking_system:
            return {"error": "트래킹 시스템이 초기화되지 않았습니다."}
        
        # 이미지에서 얼굴 검출
        face_detections = tracking_system.face_detector.detect_faces(img)
        
        if not face_detections:
            return {"error": "이미지에서 얼굴을 검출할 수 없습니다."}
        
        # 가장 큰 얼굴 선택 (중앙에 있을 가능성이 높음)
        largest_face = max(face_detections, key=lambda x: x.area)
        
        # 얼굴 임베딩 계산
        embedding = tracking_system.embedding_processor.compute_face_embedding(img, largest_face)
        
        if embedding is None:
            return {"error": "얼굴 임베딩을 생성할 수 없습니다."}
        
        # 타겟 생성
        from tracking_system.models import TrackingTarget
        target = TrackingTarget(
            name=name,
            embedding=embedding,
            last_face_detection=largest_face
        )
        
        # 트래킹 시스템에 타겟 설정
        tracking_system.current_target = target
        tracking_system.tracking_state = TrackingState.TRACKING
        tracking_system._reset_tracking_state()
        
        # 타겟 데이터 저장
        tracking_system._save_target_data(target, img)
        
        # 타겟 저장
        target_persons[name] = {
            "added_at": datetime.now().isoformat(),
            "image_shape": img.shape,
            "face_bbox": largest_face.bbox,
            "embedding_shape": embedding.shape
        }
        
        if current_session_id:
            tracking_sessions[current_session_id]["targets"][name] = {
                "added_at": datetime.now().isoformat(),
                "image_size": img.shape,
                "face_bbox": largest_face.bbox
            }
        
        return {
            "status": "success",
            "message": f"타겟 '{name}'이 추가되었습니다.",
            "target_name": name,
            "face_detected": True,
            "embedding_created": True,
            "tracking_active": True
        }
    
    except Exception as e:
        return {"error": f"서버 오류: {str(e)}"}

# 타겟 목록 조회
@app.get("/targets/list")
async def list_targets():
    """등록된 타겟 목록 조회"""
    targets = list(target_persons.keys())
    current_target = None
    tracking_state = "no_target"
    
    if tracking_system:
        current_target = tracking_system.get_current_target()
        tracking_state = tracking_system.get_tracking_state().value
    
    return {
        "targets": targets,
        "count": len(targets),
        "current_tracking_target": current_target.name if current_target else None,
        "tracking_state": tracking_state
    }

# 실시간 통계 조회
@app.get("/statistics/live")
async def get_live_statistics():
    """실시간 통계 조회"""
    # 트래킹 시스템 통계 가져오기
    tracking_stats = {}
    tracking_system_info = {
        "current_target": None,
        "tracking_state": "no_target",
        "total_targets_created": 0,
        "lost_frame_count": 0,
        "target_suspended": False,
        "is_tracking_active": False
    }
    
    if tracking_system:
        tracking_stats = tracking_system.get_statistics()
        tracking_system_info = {
            "current_target": tracking_stats.get("current_target"),
            "tracking_state": tracking_stats.get("tracking_state"),
            "total_targets_created": tracking_stats.get("total_targets_created", 0),
            "lost_frame_count": tracking_stats.get("lost_frame_count", 0),
            "target_suspended": tracking_stats.get("target_suspended", False),
            "is_tracking_active": tracking_system.is_tracking_active()
        }
        
        # 임베딩 정보 추가
        try:
            embedding_info = tracking_system.embedding_processor.get_embedding_info()
            tracking_system_info["embedding_models"] = embedding_info.get("available_models", [])
            tracking_system_info["ensemble_enabled"] = embedding_info.get("ensemble_enabled", False)
        except Exception as e:
            tracking_system_info["embedding_models"] = []
            tracking_system_info["ensemble_enabled"] = False
    
    stats = {
        "active_connections": len(active_websockets),
        "verified_connections": sum(1 for user in verified_users.values() if user["is_verified"]),
        "total_targets": len(target_persons),
        "is_recording": is_recording,
        "timestamp": datetime.now().isoformat(),
        "tracking_system": tracking_system_info
    }
    
    if current_session_id:
        session = tracking_sessions[current_session_id]
        stats["session_id"] = current_session_id
        stats["session_name"] = session["session_name"]
        stats["session_duration"] = (datetime.now() - session["start_time"]).total_seconds()
    
    return stats

# 임베딩 정보 조회 API
@app.get("/embedding/info")
async def get_embedding_info():
    """임베딩 시스템 정보 조회"""
    if not tracking_system:
        return {
            "error": "트래킹 시스템이 초기화되지 않았습니다.",
            "embedding_available": False
        }
    
    try:
        embedding_info = tracking_system.embedding_processor.get_embedding_info()
        return {
            "embedding_available": True,
            "embedding_info": embedding_info,
            "system_status": "active"
        }
    except Exception as e:
        return {
            "error": f"임베딩 정보 조회 실패: {str(e)}",
            "embedding_available": False
        }

# 트래킹 제어 API들
@app.post("/tracking/clear_target")
async def clear_tracking_target():
    """현재 추적 타겟 해제"""
    if not tracking_system:
        return {
            "status": "error",
            "message": "트래킹 시스템이 초기화되지 않았습니다."
        }
    
    tracking_system.clear_target()
    return {
        "status": "success",
        "message": "추적 타겟이 해제되었습니다.",
        "tracking_state": tracking_system.get_tracking_state().value
    }

@app.post("/tracking/suspend")
async def suspend_tracking():
    """추적 일시 중지"""
    if not tracking_system:
        return {
            "status": "error",
            "message": "트래킹 시스템이 초기화되지 않았습니다."
        }
    
    tracking_system.suspend_tracking()
    return {
        "status": "success",
        "message": "추적이 일시 중지되었습니다.",
        "tracking_state": tracking_system.get_tracking_state().value
    }

@app.get("/tracking/status")
async def get_tracking_status():
    """트래킹 상태 조회"""
    if not tracking_system:
        return {
            "is_tracking_active": False,
            "tracking_state": "no_target",
            "current_target": None,
            "statistics": {},
            "error": "트래킹 시스템이 초기화되지 않았습니다."
        }
    
    current_target = tracking_system.get_current_target()
    tracking_stats = tracking_system.get_statistics()
    
    return {
        "is_tracking_active": tracking_system.is_tracking_active(),
        "tracking_state": tracking_system.get_tracking_state().value,
        "current_target": {
            "name": current_target.name if current_target else None,
            "created_at": current_target.created_at if current_target else None,
            "last_face_detection": current_target.last_face_detection.bbox if current_target and current_target.last_face_detection else None,
            "last_person_detection": current_target.last_person_detection.bbox if current_target and current_target.last_person_detection else None
        },
        "statistics": tracking_stats
    }

# 프레임에서 클릭 기반 타겟 설정 API
@app.post("/tracking/set_target_by_click")
async def set_target_by_click(x: int = Form(...), y: int = Form(...)):
    """클릭 좌표로 타겟 설정"""
    # 현재 프레임이 없으므로 실시간 처리는 WebSocket에서 처리
    return {
        "status": "info",
        "message": "클릭 기반 타겟 설정은 WebSocket 연결을 통해 실시간으로 처리됩니다.",
        "coordinates": {"x": x, "y": y}
    }

# 트래킹 결과 조회 API
@app.get("/tracking/current_result")
async def get_current_tracking_result():
    """현재 트래킹 결과 조회"""
    if not tracking_system:
        return {
            "is_tracking": False,
            "message": "트래킹 시스템이 초기화되지 않았습니다."
        }
    
    if not tracking_system.is_tracking_active():
        return {
            "is_tracking": False,
            "message": "현재 추적 중인 타겟이 없습니다."
        }
    
    # 실제 트래킹 결과는 실시간 프레임에서 계산되므로
    # 현재 저장된 타겟 정보만 반환
    current_target = tracking_system.get_current_target()
    
    return {
        "is_tracking": True,
        "target_name": current_target.name if current_target else None,
        "tracking_state": tracking_system.get_tracking_state().value,
        "last_face_detection": current_target.last_face_detection.bbox if current_target and current_target.last_face_detection else None,
        "last_person_detection": current_target.last_person_detection.bbox if current_target and current_target.last_person_detection else None,
        "created_at": current_target.created_at if current_target else None
    }

# 서버 실행
if __name__ == "__main__":
    import uvicorn
    print("🚀 비디오 트래킹 서버 시작 중...")
    print("📡 포트: 8002")
    print("🌐 접속 URL: http://localhost:8002")
    print("🔗 WebSocket URL: ws://localhost:8002/ws/video")
    print("📋 API 문서: http://localhost:8002/docs")
    print("=" * 50)
    
    uvicorn.run(
        "http_video_server:app",
        host="0.0.0.0",
        port=8002,
        reload=False,
        log_level="info"
    )
