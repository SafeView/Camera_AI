import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from AI_processor import process_frame
import io

app = FastAPI()

# 연결된 WebSocket 추적용
active_websockets = set()

@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    await websocket.accept()
    active_websockets.add(websocket)
    try:
        while True:
            data = await websocket.receive_bytes()
            nparr = np.frombuffer(data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            frame = process_frame(frame, mode="face_plate")
            _, jpg = cv2.imencode('.jpg', frame)
            await websocket.send_bytes(jpg.tobytes())
    except Exception as e:
        print("WebSocket closed:", e)
    finally:
        active_websockets.discard(websocket)

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
