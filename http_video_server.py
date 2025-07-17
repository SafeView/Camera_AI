import cv2
import numpy as np
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from AI_processor import process_frame
import io

app = FastAPI()

@app.websocket("/ws/video")
async def video_ws(websocket: WebSocket):
    await websocket.accept()
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
