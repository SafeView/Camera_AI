from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional
import cv2
from AI_processor import process_frame
from RTSP import stream_rtsp_and_process, stream_stop

app = FastAPI()

class RTSPRequest(BaseModel):
    rtsp_url: str
    mosaic_mode: Optional[str] = "face_plate"

@app.post("/mosaic_rtsp/")
def mosaic_rtsp(request: RTSPRequest):
    # RTSP 스트림을 받아 모자이크 처리 (실제 서비스에서는 비동기/프레임 반환 등 추가 필요)
    stream_rtsp_and_process(request.rtsp_url, mosaic_mode=request.mosaic_mode)
    return {"status": "processing started"}

@app.post("/stop_stream/")
def stop_stream():
    result = stream_stop()
    if result:
        return {"status": "stream stopped"}
    else:
        return {"status": "no active stream"}
