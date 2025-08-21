from server.app import app
from server.config import HOST, PORT, S3_BUCKET_NAME, USE_S3, FACE_DETECTION_CONFIDENCE_THRESHOLD, FACE_SIMILARITY_THRESHOLD, PROCESSING_DURATION_SECONDS

if __name__ == "__main__":
    import uvicorn
    print(f"통합 모듈형 API 서버 시작: http://{HOST}:{PORT}")
    print(f"S3 버킷: {S3_BUCKET_NAME if USE_S3 else '비활성화'}")
    print(f"얼굴 검출 신뢰도 임계값: {FACE_DETECTION_CONFIDENCE_THRESHOLD}")
    print(f"얼굴 유사도 임계값: {FACE_SIMILARITY_THRESHOLD}")
    print(f"처리 시간: {PROCESSING_DURATION_SECONDS}s")
    uvicorn.run(app, host=HOST, port=PORT)
