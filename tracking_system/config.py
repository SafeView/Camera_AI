"""
설정 및 상수 관리 모듈
"""

class Config:
    """시스템 전체 설정"""
    
    # Pinecone 설정
    PINECONE_API_KEY = "pcsk_6LujBQ_KpiPt7ZTH7xzr1V3tAdjaJvW2M7xFAMTHKzYGkp2iYK1o57Q4aiWAFiAVdcJkP8"
    PINECONE_INDEX_NAME = "safeview"
    PINECONE_CLOUD = "aws"
    PINECONE_REGION = "us-east-1"
    EMBEDDING_DIMENSION = 1024
    
    # 서버 설정
    SERVER_URL = "http://localhost:8002"
    SERVER_TIMEOUT = 5
    
    # 추적 알고리즘 설정 (다중 인물 최적화)
    STRICT_THRESHOLD_INSIGHT = 0.95  # 매우 엄격한 임계값 (다른 사람 방지)
    STRICT_THRESHOLD_HISTOGRAM = 0.85  # 매우 엄격한 임계값 (다른 사람 방지)
    SIMILARITY_THRESHOLD_INSIGHT = 0.90  # 매우 엄격한 임계값 (다른 사람 방지)
    SIMILARITY_THRESHOLD_HISTOGRAM = 0.75  # 매우 엄격한 임계값 (다른 사람 방지)
    MAX_MOVEMENT_MULTIPLIER = 1.5  # 더 엄격한 이동 제한
    LOST_FRAME_THRESHOLD = 999999  # 거의 무제한 (다른 사람 자동 지정 방지)
    
    # 얼굴 검출 설정
    MIN_FACE_SIZE = 30
    MAX_FACE_SIZE_RATIO = 0.7
    MIN_ASPECT_RATIO = 0.6
    MAX_ASPECT_RATIO = 1.6
    
    # MediaPipe 설정 (다중 인물 최적화)
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE = 0.6  # 더 많은 얼굴 검출
    
    # Haar Cascade 설정 (다중 인물 최적화)
    HAAR_SCALE_FACTOR = 1.05  # 더 세밀한 스케일링
    HAAR_MIN_NEIGHBORS = 4    # 더 많은 검출 허용
    HAAR_MIN_SIZE = (30, 30)  # 더 작은 얼굴도 검출
    
    # UI 설정
    WINDOW_NAME = "Click Tracking System"
    
    # 임베딩 모델 설정
    INSIGHT_FACE_DET_SIZE = (640, 640)
    HISTOGRAM_SIZE = (128, 128)
    HISTOGRAM_BINS = [8, 8, 8]
    HISTOGRAM_RANGES = [0, 256, 0, 256, 0, 256]
    
    # 개선된 임베딩 설정
    MIN_FACE_SIZE_FOR_EMBEDDING = 20  # 더 작은 얼굴도 허용
    MIN_BRIGHTNESS = 20  # 더 어두운 이미지도 허용
    MAX_BRIGHTNESS = 235  # 더 밝은 이미지도 허용
    MIN_CONTRAST = 5  # 더 낮은 대비도 허용
    ENSEMBLE_ENABLED = True
    QUALITY_VALIDATION_ENABLED = True
    
    # 성능 최적화 설정
    USE_GPU_IF_AVAILABLE = True
    BATCH_PROCESSING_ENABLED = False  # 실시간 처리를 위해 비활성화
    CACHE_EMBEDDINGS = True  # 임베딩 캐싱 활성화
    
    # IoU 임계값 (다중 인물 최적화)
    IOU_THRESHOLD = 0.5  # 더 엄격한 중복 제거
    
    # HOG 사람 검출 설정 (다중 인물 최적화)
    HOG_WIN_STRIDE = (4, 4)        # 더 세밀한 검출
    HOG_PADDING = (16, 16)         # 줄어든 패딩
    HOG_SCALE = 1.02               # 더 세밀한 스케일
    HOG_HIT_THRESHOLD = 0.3        # 더 민감한 검출
    
    # YOLO 설정
    YOLO_CONFIDENCE_THRESHOLD = 0.4  # 더 많은 사람 검출
    YOLO_NMS_THRESHOLD = 0.4         # Non-Maximum Suppression
