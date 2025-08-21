"""
외부 라이브러리 의존성 관리 모듈
"""

class Dependencies:
    """외부 라이브러리 의존성 확인 및 관리"""
    
    def __init__(self):
        self.pinecone_available = self._check_pinecone()
        self.insight_available = self._check_insight()
        self.deepface_available = self._check_deepface()
        self.mediapipe_available = self._check_mediapipe()
    
    def _check_pinecone(self) -> bool:
        """Pinecone 라이브러리 확인"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            return True
        except ImportError:
            return False
    
    def _check_insight(self) -> bool:
        """InsightFace 라이브러리 확인"""
        try:
            from insightface.app import FaceAnalysis
            return True
        except Exception:
            return False
    
    def _check_deepface(self) -> bool:
        """DeepFace 라이브러리 확인"""
        try:
            from deepface import DeepFace
            return True
        except ImportError:
            return False
        except Exception as e:
            # NumPy 호환성 문제 등으로 인한 오류
            print(f"⚠️ DeepFace 로드 중 오류: {e}")
            return False
    
    def _check_mediapipe(self) -> bool:
        """MediaPipe 라이브러리 확인"""
        try:
            import mediapipe as mp
            return True
        except ImportError:
            return False
    

    
    def get_best_embedding_model(self) -> str:
        """사용 가능한 최고 품질의 임베딩 모델 반환"""
        if self.insight_available:
            return "insightface"
        elif self.deepface_available:
            return "deepface"
        else:
            return "histogram"
    
    def has_vector_db(self) -> bool:
        """벡터 DB 사용 가능 여부"""
        return self.pinecone_available
    
    def has_advanced_detection(self) -> bool:
        """고급 얼굴 검출 사용 가능 여부"""
        return self.mediapipe_available
