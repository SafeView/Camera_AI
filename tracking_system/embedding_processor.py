"""
개선된 얼굴 임베딩 처리 모듈
"""

import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from .models import FaceDetection, FaceEmbedder
from .config import Config
from .dependencies import Dependencies

class EnhancedInsightFaceEmbedder:
    """개선된 InsightFace 기반 임베딩"""
    
    def __init__(self):
        from insightface.app import FaceAnalysis
        self.model = FaceAnalysis(providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=0, det_size=Config.INSIGHT_FACE_DET_SIZE)
    
    def compute_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 ROI에서 임베딩 추출 (품질 검증 포함)"""
        try:
            # 전처리: 노이즈 제거 및 대비 향상
            processed_roi = self._preprocess_face(face_roi)
            
            img_rgb = cv2.cvtColor(processed_roi, cv2.COLOR_BGR2RGB)
            faces = self.model.get(img_rgb)
            
            if faces:
                embedding = faces[0].normed_embedding.astype(np.float32)
                
                # 품질 검증
                if self._validate_embedding_quality(embedding, face_roi):
                    return embedding
                    
        except Exception as e:
            print(f"InsightFace 임베딩 오류: {e}")
        return None
    
    def _preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """얼굴 이미지 전처리"""
        # 크기 정규화 (최소 112x112)
        min_size = 112
        h, w = face_roi.shape[:2]
        
        if h < min_size or w < min_size:
            scale = max(min_size / h, min_size / w)
            new_h, new_w = int(h * scale), int(w * scale)
            face_roi = cv2.resize(face_roi, (new_w, new_h))
        
        # 가우시안 블러로 노이즈 제거
        face_roi = cv2.GaussianBlur(face_roi, (3, 3), 0)
        
        # 히스토그램 평활화로 대비 향상
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])
        face_roi = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return face_roi
    
    def _validate_embedding_quality(self, embedding: np.ndarray, face_roi: np.ndarray) -> bool:
        """임베딩 품질 검증"""
        # 임베딩 차원 검증
        if len(embedding) < 100:
            return False
        
        # 임베딩 정규화 검증
        norm = np.linalg.norm(embedding)
        if norm < 0.1 or norm > 10.0:
            return False
        
        # 얼굴 크기 검증
        h, w = face_roi.shape[:2]
        if h < 30 or w < 30:
            return False
        
        return True

class EnhancedDeepFaceEmbedder:
    """개선된 DeepFace 기반 임베딩"""
    
    def __init__(self):
        pass
    
    def compute_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 ROI에서 임베딩 추출 (품질 검증 포함)"""
        try:
            # 전처리
            processed_roi = self._preprocess_face(face_roi)
            
            from deepface import DeepFace
            rep = DeepFace.represent(
                img_path=processed_roi, 
                model_name='ArcFace', 
                enforce_detection=False,
                detector_backend='opencv'
            )
            
            if isinstance(rep, list) and len(rep) > 0 and 'embedding' in rep[0]:
                embedding = np.array(rep[0]['embedding'], dtype=np.float32)
                
                # 품질 검증
                if self._validate_embedding_quality(embedding, face_roi):
                    return embedding
                    
        except Exception as e:
            print(f"DeepFace 임베딩 오류: {e}")
        return None
    
    def _preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """얼굴 이미지 전처리"""
        # 크기 정규화
        face_roi = cv2.resize(face_roi, (224, 224))
        
        # 노이즈 제거
        face_roi = cv2.medianBlur(face_roi, 3)
        
        return face_roi
    
    def _validate_embedding_quality(self, embedding: np.ndarray, face_roi: np.ndarray) -> bool:
        """임베딩 품질 검증"""
        if len(embedding) < 100:
            return False
        
        norm = np.linalg.norm(embedding)
        if norm < 0.1 or norm > 10.0:
            return False
        
        return True

class EnhancedHistogramEmbedder:
    """개선된 히스토그램 기반 임베딩"""
    
    def __init__(self):
        pass
    
    def compute_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 ROI에서 임베딩 추출 (다중 채널 히스토그램)"""
        try:
            # 전처리
            processed_roi = self._preprocess_face(face_roi)
            
            # 다중 채널 히스토그램 계산
            embedding = self._compute_multi_channel_histogram(processed_roi)
            
            if embedding is not None and self._validate_embedding_quality(embedding, face_roi):
                return embedding
                
        except Exception as e:
            print(f"히스토그램 임베딩 오류: {e}")
        return None
    
    def _preprocess_face(self, face_roi: np.ndarray) -> np.ndarray:
        """얼굴 이미지 전처리"""
        # 크기 정규화
        roi = cv2.resize(face_roi, Config.HISTOGRAM_SIZE)
        
        # 노이즈 제거
        roi = cv2.GaussianBlur(roi, (3, 3), 0)
        
        return roi
    
    def _compute_multi_channel_histogram(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """다중 채널 히스토그램 계산"""
        try:
            # BGR 히스토그램
            bgr_hist = cv2.calcHist([roi], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            bgr_hist = bgr_hist.flatten()
            
            # HSV 히스토그램
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hsv_hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            hsv_hist = hsv_hist.flatten()
            
            # 그레이스케일 히스토그램
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            gray_hist = cv2.calcHist([gray_roi], [0], None, [64], [0, 256])
            gray_hist = gray_hist.flatten()
            
            # 히스토그램 결합
            combined_hist = np.concatenate([bgr_hist, hsv_hist, gray_hist])
            
            # 정규화
            s = combined_hist.sum()
            if s > 0:
                combined_hist = combined_hist / s
                return combined_hist.astype(np.float32)
                
        except Exception as e:
            print(f"히스토그램 계산 오류: {e}")
        
        return None
    
    def _validate_embedding_quality(self, embedding: np.ndarray, face_roi: np.ndarray) -> bool:
        """임베딩 품질 검증"""
        if len(embedding) < 100:
            return False
        
        # 히스토그램 다양성 검증
        unique_values = len(np.unique(embedding))
        if unique_values < 10:
            return False
        
        return True

class EnsembleEmbedder:
    """앙상블 임베딩 (여러 모델 결합)"""
    
    def __init__(self, embedders: List[FaceEmbedder]):
        self.embedders = embedders
    
    def compute_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """앙상블 임베딩 계산"""
        embeddings = []
        
        for embedder in self.embedders:
            try:
                embedding = embedder.compute_embedding(face_roi)
                if embedding is not None:
                    embeddings.append(embedding)
            except Exception as e:
                print(f"앙상블 임베딩 오류: {e}")
        
        if not embeddings:
            return None
        
        # 앙상블 평균 계산
        ensemble_embedding = np.mean(embeddings, axis=0)
        
        # 정규화
        norm = np.linalg.norm(ensemble_embedding)
        if norm > 0:
            ensemble_embedding = ensemble_embedding / norm
        
        return ensemble_embedding.astype(np.float32)

class EnhancedEmbeddingProcessor:
    """개선된 임베딩 처리기"""
    
    def __init__(self, dependencies: Dependencies):
        self.deps = dependencies
        self.embedders = self._create_embedders()
        self.ensemble_embedder = EnsembleEmbedder(self.embedders) if len(self.embedders) > 1 else None
        self.primary_embedder = self.embedders[0] if self.embedders else None
    
    def _create_embedders(self) -> List[FaceEmbedder]:
        """사용 가능한 임베딩 모델들 생성"""
        embedders = []
        
        # InsightFace 시도
        try:
            embedders.append(EnhancedInsightFaceEmbedder())
            print("✅ InsightFace 임베딩 모델 로드 성공")
        except Exception as e:
            print(f"❌ InsightFace 로드 실패: {e}")
        
        # DeepFace 시도
        try:
            embedders.append(EnhancedDeepFaceEmbedder())
            print("✅ DeepFace 임베딩 모델 로드 성공")
        except Exception as e:
            print(f"❌ DeepFace 로드 실패: {e}")
            # DeepFace가 실패해도 계속 진행 (히스토그램으로 폴백)
        
        # 히스토그램 (항상 사용 가능)
        embedders.append(EnhancedHistogramEmbedder())
        print("✅ 히스토그램 임베딩 모델 로드 성공")
        
        return embedders
    
    def compute_face_embedding(self, frame: np.ndarray, detection: FaceDetection) -> Optional[np.ndarray]:
        """얼굴 임베딩 계산 (개선된 버전)"""
        x, y, w, h = detection.bbox
        
        # ROI 추출 및 검증
        roi = frame[y:y+h, x:x+w]
        if roi.size == 0:
            return None
        
        # ROI 품질 검증
        if not self._validate_roi_quality(roi):
            print("⚠️ ROI 품질이 낮습니다")
            return None
        
        # 앙상블 임베딩 계산 (여러 모델이 있는 경우)
        if self.ensemble_embedder:
            embedding = self.ensemble_embedder.compute_embedding(roi)
        else:
            # 단일 모델 사용
            embedding = self.primary_embedder.compute_embedding(roi) if self.primary_embedder else None
        
        if embedding is None:
            return None
        
        # 차원 정규화 및 품질 검증
        normalized_embedding = self._normalize_embedding(embedding)
        
        if self._validate_final_embedding(normalized_embedding):
            return normalized_embedding
        
        return None
    
    def _validate_roi_quality(self, roi: np.ndarray) -> bool:
        """ROI 품질 검증"""
        h, w = roi.shape[:2]
        
        # 크기 검증
        if h < 20 or w < 20:
            return False
        
        # 밝기 검증
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 30 or mean_brightness > 220:
            return False
        
        # 대비 검증
        std_brightness = np.std(gray)
        if std_brightness < 10:
            return False
        
        return True
    
    def _normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """임베딩 정규화"""
        current_dim = len(embedding)
        target_dim = Config.EMBEDDING_DIMENSION
        
        # 차원 조정
        if current_dim < target_dim:
            normalized_embedding = np.zeros(target_dim, dtype=np.float32)
            normalized_embedding[:current_dim] = embedding
            embedding = normalized_embedding
        elif current_dim > target_dim:
            embedding = embedding[:target_dim]
        
        # 정규화
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        
        return embedding
    
    def _validate_final_embedding(self, embedding: np.ndarray) -> bool:
        """최종 임베딩 품질 검증"""
        # 차원 검증
        if len(embedding) != Config.EMBEDDING_DIMENSION:
            return False
        
        # 정규화 검증
        norm = np.linalg.norm(embedding)
        if abs(norm - 1.0) > 0.1:
            return False
        
        # NaN/Inf 검증
        if np.any(np.isnan(embedding)) or np.any(np.isinf(embedding)):
            return False
        
        return True
    
    def compute_person_embedding(self, person_roi: np.ndarray) -> Optional[np.ndarray]:
        """사람 전체 임베딩 계산 (개선된 버전)"""
        try:
            if person_roi.size == 0:
                return None
            
            # 크기 정규화
            roi = cv2.resize(person_roi, (256, 256))
            
            # 다중 특징 추출
            features = []
            
            # 1. 색상 히스토그램
            color_hist = self._compute_color_histogram(roi)
            if color_hist is not None:
                features.append(color_hist)
            
            # 2. HOG 특징
            hog_features = self._compute_hog_features(roi)
            if hog_features is not None:
                features.append(hog_features)
            
            # 3. 텍스처 특징
            texture_features = self._compute_texture_features(roi)
            if texture_features is not None:
                features.append(texture_features)
            
            if not features:
                return None
            
            # 특징 결합
            combined_features = np.concatenate(features)
            
            # 정규화
            normalized_features = self._normalize_embedding(combined_features)
            
            return normalized_features
            
        except Exception as e:
            print(f"사람 임베딩 계산 오류: {e}")
            return None
    
    def _compute_color_histogram(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """색상 히스토그램 계산"""
        try:
            # HSV 히스토그램
            hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv_roi], [0, 1, 2], None, [12, 12, 12], [0, 180, 0, 256, 0, 256])
            hist = hist.flatten()
            
            # 정규화
            s = hist.sum()
            if s > 0:
                return (hist / s).astype(np.float32)
        except Exception:
            pass
        return None
    
    def _compute_hog_features(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """HOG 특징 계산"""
        try:
            from skimage.feature import hog
            from skimage import data, exposure
            
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            features = hog(gray, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))
            return features.astype(np.float32)
        except Exception:
            pass
        return None
    
    def _compute_texture_features(self, roi: np.ndarray) -> Optional[np.ndarray]:
        """텍스처 특징 계산"""
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # GLCM 특징 (간단한 버전)
            features = []
            
            # 대비, 상관관계, 에너지 등
            for i in range(0, gray.shape[0]-1, 32):
                for j in range(0, gray.shape[1]-1, 32):
                    patch = gray[i:i+32, j:j+32]
                    if patch.size > 0:
                        features.extend([np.mean(patch), np.std(patch)])
            
            if features:
                return np.array(features, dtype=np.float32)
        except Exception:
            pass
        return None
    
    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """코사인 유사도 계산 (개선된 버전)"""
        try:
            dot_product = np.dot(a, b)
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
            
            similarity = dot_product / (norm_a * norm_b)
            
            # 유효성 검증
            if np.isnan(similarity) or np.isinf(similarity):
                return 0.0
            
            return max(0.0, min(1.0, similarity))  # 0-1 범위로 제한
            
        except Exception as e:
            print(f"유사도 계산 오류: {e}")
            return 0.0
    
    def get_embedding_info(self) -> Dict[str, Any]:
        """임베딩 모델 정보 반환"""
        return {
            "available_models": [type(emb).__name__ for emb in self.embedders],
            "ensemble_enabled": self.ensemble_embedder is not None,
            "primary_model": type(self.primary_embedder).__name__ if self.primary_embedder else None,
            "embedding_dimension": Config.EMBEDDING_DIMENSION
        }

# 기존 클래스들 (하위 호환성)
class InsightFaceEmbedder(EnhancedInsightFaceEmbedder):
    pass

class DeepFaceEmbedder(EnhancedDeepFaceEmbedder):
    pass

class HistogramEmbedder(EnhancedHistogramEmbedder):
    pass

class EmbeddingProcessor(EnhancedEmbeddingProcessor):
    pass
