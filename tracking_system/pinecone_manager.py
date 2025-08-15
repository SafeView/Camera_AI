"""
Pinecone 벡터 데이터베이스 관리 모듈
"""

import uuid
import numpy as np
from datetime import datetime
from typing import Tuple, Optional
from .models import TrackingTarget, FaceDetection
from .config import Config
from .dependencies import Dependencies

class PineconeManager:
    """Pinecone 벡터 DB 관리"""
    
    def __init__(self, dependencies: Dependencies):
        self.deps = dependencies
        self.client = None
        self.index = None
        
        if self.deps.pinecone_available:
            self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Pinecone 초기화"""
        try:
            from pinecone import Pinecone, ServerlessSpec
            
            self.client = Pinecone(api_key=Config.PINECONE_API_KEY)
            
            # 인덱스 연결/생성
            index_name = Config.PINECONE_INDEX_NAME
            existing_indexes = [idx.name for idx in self.client.list_indexes()]
            
            if index_name not in existing_indexes:
                self.client.create_index(
                    name=index_name,
                    dimension=Config.EMBEDDING_DIMENSION,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=Config.PINECONE_CLOUD, 
                        region=Config.PINECONE_REGION
                    )
                )
            
            self.index = self.client.Index(index_name)
            
        except Exception as e:
            self.client = None
            self.index = None
    
    def save_embedding(self, target: TrackingTarget, detection: FaceDetection, frame_shape: Tuple[int, int]) -> bool:
        """임베딩을 Pinecone에 저장"""
        if not self.index:
            return False
        
        try:
            face_id = f"{target.name}_{uuid.uuid4().hex[:8]}"
            
            metadata = self._create_metadata(target, detection, frame_shape)
            
            self.index.upsert(vectors=[(face_id, target.embedding.tolist(), metadata)])
            return True
            
        except Exception as e:
            return False
    
    def _create_metadata(self, target: TrackingTarget, detection: FaceDetection, frame_shape: Tuple[int, int]) -> dict:
        """메타데이터 생성"""
        return {
            "person_name": str(target.name),
            "timestamp": datetime.now().isoformat(),
            "bbox_x": int(detection.x),
            "bbox_y": int(detection.y), 
            "bbox_w": int(detection.w),
            "bbox_h": int(detection.h),
            "frame_width": int(frame_shape[1]),
            "frame_height": int(frame_shape[0]),
            "confidence": float(detection.confidence),
            "source": "click_tracking_system"
        }
    
    def search_similar_faces(self, embedding: np.ndarray, top_k: int = 5, threshold: float = 0.8) -> list:
        """유사한 얼굴 검색"""
        if not self.index:
            return []
        
        try:
            results = self.index.query(
                vector=embedding.tolist(),
                top_k=top_k,
                include_metadata=True,
                include_values=False
            )
            
            # 임계값 이상의 결과만 반환
            filtered_results = []
            for match in results.matches:
                if match.score >= threshold:
                    filtered_results.append({
                        'id': match.id,
                        'score': match.score,
                        'metadata': match.metadata
                    })
            
            return filtered_results
            
        except Exception as e:
            return []
    
    def delete_face(self, face_id: str) -> bool:
        """특정 얼굴 데이터 삭제"""
        if not self.index:
            return False
        
        try:
            self.index.delete(ids=[face_id])
            return True
        except Exception as e:
            return False
    
    def get_index_stats(self) -> Optional[dict]:
        """인덱스 통계 정보 조회"""
        if not self.index:
            return None
        
        try:
            stats = self.index.describe_index_stats()
            return stats
        except Exception as e:
            return None
    
    def is_available(self) -> bool:
        """Pinecone 사용 가능 여부"""
        return self.index is not None
