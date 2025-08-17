"""
저장된 사람 관리 모듈
"""

import os
import json
import numpy as np
import cv2
from typing import List, Optional, Dict, Any
from datetime import datetime
from .models import StoredPerson, RecognitionResult, FaceDetection
from .embedding_processor import EmbeddingProcessor
from .dependencies import Dependencies


class StoredPersonManager:
    """저장된 사람 관리자"""
    
    def __init__(self, storage_dir: str = "stored_persons"):
        self.storage_dir = storage_dir
        self.persons: Dict[str, StoredPerson] = {}
        self.embedding_processor = EmbeddingProcessor(Dependencies())
        
        # 저장 디렉토리 생성
        os.makedirs(storage_dir, exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(storage_dir, "embeddings"), exist_ok=True)
        
        # 저장된 사람들 로드
        self._load_stored_persons()
    
    def add_person(self, name: str, face_image: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """새로운 사람 추가"""
        try:
            # 얼굴 임베딩 계산
            embedding = self._compute_face_embedding(face_image)
            if embedding is None:
                return False
            
            # 고유 ID 생성
            person_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # 이미지 저장
            image_path = os.path.join(self.storage_dir, "images", f"{person_id}.jpg")
            cv2.imwrite(image_path, face_image)
            
            # 임베딩 저장
            embedding_path = os.path.join(self.storage_dir, "embeddings", f"{person_id}.npy")
            np.save(embedding_path, embedding)
            
            # 메타데이터 저장
            meta_path = os.path.join(self.storage_dir, "embeddings", f"{person_id}.json")
            meta_data = {
                "id": person_id,
                "name": name,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, ensure_ascii=False, indent=2)
            
            # 메모리에 추가
            person = StoredPerson(
                id=person_id,
                name=name,
                embedding=embedding,
                face_image=face_image,
                metadata=metadata or {}
            )
            self.persons[person_id] = person
            
            return True
            
        except Exception as e:
            return False
    
    def remove_person(self, person_id: str) -> bool:
        """사람 제거"""
        try:
            if person_id not in self.persons:
                return False
            
            # 파일 삭제
            image_path = os.path.join(self.storage_dir, "images", f"{person_id}.jpg")
            embedding_path = os.path.join(self.storage_dir, "embeddings", f"{person_id}.npy")
            meta_path = os.path.join(self.storage_dir, "embeddings", f"{person_id}.json")
            
            for path in [image_path, embedding_path, meta_path]:
                if os.path.exists(path):
                    os.remove(path)
            
            # 메모리에서 제거
            del self.persons[person_id]
            
            return True
            
        except Exception as e:
            return False
    
    def recognize_person(self, face_detection: FaceDetection, frame: np.ndarray, 
                        threshold: float = 0.8) -> Optional[RecognitionResult]:
        """사람 인식"""
        try:
            # 얼굴 ROI 추출
            x, y, w, h = face_detection.bbox
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                return None
            
            # 얼굴 임베딩 계산
            embedding = self._compute_face_embedding(face_roi)
            if embedding is None:
                return None
            
            # 모든 저장된 사람들과 비교
            best_match = None
            best_similarity = 0.0
            
            for person in self.persons.values():
                similarity = EmbeddingProcessor.cosine_similarity(embedding, person.embedding)
                
                if similarity > best_similarity and similarity >= threshold:
                    best_similarity = similarity
                    best_match = person
            
            if best_match:
                return RecognitionResult(
                    person=best_match,
                    similarity=best_similarity,
                    face_detection=face_detection
                )
            
            return None
            
        except Exception as e:
            return None
    
    def get_all_persons(self) -> List[StoredPerson]:
        """모든 저장된 사람 반환"""
        return list(self.persons.values())
    
    def get_person_by_id(self, person_id: str) -> Optional[StoredPerson]:
        """ID로 사람 조회"""
        return self.persons.get(person_id)
    
    def get_person_by_name(self, name: str) -> List[StoredPerson]:
        """이름으로 사람 조회"""
        return [person for person in self.persons.values() if person.name == name]
    
    def _compute_face_embedding(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """얼굴 임베딩 계산"""
        try:
            # 크기 정규화
            face_roi = cv2.resize(face_roi, (128, 128))
            
            # 임베딩 계산
            embedding = self.embedding_processor.embedder.compute_embedding(face_roi)
            if embedding is None:
                return None
            
            # 1024차원으로 패딩
            return self.embedding_processor._pad_to_target_dimension(embedding)
            
        except Exception as e:
            return None
    
    def _load_stored_persons(self):
        """저장된 사람들 로드"""
        try:
            embeddings_dir = os.path.join(self.storage_dir, "embeddings")
            
            for filename in os.listdir(embeddings_dir):
                if filename.endswith('.json'):
                    person_id = filename[:-5]  # .json 제거
                    
                    # 메타데이터 로드
                    meta_path = os.path.join(embeddings_dir, filename)
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta_data = json.load(f)
                    
                    # 임베딩 로드
                    embedding_path = os.path.join(embeddings_dir, f"{person_id}.npy")
                    if os.path.exists(embedding_path):
                        embedding = np.load(embedding_path)
                        
                        # 이미지 로드
                        image_path = os.path.join(self.storage_dir, "images", f"{person_id}.jpg")
                        face_image = None
                        if os.path.exists(image_path):
                            face_image = cv2.imread(image_path)
                        
                        # StoredPerson 객체 생성
                        person = StoredPerson(
                            id=person_id,
                            name=meta_data["name"],
                            embedding=embedding,
                            face_image=face_image,
                            metadata=meta_data.get("metadata", {}),
                            created_at=meta_data.get("created_at", "")
                        )
                        
                        self.persons[person_id] = person
                        
        except Exception as e:
            pass
    
    def get_statistics(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            "total_persons": len(self.persons),
            "storage_dir": self.storage_dir,
            "person_names": list(set(person.name for person in self.persons.values()))
        }

