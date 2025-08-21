"""
서버 연동 관리 모듈
"""

import os
import cv2
import requests
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
from .models import TrackingTarget
from .config import Config

class ServerManager:
    """서버 연동 관리"""
    
    def __init__(self, server_url: str = None):
        self.server_url = server_url or Config.SERVER_URL
        self.session_id = None
        self.timeout = Config.SERVER_TIMEOUT
    
    def start_session(self, session_name: str = None) -> bool:
        """추적 세션 시작"""
        try:
            if session_name is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                session_name = f"session_{timestamp}_click_tracking"
            
            response = requests.post(
                f"{self.server_url}/tracking/start", 
                params={"session_name": session_name}, 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                self.session_id = result.get("session_id", session_name)
                return True
            else:
                return False
                
        except Exception as e:
            return False
    
    def stop_session(self) -> Optional[Dict[str, Any]]:
        """추적 세션 종료"""
        if not self.session_id:
            return None
        
        try:
            response = requests.post(
                f"{self.server_url}/tracking/stop", 
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                self.session_id = None
                return result
            else:
                return None
                
        except Exception as e:
            return None
    
    def add_target(self, target: TrackingTarget, face_image: np.ndarray) -> bool:
        """서버에 타겟 추가"""
        try:
            # 임시 파일로 이미지 저장
            temp_filename = f"temp_{target.name}_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(temp_filename, face_image)
            
            try:
                with open(temp_filename, 'rb') as f:
                    files = {"image": (temp_filename, f, "image/jpeg")}
                    data = {"name": target.name}
                    
                    response = requests.post(
                        f"{self.server_url}/targets/add", 
                        files=files, 
                        data=data, 
                        timeout=self.timeout
                    )
                
                if response.status_code == 200:
                    return True
                else:
                    return False
                    
            finally:
                # 임시 파일 삭제
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
                    
        except Exception as e:
            return False
    
    def get_targets_list(self) -> Optional[list]:
        """등록된 타겟 목록 조회"""
        try:
            response = requests.get(
                f"{self.server_url}/targets/list",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json().get("targets", [])
            else:
                return None
                
        except Exception as e:
            return None
    
    def get_live_statistics(self) -> Optional[Dict[str, Any]]:
        """실시간 통계 조회"""
        try:
            response = requests.get(
                f"{self.server_url}/statistics/live",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
                
        except Exception as e:
            return None
    
    def remove_target(self, target_name: str) -> bool:
        """타겟 제거"""
        try:
            response = requests.delete(
                f"{self.server_url}/targets/{target_name}",
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                return True
            else:
                return False
                
        except Exception as e:
            return False
    
    def is_server_available(self) -> bool:
        """서버 가용성 확인"""
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=2
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_session_id(self) -> Optional[str]:
        """현재 세션 ID 반환"""
        return self.session_id
