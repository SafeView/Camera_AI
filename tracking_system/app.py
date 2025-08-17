"""
메인 애플리케이션 모듈
"""

import cv2
import numpy as np
from typing import List, Tuple
from .models import FaceDetection, TrackingState
from .face_tracker import FaceTracker
from .dependencies import Dependencies
from .config import Config
from .crowd_optimizer import CrowdOptimizer

class ClickTrackingApp:
    """메인 애플리케이션"""
    
    def __init__(self):
        # 의존성 확인 및 초기화
        self.dependencies = Dependencies()
        self.tracker = FaceTracker(self.dependencies)
        self.crowd_optimizer = CrowdOptimizer()
        self.cap = None
        
        # UI 상태
        self.window_name = Config.WINDOW_NAME
    
    def run(self):
        """메인 실행 루프"""
        self._print_welcome_message()
        
        # 초기화
        if not self._initialize_system():
            return
        
        # 메인 루프
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("🛑 키보드 인터럽트로 종료됩니다.")
        finally:
            self._cleanup()
    
    def _print_welcome_message(self):
        """환영 메시지 출력"""
        print("🚀 클릭 기반 특정 인물 트래킹 시스템 시작")
        print("=" * 60)
        print("사용 방법:")
        print("  1. 화면에서 얼굴을 클릭하면 자동으로 추적 시작")
        print("  2. 추적 중인 대상은 노란색 박스로 표시")
        print("  3. 타겟이 화면 밖으로 나가도 정보 유지, 돌아오면 자동 재추적")
        print("  4. 얼굴 임베딩이 Pinecone DB에 자동 저장")
        print("키 안내:")
        print("  'q' - 종료")
        print("  'c' - 타겟 완전 해제")
        print("  'f' - 추적 일시 중지")
        print("  's' - 통계 출력")
        print("=" * 60)
    
    def _initialize_system(self) -> bool:
        """시스템 초기화"""
        # 서버 세션 시작
        self.tracker.server_manager.start_session()
        
        # 카메라 초기화
        return self._initialize_camera()
    
    def _initialize_camera(self) -> bool:
        """카메라 초기화"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ 카메라를 열 수 없습니다.")
            return False
        
        # 윈도우 설정
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        return True
    
    def _main_loop(self):
        """메인 루프"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # 추적 업데이트
            tracking_result = self.tracker.update_tracking(frame)
            
            # 프레임 처리 및 표시
            processed_frame = self._process_frame(frame, tracking_result)
            cv2.imshow(self.window_name, processed_frame)
            
            # 키 입력 처리
            if self._handle_key_input():
                break
    
    def _process_frame(self, frame: np.ndarray, tracking_result) -> np.ndarray:
        """프레임 처리 및 UI 그리기"""
        # 얼굴과 사람 검출
        face_detections = self.tracker.face_detector.detect_faces(frame)
        person_detections = self.tracker.person_detector.detect_persons(frame)
        
        # 군중 환경 최적화 적용
        face_detections, person_detections = self.crowd_optimizer.optimize_for_crowd(
            frame, face_detections, person_detections
        )
        
        # 모든 검출된 얼굴에 회색 박스
        for detection in face_detections:
            self._draw_detection_box(frame, detection, (128, 128, 128), 1)
        
        # 모든 검출된 사람에 연한 파란색 박스
        for detection in person_detections:
            self._draw_detection_box(frame, detection, (255, 128, 64), 1)
        
        # 추적 중인 타겟 표시
        if tracking_result.is_tracking:
            self._draw_tracking_boxes(frame, tracking_result)
        
        # 상태 정보 표시 (군중 통계 포함)
        self._draw_status_info(frame, face_detections, person_detections, tracking_result)
        
        return frame
    
    def _draw_detection_box(self, frame: np.ndarray, detection: FaceDetection, color: Tuple[int, int, int], thickness: int):
        """검출된 얼굴 박스 그리기"""
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    def _draw_tracking_boxes(self, frame: np.ndarray, tracking_result):
        """추적 중인 타겟 박스들 그리기"""
        # 얼굴 박스 (노란색)
        if tracking_result.face_detection:
            detection = tracking_result.face_detection
            x, y, w, h = detection.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 3)
            
            # 타겟 이름과 유사도 표시
            label = f"{tracking_result.target_name}: {tracking_result.similarity:.3f}"
            cv2.putText(frame, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # 사람 전체 박스 (초록색)
        if tracking_result.person_detection:
            detection = tracking_result.person_detection
            x, y, w, h = detection.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 전신 라벨
            person_label = f"{tracking_result.target_name} (Full Body)"
            cv2.putText(frame, person_label, (x, y + h + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    def _draw_status_info(self, frame: np.ndarray, face_detections: List, person_detections: List, tracking_result):
        """상태 정보 표시"""
        # 검출 수
        cv2.putText(frame, f"Faces: {len(face_detections)}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Persons: {len(person_detections)}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 128, 64), 2)
        
        # 추적 상태
        status_text, color = self._get_status_display(tracking_result.state)
        if tracking_result.target_name:
            cv2.putText(frame, f"{status_text}: {tracking_result.target_name}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # 추적 모드 표시
            if tracking_result.person_detection and tracking_result.face_detection:
                mode = "Face + Body"
            elif tracking_result.person_detection:
                mode = "Body Only"
            elif tracking_result.face_detection:
                mode = "Face Only"
            else:
                mode = "None"
            
            cv2.putText(frame, f"Mode: {mode}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        else:
            cv2.putText(frame, "No Target Selected", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # 군중 통계 표시
        crowd_stats = self.crowd_optimizer.get_crowd_statistics()
        cv2.putText(frame, f"Density: {crowd_stats['crowd_density']}", (10, 150),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (64, 128, 255), 2)
    
    def _get_status_display(self, state: TrackingState) -> Tuple[str, Tuple[int, int, int]]:
        """상태별 표시 텍스트와 색상 반환"""
        status_map = {
            TrackingState.TRACKING: ("Tracking ✓", (0, 255, 255)),
            TrackingState.SEARCHING: ("Searching 🔍", (0, 255, 255)),
            TrackingState.SUSPENDED: ("Suspended ⏸️", (0, 165, 255)),
            TrackingState.NO_TARGET: ("No Target", (128, 128, 128))
        }
        return status_map.get(state, ("Unknown", (128, 128, 128)))
    
    def _mouse_callback(self, event, x, y, flags, param):
        """마우스 클릭 이벤트 처리"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 현재 프레임 획득
            ret, frame = self.cap.read()
            if ret:
                self.tracker.start_tracking(frame, (x, y))
    
    def _handle_key_input(self) -> bool:
        """키 입력 처리"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return True
        elif key == ord('c'):
            self.tracker.clear_target()
        elif key == ord('f'):
            self.tracker.suspend_tracking()
        elif key == ord('s'):
            self._print_statistics()
        
        return False
    
    def _print_statistics(self):
        """통계 정보 출력"""
        stats = self.tracker.get_statistics()
        print("📊 현재 시스템 상태:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    def _cleanup(self):
        """정리 작업"""
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # 서버 세션 종료
        self.tracker.server_manager.stop_session()
