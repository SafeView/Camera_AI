"""
녹화된 영상에서 사람 추적 시스템
"""

import cv2
import numpy as np
import time
import threading
import queue
from typing import Optional, Tuple
from .face_tracker import FaceTracker
from .dependencies import Dependencies
from .models import TrackingResult, TrackingState


class VideoTracker:
    """녹화된 영상에서 사람 추적 시스템"""
    
    def __init__(self):
        self.deps = Dependencies()
        self.tracker = FaceTracker(self.deps)
        self.cap = None
        self.current_frame = None
        self.is_playing = True
        self.current_frame_number = 0
        self.total_frames = 0
        self.fps = 30
        
        # 성능 최적화 설정 - 더 강력하게
        self.skip_frames = 4  # 4프레임마다 1프레임만 처리 (더 적극적)
        self.frame_counter = 0
        self.last_detection_time = 0
        self.detection_interval = 0.1  # 0.1초마다 검출 (더 자주)
        
        # 추적 시작 시 성능 최적화 - 더 오래 지속
        self.tracking_start_time = 0
        self.is_tracking_starting = False
        self.tracking_start_interval = 1.0  # 추적 시작 후 1초간 검출 간격 늘리기
        
        # UI 상태
        self.window_name = "Video Tracking System"
        self.click_point = None
        
        # 성능 모니터링
        self.frame_times = []
        self.avg_fps = 0
        
        # 백그라운드 작업 큐
        self.background_queue = queue.Queue()
        self.background_thread = None
        self.start_background_worker()
        
        # 검출 결과 캐싱 - 더 오래 유지
        self.detection_cache = {}
        self.cache_timeout = 1.0  # 1초 캐시 유효시간 (더 오래)
        
        # 추적 결과 캐싱
        self.tracking_cache = {}
        self.tracking_cache_timeout = 0.5
        
        # 프레임 해상도 조절
        self.target_width = 640  # 목표 너비
        self.target_height = 480  # 목표 높이
        self.resize_enabled = True
        
        # 검출 비활성화 모드
        self.detection_disabled = False  # 초기에는 검출 활성화
        self.detection_disable_duration = 1.0  # 1초간 검출 비활성화
        
        # 엄격한 타겟 모드 - 다른 사람 자동 지정 방지
        self.strict_target_mode = True
        self.original_target_name = None
        
        # Pinecone 저장 관련 - 한 번만 저장
        self.pinecone_saved = False  # 이미 저장되었는지 확인
        self.saved_target_name = None  # 저장된 타겟 이름
        
        # 타겟 카운터 - 고유한 이름 생성용
        self.target_counter = 0
    
    def start_background_worker(self):
        """백그라운드 작업자 스레드 시작"""
        def background_worker():
            while True:
                try:
                    task = self.background_queue.get(timeout=1.0)
                    if task is None:  # 종료 신호
                        break
                    
                    task_type, data = task
                    if task_type == "start_tracking":
                        self._execute_tracking_start(data)
                    elif task_type == "save_to_pinecone":
                        self._execute_pinecone_save(data)
                    
                    self.background_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"❌ 백그라운드 작업 오류: {e}")
        
        self.background_thread = threading.Thread(target=background_worker, daemon=True)
        self.background_thread.start()
    
    def load_video(self, video_path: str) -> bool:
        """비디오 파일 로드"""
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            print(f"❌ 영상을 열 수 없습니다: {video_path}")
            return False
        
        # 비디오 정보 가져오기
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        print(f"✅ 영상 로드 완료: {self.total_frames} 프레임, {self.fps:.2f} FPS")
        
        # 윈도우 설정
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)
        
        return True
    
    def run(self):
        """메인 실행 루프"""
        self._print_welcome_message()
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            print("🛑 키보드 인터럽트로 종료됩니다.")
        finally:
            self._cleanup()
    
    def _print_welcome_message(self):
        """환영 메시지 출력"""
        print("🎬 녹화된 영상 추적 시스템 시작")
        print("=" * 50)
        print("사용 방법:")
        print("  1. 영상에서 얼굴을 클릭하면 자동으로 추적 시작")
        print("  2. 추적 중인 대상은 노란색 박스로 표시")
        print("키 안내:")
        print("  'q' - 종료")
        print("  'c' - 타겟 완전 해제")
        print("  'p' - 재생/일시정지")
        print("  'r' - 처음으로 되돌리기")
        print("  '1-9' - 프레임 스킵 조절")
        print("  'd' - 검출 모드 토글")
        print("  's' - 해상도 조절 토글")
        print("  'f' - 강제 검출 모드 활성화")
        print("  'n' - 새로운 타겟 선택 (현재 타겟 해제)")
        print("  'h' - 도움말 표시")
        print("=" * 50)
    
    def _main_loop(self):
        """메인 루프"""
        while True:
            loop_start = time.time()
            
            if self.is_playing:
                ret, frame = self.cap.read()
                if not ret:
                    # 영상 끝에 도달하면 처음으로 되돌리기
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.current_frame_number = 0
                    continue
                
                # 프레임 크기 조절 (성능 향상)
                if self.resize_enabled:
                    frame = cv2.resize(frame, (self.target_width, self.target_height))
                
                self.current_frame = frame.copy()
                self.current_frame_number = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            else:
                # 일시정지 상태에서는 현재 프레임 유지
                if self.current_frame is None:
                    ret, frame = self.cap.read()
                    if ret:
                        if self.resize_enabled:
                            frame = cv2.resize(frame, (self.target_width, self.target_height))
                        self.current_frame = frame.copy()
                    else:
                        break
                frame = self.current_frame.copy()
            
            # 프레임 스킵 처리
            self.frame_counter += 1
            if self.frame_counter % self.skip_frames != 0:
                # 스킵된 프레임은 이전 추적 결과 사용
                if hasattr(self, 'last_tracking_result'):
                    tracking_result = self.last_tracking_result
                else:
                    tracking_result = self.tracker.update_tracking(frame)
            else:
                # 검출 비활성화 모드 체크
                if self.detection_disabled:
                    # 검출은 하지 않지만 추적은 계속 업데이트
                    tracking_result = self.tracker.update_tracking(frame)
                    self.last_tracking_result = tracking_result
                else:
                    # 검출 간격 체크
                    current_time = time.time()
                    detection_interval = self._get_detection_interval()
                    
                    if current_time - self.last_detection_time >= detection_interval:
                        tracking_result = self._get_cached_tracking_result(frame, current_time)
                        self.last_tracking_result = tracking_result
                        self.last_detection_time = current_time
                        
                        # 추적 시작 상태 업데이트
                        if self.is_tracking_starting:
                            if current_time - self.tracking_start_time >= self.tracking_start_interval:
                                self.is_tracking_starting = False
                                print("⚡ 추적 성능 최적화 완료")
                    else:
                        # 검출 간격 내에서는 이전 결과 사용
                        tracking_result = self.last_tracking_result if hasattr(self, 'last_tracking_result') else self.tracker.update_tracking(frame)
            
            # 엄격한 타겟 모드 필터링 적용
            tracking_result = self._filter_tracking_result(tracking_result)
            
            # 프레임 처리 및 표시
            processed_frame = self._process_frame(frame, tracking_result)
            cv2.imshow(self.window_name, processed_frame)
            
            # 키 입력 처리
            if self._handle_key_input():
                break
            
            # 성능 모니터링
            frame_time = time.time() - loop_start
            self.frame_times.append(frame_time)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)
            
            # 평균 FPS 계산
            if len(self.frame_times) > 0:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                self.avg_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # 프레임 속도 조절 - 더 빠르게
            target_delay = max(1, int(1000 / min(60, self.fps * 2)))  # 더 빠른 재생
            cv2.waitKey(target_delay)
    
    def _get_detection_interval(self) -> float:
        """현재 상황에 따른 검출 간격 반환"""
        if self.is_tracking_starting:
            return self.tracking_start_interval
        return self.detection_interval
    
    def _get_cached_tracking_result(self, frame: np.ndarray, current_time: float) -> TrackingResult:
        """캐시된 추적 결과 가져오기"""
        frame_hash = hash(frame.tobytes())
        
        if frame_hash in self.detection_cache:
            cache_time, cache_result = self.detection_cache[frame_hash]
            if current_time - cache_time < self.cache_timeout:
                return cache_result
        
        # 추적 결과 캐시도 확인
        if frame_hash in self.tracking_cache:
            cache_time, cache_result = self.tracking_cache[frame_hash]
            if current_time - cache_time < self.tracking_cache_timeout:
                return cache_result
        
        result = self.tracker.update_tracking(frame)
        
        # 두 캐시 모두에 저장
        self.detection_cache[frame_hash] = (current_time, result)
        self.tracking_cache[frame_hash] = (current_time, result)
        
        # 캐시 크기 제한
        if len(self.detection_cache) > 10:  # 더 작은 캐시
            oldest_key = min(self.detection_cache.keys(), 
                           key=lambda k: self.detection_cache[k][0])
            del self.detection_cache[oldest_key]
        
        if len(self.tracking_cache) > 10:
            oldest_key = min(self.tracking_cache.keys(), 
                           key=lambda k: self.tracking_cache[k][0])
            del self.tracking_cache[oldest_key]
        
        return result
    
    def _filter_tracking_result(self, tracking_result: TrackingResult) -> TrackingResult:
        """엄격한 타겟 모드에서 추적 결과 필터링"""
        if not self.strict_target_mode or not self.original_target_name:
            return tracking_result
        
        # 원래 타겟과 다른 사람이 추적되고 있다면 무시
        if (tracking_result.is_tracking and 
            tracking_result.target_name and 
            tracking_result.target_name != self.original_target_name):
            
            print(f"⚠️ 다른 사람이 감지됨: {tracking_result.target_name} (원래 타겟: {self.original_target_name})")
            print(f"⚠️ 유사도: {tracking_result.similarity:.3f} - 원래 타겟 유지")
            
            # 원래 타겟이 없으면 추적 중단
            return TrackingResult(
                is_tracking=False,
                state=TrackingState.NO_TARGET,
                target_name=self.original_target_name,
                face_detection=None,
                person_detection=None,
                similarity=0.0
            )
        
        # 원래 타겟이 추적되고 있다면 유사도가 낮아도 유지
        if (tracking_result.is_tracking and 
            tracking_result.target_name == self.original_target_name):
            
            # 유사도가 매우 낮아도 원래 타겟이면 유지
            if tracking_result.similarity < 0.3:  # 매우 낮은 유사도
                print(f"⚠️ 원래 타겟 유사도 낮음: {tracking_result.similarity:.3f} - 하지만 원래 타겟이므로 유지")
                # 유사도를 강제로 높게 설정
                tracking_result.similarity = 0.8
        
        return tracking_result
    
    def _process_frame(self, frame: np.ndarray, tracking_result: TrackingResult) -> np.ndarray:
        """프레임 처리 및 UI 그리기"""
        # 검출 비활성화 모드에서는 최소한의 처리만
        if self.detection_disabled:
            # 추적 중인 타겟만 표시 (박스는 항상 보이게)
            if tracking_result.is_tracking:
                self._draw_tracking_boxes(frame, tracking_result)
            
            # 기본 상태 정보만 표시
            self._draw_basic_status_info(frame, tracking_result)
            return frame
        
        current_time = time.time()
        detection_interval = self._get_detection_interval()
        
        if current_time - self.last_detection_time >= detection_interval:
            face_detections = self.tracker.face_detector.detect_faces(frame)
            person_detections = self.tracker.person_detector.detect_persons(frame)
            
            self.last_face_detections = face_detections
            self.last_person_detections = person_detections
        else:
            face_detections = getattr(self, 'last_face_detections', [])
            person_detections = getattr(self, 'last_person_detections', [])
        
        # 모든 검출된 얼굴에 회색 박스
        for detection in face_detections:
            self._draw_detection_box(frame, detection, (128, 128, 128), 1)
        
        # 모든 검출된 사람에 연한 파란색 박스
        for detection in person_detections:
            self._draw_detection_box(frame, detection, (255, 128, 64), 1)
        
        # 추적 중인 타겟 표시 (가장 중요한 부분 - 항상 표시)
        if tracking_result.is_tracking:
            self._draw_tracking_boxes(frame, tracking_result)
        
        # 상태 정보 표시
        self._draw_status_info(frame, face_detections, person_detections, tracking_result)
        
        return frame
    
    def _draw_detection_box(self, frame: np.ndarray, detection, color: Tuple[int, int, int], thickness: int):
        """검출된 얼굴 박스 그리기"""
        x, y, w, h = detection.bbox
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    
    def _draw_tracking_boxes(self, frame: np.ndarray, tracking_result: TrackingResult):
        """추적 중인 타겟 박스들 그리기"""
        # 전신 박스 (초록색) - 우선 표시
        if tracking_result.person_detection:
            detection = tracking_result.person_detection
            x, y, w, h = detection.bbox
            # 두꺼운 초록색 박스
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
            # 검은색 테두리 추가
            cv2.rectangle(frame, (x-1, y-1), (x + w + 1, y + h + 1), (0, 0, 0), 5)
            
            # 전신 라벨 - 더 큰 폰트
            person_label = f"{tracking_result.target_name} (전신 추적)"
            cv2.putText(frame, person_label, (x, y + h + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # 검은색 배경
            cv2.putText(frame, person_label, (x, y + h + 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # 초록색 텍스트
        
        # 얼굴 박스 (노란색) - 보조 표시
        if tracking_result.face_detection:
            detection = tracking_result.face_detection
            x, y, w, h = detection.bbox
            # 두꺼운 노란색 박스
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 4)
            # 검은색 테두리 추가
            cv2.rectangle(frame, (x-1, y-1), (x + w + 1, y + h + 1), (0, 0, 0), 6)
            
            # 타겟 이름과 유사도 표시 - 더 큰 폰트
            label = f"{tracking_result.target_name}: {tracking_result.similarity:.3f}"
            cv2.putText(frame, label, (x, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)  # 검은색 배경
            cv2.putText(frame, label, (x, y - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)  # 노란색 텍스트
    
    def _draw_basic_status_info(self, frame: np.ndarray, tracking_result: TrackingResult):
        """기본 상태 정보만 표시 (검출 비활성화 모드용)"""
        # 추적 상태 - 더 눈에 띄게
        if tracking_result.target_name:
            # 검은색 배경
            cv2.putText(frame, f"Tracking: {tracking_result.target_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
            # 노란색 텍스트
            cv2.putText(frame, f"Tracking: {tracking_result.target_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            cv2.putText(frame, "No Target", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # 성능 정보
        cv2.putText(frame, f"FPS: {self.avg_fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 검출 비활성화 표시 - 더 눈에 띄게
        cv2.putText(frame, "DETECTION DISABLED", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)  # 검은색 배경
        cv2.putText(frame, "DETECTION DISABLED", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)  # 주황색 텍스트
        
        # 추적 박스 안내
        if tracking_result.is_tracking:
            cv2.putText(frame, "TRACKING BOX ACTIVE", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)  # 검은색 배경
            cv2.putText(frame, "TRACKING BOX ACTIVE", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)  # 초록색 텍스트
    
    def _draw_status_info(self, frame: np.ndarray, face_detections, person_detections, tracking_result: TrackingResult):
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
            
            # 원래 타겟 정보 표시
            if self.original_target_name and self.original_target_name != tracking_result.target_name:
                cv2.putText(frame, f"Original: {self.original_target_name}", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            # 유사도 표시
            cv2.putText(frame, f"Similarity: {tracking_result.similarity:.3f}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "No Target Selected", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 2)
        
        # 프레임 정보
        current_time = self.current_frame_number / self.fps
        total_time = self.total_frames / self.fps
        cv2.putText(frame, f"Time: {current_time:.1f}s / {total_time:.1f}s", (10, 180),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 성능 정보
        cv2.putText(frame, f"FPS: {self.avg_fps:.1f}", (10, 210),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"Skip: {self.skip_frames}", (10, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # 추적 시작 상태 표시
        if self.is_tracking_starting:
            cv2.putText(frame, "OPTIMIZING...", (10, 270),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
        
        # 재생 상태
        play_status = "PLAYING" if self.is_playing else "PAUSED"
        y_pos = 300 if self.is_tracking_starting else 270
        cv2.putText(frame, play_status, (10, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.is_playing else (0, 165, 255), 2)
        
        # Strict Target Mode 상태 표시
        if self.strict_target_mode:
            cv2.putText(frame, "STRICT MODE", (10, y_pos + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
    
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
            print(f"🖱️ 클릭된 좌표: ({x}, {y})")
            
            if self.current_frame is not None:
                # 추적 시작 시 성능 최적화 모드 활성화
                self.is_tracking_starting = True
                self.tracking_start_time = time.time()
                
                # 검출 비활성화 모드 일시 해제 (검출을 위해)
                was_detection_disabled = self.detection_disabled
                self.detection_disabled = False
                print("⚡ 추적 시작 - 성능 최적화 모드 활성화")
                
                # 백그라운드에서 추적 시작
                self.background_queue.put(("start_tracking", {
                    "frame": self.current_frame.copy(),
                    "click_point": (x, y),
                    "restore_detection_disabled": was_detection_disabled
                }))
    
    def _execute_tracking_start(self, data):
        """백그라운드에서 추적 시작 실행"""
        try:
            frame = data["frame"]
            click_point = data["click_point"]
            restore_detection_disabled = data.get("restore_detection_disabled", False)
            
            print(f"🔍 추적 시작 시도 - 클릭 좌표: {click_point}")
            
            # 고유한 타겟 이름 생성
            self.target_counter += 1
            unique_target_name = f"Person_{self.target_counter}"
            
            # FaceTracker의 person_counter 설정 (고유한 이름을 위해)
            self.tracker.person_counter = self.target_counter - 1  # 0부터 시작하므로 -1
            
            # 디버깅: 검출 결과 확인
            face_detections = self.tracker.face_detector.detect_faces(frame)
            person_detections = self.tracker.person_detector.detect_persons(frame)
            
            print(f"🔍 검출된 얼굴 수: {len(face_detections)}")
            print(f"🔍 검출된 사람 수: {len(person_detections)}")
            
            # 검출이 전혀 되지 않는 경우 디버깅 정보
            if len(face_detections) == 0 and len(person_detections) == 0:
                print("⚠️ 검출이 전혀 되지 않음 - 가능한 원인:")
                print(f"  - 검출 모드: {'비활성화' if self.detection_disabled else '활성화'}")
                print(f"  - 프레임 크기: {frame.shape}")
                print(f"  - 리사이즈: {'활성화' if self.resize_enabled else '비활성화'}")
                if self.resize_enabled:
                    print(f"  - 목표 크기: {self.target_width}x{self.target_height}")
            
            # 클릭 위치가 검출된 영역 안에 있는지 확인
            click_x, click_y = click_point
            valid_face_detections = []
            valid_person_detections = []
            
            # 얼굴 검출 결과에서 클릭 위치가 영역 안에 있는지 확인
            for face in face_detections:
                fx, fy, fw, fh = face.bbox
                if fx <= click_x <= fx + fw and fy <= click_y <= fy + fh:
                    valid_face_detections.append(face)
                    print(f"✅ 클릭 위치가 얼굴 영역 안: {face.bbox}")
                else:
                    # 클릭 위치가 영역 밖이어도 가까우면 포함
                    face_center_x = fx + fw // 2
                    face_center_y = fy + fh // 2
                    distance = ((click_x - face_center_x) ** 2 + (click_y - face_center_y) ** 2) ** 0.5
                    if distance < 150:  # 150픽셀 이내
                        valid_face_detections.append(face)
                        print(f"✅ 클릭 위치 근처 얼굴: {face.bbox}, 거리: {distance:.1f}px")
                    else:
                        print(f"❌ 클릭 위치에서 멀리 떨어진 얼굴: {face.bbox}, 거리: {distance:.1f}px")
            
            # 사람 검출 결과에서 클릭 위치가 영역 안에 있는지 확인
            for person in person_detections:
                px, py, pw, ph = person.bbox
                if px <= click_x <= px + pw and py <= click_y <= py + ph:
                    valid_person_detections.append(person)
                    print(f"✅ 클릭 위치가 사람 영역 안: {person.bbox}")
                else:
                    # 클릭 위치가 영역 밖이어도 가까우면 포함
                    person_center_x = px + pw // 2
                    person_center_y = py + ph // 2
                    distance = ((click_x - person_center_x) ** 2 + (click_y - person_center_y) ** 2) ** 0.5
                    if distance < 200:  # 200픽셀 이내
                        valid_person_detections.append(person)
                        print(f"✅ 클릭 위치 근처 사람: {person.bbox}, 거리: {distance:.1f}px")
                    else:
                        print(f"❌ 클릭 위치에서 멀리 떨어진 사람: {person.bbox}, 거리: {distance:.1f}px")
            
            # 가장 가까운 얼굴과 사람 찾기
            closest_face = self.tracker.face_detector.find_closest_face(valid_face_detections, click_point)
            closest_person = self.tracker.person_detector.find_closest_person(valid_person_detections, click_point)
            
            # 전신 기반 추적 우선 (더 안정적)
            if closest_person:
                print(f"🎯 가장 가까운 사람 선택 (전신 기반): {closest_person.bbox}")
                # 전신 기반 추적을 위해 얼굴 정보도 함께 저장
                if closest_face:
                    print(f"🔍 전신과 함께 얼굴 정보도 저장: {closest_face.bbox}")
            elif closest_face:
                print(f"🎯 얼굴만 선택 (전신 없음): {closest_face.bbox}")
            else:
                print("❌ 클릭 위치 근처에 얼굴이나 사람이 없습니다")
            
            # 추적 시작
            success = self.tracker.start_tracking(frame, click_point)
            
            if success:
                print("✅ 추적 시작 완료")
                
                # 원래 타겟 이름 저장 (strict_target_mode용)
                if self.tracker.current_target:
                    # 생성된 타겟 이름을 고유한 이름으로 변경
                    self.tracker.current_target.name = unique_target_name
                    self.original_target_name = unique_target_name
                    self.saved_target_name = unique_target_name
                    
                    # 전신 검출 정보 우선 저장 (Pinecone 저장용)
                    if closest_person:
                        self.tracker.current_target.last_person_detection = closest_person
                        print(f"💾 전신 검출 정보 저장: {closest_person.bbox}")
                        
                        # 전신 기반 추적임을 표시
                        print(f"🎯 전신 기반 추적 시작: {unique_target_name}")
                    
                    # 얼굴 검출 정보도 함께 저장 (보조용)
                    if closest_face:
                        self.tracker.current_target.last_face_detection = closest_face
                        print(f"💾 얼굴 검출 정보 저장 (보조): {closest_face.bbox}")
                    
                    print(f"🎯 원래 타겟 설정: {self.original_target_name}")
                    
                    # 한 번만 Pinecone 저장
                    if not self.pinecone_saved:
                        self.background_queue.put(("save_to_pinecone", {
                            "target": self.tracker.current_target,
                            "frame": frame
                        }))
                        self.pinecone_saved = True
                        print(f"💾 Pinecone 저장 예약: {self.original_target_name}")
            else:
                print("❌ 추적 시작 실패 - 상세 원인:")
                if not face_detections and not person_detections:
                    print("  - 얼굴과 사람이 모두 검출되지 않음")
                elif not closest_face and not closest_person:
                    print("  - 클릭 위치 근처에 얼굴이나 사람이 없음")
                else:
                    print("  - 임베딩 생성 실패 또는 기타 오류")
                    # 전신 기반 임베딩 생성 시도
                    if closest_person and not closest_face:
                        print("  - 전신 기반 임베딩 생성 시도 중...")
                        try:
                            # 전신 영역에서 임베딩 생성 시도
                            px, py, pw, ph = closest_person.bbox
                            person_roi = frame[py:py+ph, px:px+pw]
                            
                            if person_roi.size > 0:
                                # 전신 히스토그램 임베딩 생성
                                embedding = self.tracker.embedding_processor.compute_histogram_embedding(person_roi)
                                if embedding is not None:
                                    print("  - 전신 히스토그램 임베딩 생성 성공")
                                else:
                                    print("  - 전신 히스토그램 임베딩 생성 실패")
                            else:
                                print("  - 전신 영역이 비어있음")
                        except Exception as e:
                            print(f"  - 전신 임베딩 생성 오류: {e}")
            
            # 검출 모드 복원
            self.detection_disabled = restore_detection_disabled
            if restore_detection_disabled:
                print("🔇 검출 모드 비활성화로 복원")
                
        except Exception as e:
            print(f"❌ 추적 시작 오류: {e}")
            import traceback
            traceback.print_exc()
    
    def _execute_pinecone_save(self, data):
        """백그라운드에서 Pinecone 저장 실행"""
        try:
            target = data["target"]
            frame = data["frame"]
            
            print(f"💾 Pinecone 저장 시도: {target.name}")
            
            # 이미 저장된 타겟인지 확인
            if self.saved_target_name == target.name and self.pinecone_saved:
                print(f"⚠️ 이미 저장된 타겟: {target.name}")
                return
            
            # 프레임 크기 정보 가져오기
            frame_shape = frame.shape[:2]
            if self.resize_enabled:
                # 리사이즈된 프레임이므로 원본 크기로 변환
                original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_shape = (original_height, original_width)
            
            # 전신 검출이 있으면 전신 기반으로 저장 (우선)
            if hasattr(target, 'last_person_detection') and target.last_person_detection:
                print("💾 전신 기반 Pinecone 저장 시도")
                
                # 전신 영역을 얼굴 검출로 변환하여 저장
                person_detection = target.last_person_detection
                px, py, pw, ph = person_detection.bbox
                
                # 전신 영역에서 얼굴 영역 추정 (상단 1/3)
                face_x = px
                face_y = py
                face_w = pw
                face_h = ph // 3  # 전신 높이의 1/3을 얼굴로 간주
                
                # 임시 얼굴 검출 객체 생성
                from .models import FaceDetection
                temp_face_detection = FaceDetection(
                    bbox=(face_x, face_y, face_w, face_h),
                    confidence=0.9  # 전신 기반이므로 높은 신뢰도
                )
                
                success = self.tracker.pinecone_manager.save_embedding(
                    target, temp_face_detection, frame_shape
                )
                if success:
                    print("💾 Pinecone 저장 완료 (전신 기반)")
                    self.saved_target_name = target.name
                    self.pinecone_saved = True
                else:
                    print("❌ Pinecone 저장 실패 (전신 기반)")
                    self.pinecone_saved = False
                    
            # 얼굴 검출이 있으면 얼굴 기반으로 저장 (보조)
            elif target.last_face_detection:
                success = self.tracker.pinecone_manager.save_embedding(
                    target, target.last_face_detection, frame_shape
                )
                if success:
                    print("💾 Pinecone 저장 완료 (얼굴 기반)")
                    self.saved_target_name = target.name
                    self.pinecone_saved = True
                else:
                    print("❌ Pinecone 저장 실패 (얼굴 기반)")
                    self.pinecone_saved = False
            else:
                print("❌ Pinecone 저장 실패: 전신과 얼굴 검출 모두 없음")
                self.pinecone_saved = False
                    
        except Exception as e:
            print(f"❌ Pinecone 저장 오류: {e}")
            import traceback
            traceback.print_exc()
            self.pinecone_saved = False  # 오류 시 다시 시도 가능하도록
    
    def _handle_key_input(self) -> bool:
        """키 입력 처리"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            return True
        elif key == ord('c'):
            self.tracker.clear_target()
            self.is_tracking_starting = False
            self.detection_disabled = False
            self.original_target_name = None  # 원래 타겟 이름 초기화
            self.pinecone_saved = False  # 타겟 해제 시 저장 상태 초기화
            self.saved_target_name = None  # 저장된 타겟 이름 초기화
            # target_counter는 유지 (다음 타겟은 고유한 번호를 가짐)
            print("🗑️ 타겟 해제")
        elif key == ord('p'):
            self.is_playing = not self.is_playing
            print(f"⏯️ {'재생' if self.is_playing else '일시정지'}")
        elif key == ord('r'):
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame_number = 0
            print("⏮️ 처음으로 되돌리기")
        elif key >= ord('1') and key <= ord('9'):
            # 프레임 스킵 조절
            self.skip_frames = key - ord('0')
            print(f"⚡ 프레임 스킵: {self.skip_frames}")
        elif key == ord('d'):
            # 검출 모드 토글
            self.detection_disabled = not self.detection_disabled
            if self.detection_disabled:
                print("🔇 검출 모드 비활성화 (성능 향상)")
            else:
                print("🔊 검출 모드 활성화")
        elif key == ord('s'):
            # 해상도 조절 토글
            self.resize_enabled = not self.resize_enabled
            if self.resize_enabled:
                print(f"📐 해상도 조절 활성화 ({self.target_width}x{self.target_height})")
            else:
                print("📐 해상도 조절 비활성화 (원본 크기)")
        elif key == ord('f'):
            # 강제 검출 모드 활성화
            self.detection_disabled = False
            self.last_detection_time = 0  # 즉시 검출 실행
            print("🔊 강제 검출 모드 활성화")
        elif key == ord('n'):
            # 새로운 타겟으로 변경 (현재 추적 중인 타겟 해제)
            if self.tracker.current_target:
                print("🔄 새로운 타겟 선택 모드 활성화")
                self.tracker.clear_target()
                self.is_tracking_starting = False
                self.detection_disabled = False
                self.original_target_name = None
                self.pinecone_saved = False
                self.saved_target_name = None
                print("🗑️ 현재 타겟 해제 - 새로운 타겟을 클릭하세요")
            else:
                print("⚠️ 현재 추적 중인 타겟이 없습니다")
        elif key == ord('h'):
            # 도움말 표시
            print("=" * 50)
            print("추가 키 안내:")
            print("  'f' - 강제 검출 모드 활성화")
            print("  'n' - 새로운 타겟 선택 (현재 타겟 해제)")
            print("  'h' - 도움말 표시")
            print("=" * 50)
        
        return False
    
    def _cleanup(self):
        """정리 작업"""
        # 백그라운드 작업자 종료
        if self.background_queue:
            self.background_queue.put(None)
        
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
