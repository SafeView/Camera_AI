"""
사람 전체 검출 모듈
"""

import cv2
import numpy as np
from typing import List, Optional
from .models import PersonDetection, FaceDetection
from .config import Config
from .dependencies import Dependencies

class PersonDetector:
    """사람 전체 검출기 (HOG + YOLO + MediaPipe)"""
    
    def __init__(self, dependencies: Dependencies):
        self.deps = dependencies
        
        # HOG 사람 검출기 초기화
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        # YOLO 초기화 (사용 가능한 경우)
        self.yolo_net = None
        self.yolo_output_layers = None
        self._init_yolo()
        
        # MediaPipe Pose 초기화 (사용 가능한 경우)
        self.pose_detector = None
        if self.deps.mediapipe_available:
            self._init_mediapipe_pose()
    
    def _init_yolo(self):
        """YOLO 초기화 (COCO 사전 훈련 모델)"""
        try:
            # YOLO 설정 파일과 가중치 파일 경로
            # 실제 환경에서는 이 파일들이 있어야 함
            config_path = "yolo/yolov3.cfg"
            weights_path = "yolo/yolov3.weights"
            
            if cv2.os.path.exists(config_path) and cv2.os.path.exists(weights_path):
                self.yolo_net = cv2.dnn.readNet(weights_path, config_path)
                layer_names = self.yolo_net.getLayerNames()
                self.yolo_output_layers = [layer_names[i[0] - 1] for i in self.yolo_net.getUnconnectedOutLayers()]
        except Exception as e:
            pass
    
    def _init_mediapipe_pose(self):
        """MediaPipe Pose 초기화 (다중 인물용)"""
        try:
            import mediapipe as mp
            # Holistic을 사용하여 다중 인물 검출 향상
            mp_holistic = mp.solutions.holistic
            self.holistic_detector = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                refine_face_landmarks=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4
            )
            
            # Selfie Segmentation도 추가 (사람 영역 분할)
            mp_selfie = mp.solutions.selfie_segmentation
            self.selfie_detector = mp_selfie.SelfieSegmentation(model_selection=1)
            
        except Exception as e:
            self.holistic_detector = None
            self.selfie_detector = None
    
    def detect_persons(self, frame: np.ndarray) -> List[PersonDetection]:
        """사람 검출 (HOG + YOLO + MediaPipe 하이브리드)"""
        detections = []
        h, w = frame.shape[:2]
        
        # HOG 검출
        detections.extend(self._detect_with_hog(frame))
        
        # YOLO 검출 (사용 가능한 경우)
        if self.yolo_net is not None:
            detections.extend(self._detect_with_yolo(frame, w, h))
        
        # MediaPipe 다중 검출 (사용 가능한 경우)
        if hasattr(self, 'holistic_detector') and self.holistic_detector is not None:
            detections.extend(self._detect_with_mediapipe_multiple(frame, w, h))
        
        # 중복 제거 및 필터링
        return self._filter_and_deduplicate(detections, w, h)
    
    def _detect_with_hog(self, frame: np.ndarray) -> List[PersonDetection]:
        """HOG로 사람 검출 (다중 인물 최적화)"""
        detections = []
        
        # HOG 검출 - 더 세밀한 설정으로 여러 사람 검출
        persons, weights = self.hog.detectMultiScale(
            frame, 
            winStride=Config.HOG_WIN_STRIDE,
            padding=Config.HOG_PADDING,
            scale=Config.HOG_SCALE,
            hitThreshold=Config.HOG_HIT_THRESHOLD
        )
        
        for i, (x, y, w, h) in enumerate(persons):
            # weights가 스칼라인지 배열인지 확인
            if isinstance(weights, (list, np.ndarray)) and len(weights) > i:
                confidence = weights[i][0] if isinstance(weights[i], (list, np.ndarray)) else weights[i]
            else:
                confidence = 0.5
            detections.append(PersonDetection(x, y, w, h, confidence))
        
        return detections
    
    def _detect_with_yolo(self, frame: np.ndarray, w: int, h: int) -> List[PersonDetection]:
        """YOLO로 사람 검출"""
        detections = []
        
        try:
            # YOLO 입력 준비
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.yolo_net.setInput(blob)
            outputs = self.yolo_net.forward(self.yolo_output_layers)
            
            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # 사람 클래스 (COCO 데이터셋에서 person = 0)
                    if class_id == 0 and confidence > Config.YOLO_CONFIDENCE_THRESHOLD:
                        center_x = int(detection[0] * w)
                        center_y = int(detection[1] * h)
                        width = int(detection[2] * w)
                        height = int(detection[3] * h)
                        
                        x = int(center_x - width / 2)
                        y = int(center_y - height / 2)
                        
                        detections.append(PersonDetection(x, y, width, height, confidence))
        
        except Exception as e:
            pass
        
        return detections
    
    def _detect_with_mediapipe_multiple(self, frame: np.ndarray, w: int, h: int) -> List[PersonDetection]:
        """MediaPipe로 다중 사람 검출 (Holistic + Segmentation)"""
        detections = []
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Holistic 검출
            if self.holistic_detector:
                holistic_results = self.holistic_detector.process(rgb_frame)
                if holistic_results.pose_landmarks:
                    detection = self._extract_person_from_landmarks(holistic_results.pose_landmarks, w, h)
                    if detection:
                        detections.append(detection)
            
            # Selfie Segmentation으로 사람 영역 찾기
            if self.selfie_detector:
                seg_results = self.selfie_detector.process(rgb_frame)
                if seg_results.segmentation_mask is not None:
                    seg_detections = self._extract_persons_from_segmentation(seg_results.segmentation_mask, w, h)
                    detections.extend(seg_detections)
        
        except Exception as e:
            pass
        
        return detections
    
    def _extract_person_from_landmarks(self, pose_landmarks, w: int, h: int) -> Optional[PersonDetection]:
        """포즈 랜드마크에서 사람 바운딩 박스 추출"""
        try:
            landmarks = pose_landmarks.landmark
            x_coords = [lm.x * w for lm in landmarks if lm.visibility > 0.5]
            y_coords = [lm.y * h for lm in landmarks if lm.visibility > 0.5]
            
            if len(x_coords) < 5:  # 최소 5개 랜드마크 필요
                return None
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # 바운딩 박스 확장
            margin_x = int((x_max - x_min) * 0.15)
            margin_y = int((y_max - y_min) * 0.1)
            
            x = max(0, x_min - margin_x)
            y = max(0, y_min - margin_y)
            width = min(w - x, x_max - x_min + 2 * margin_x)
            height = min(h - y, y_max - y_min + 2 * margin_y)
            
            return PersonDetection(x, y, width, height, 0.8)
        except:
            return None
    
    def _extract_persons_from_segmentation(self, mask, w: int, h: int) -> List[PersonDetection]:
        """세그멘테이션 마스크에서 사람 영역들 추출"""
        detections = []
        
        try:
            # 마스크 이진화
            mask_binary = (mask > 0.5).astype(np.uint8) * 255
            
            # 컨투어 찾기
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                # 면적이 너무 작으면 무시
                area = cv2.contourArea(contour)
                if area < 1000:  # 최소 면적
                    continue
                
                # 바운딩 박스 계산
                x, y, width, height = cv2.boundingRect(contour)
                
                # 유효성 검사
                if width > 20 and height > 40:  # 최소 사람 크기
                    detections.append(PersonDetection(x, y, width, height, 0.7))
        
        except Exception as e:
            pass
        
        return detections
    
    def _filter_and_deduplicate(self, detections: List[PersonDetection], w: int, h: int) -> List[PersonDetection]:
        """검출 결과 필터링 및 고급 중복 제거"""
        if not detections:
            return []
        
        # 유효성 검사
        valid_detections = [d for d in detections if self._is_valid_person(d, w, h)]
        
        if not valid_detections:
            return []
        
        # 고급 NMS 적용
        return self._advanced_nms(valid_detections)
    
    def _advanced_nms(self, detections: List[PersonDetection]) -> List[PersonDetection]:
        """고급 Non-Maximum Suppression"""
        if len(detections) <= 1:
            return detections
        
        # 신뢰도와 면적을 결합한 점수 계산
        for detection in detections:
            detection.score = detection.confidence * 0.7 + (detection.area / 100000) * 0.3
        
        # 점수 기준 정렬
        detections.sort(key=lambda d: d.score, reverse=True)
        
        keep = []
        while detections:
            # 가장 높은 점수의 검출 선택
            current = detections.pop(0)
            keep.append(current)
            
            # 현재 검출과 겹치는 것들 제거
            remaining = []
            for detection in detections:
                iou = self._calculate_iou(current, detection)
                
                # 적응적 IoU 임계값 (크기에 따라 조정)
                adaptive_threshold = Config.IOU_THRESHOLD
                if detection.area < 5000:  # 작은 검출의 경우 더 엄격
                    adaptive_threshold *= 0.8
                elif detection.area > 50000:  # 큰 검출의 경우 더 관대
                    adaptive_threshold *= 1.2
                
                if iou <= adaptive_threshold:
                    remaining.append(detection)
            
            detections = remaining
        
        return keep
    
    def _is_valid_person(self, detection: PersonDetection, frame_w: int, frame_h: int) -> bool:
        """사람 검출 유효성 검사"""
        x, y, w, h = detection.bbox
        
        # 최소 크기 검사 (사람은 얼굴보다 훨씬 커야 함)
        min_person_size = Config.MIN_FACE_SIZE * 3
        if w < min_person_size or h < min_person_size:
            return False
        
        # 최대 크기 검사
        if w > frame_w * 0.8 or h > frame_h * 0.8:
            return False
        
        # 비율 검사 (사람은 세로로 긴 형태)
        aspect_ratio = w / h
        if aspect_ratio < 0.2 or aspect_ratio > 0.8:
            return False
        
        # 경계 검사
        if x < 0 or y < 0 or x + w > frame_w or y + h > frame_h:
            return False
        
        return True
    
    def _calculate_iou(self, detection1: PersonDetection, detection2: PersonDetection) -> float:
        """IoU(Intersection over Union) 계산"""
        x1, y1, w1, h1 = detection1.bbox
        x2, y2, w2, h2 = detection2.bbox
        
        # 교집합 계산
        xi1 = max(x1, x2)
        yi1 = max(y1, y2)
        xi2 = min(x1 + w1, x2 + w2)
        yi2 = min(y1 + h1, y2 + h2)
        
        if xi2 <= xi1 or yi2 <= yi1:
            return 0.0
        
        inter_area = (xi2 - xi1) * (yi2 - yi1)
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0
    
    def find_closest_person(self, detections: List[PersonDetection], click_point: tuple) -> Optional[PersonDetection]:
        """클릭 지점에서 가장 가까운 사람 찾기"""
        click_x, click_y = click_point
        closest_detection = None
        min_distance = float('inf')
        
        for detection in detections:
            if detection.is_close_to_point(click_x, click_y):
                center_x, center_y = detection.center
                distance = ((click_x - center_x) ** 2 + (click_y - center_y) ** 2) ** 0.5
                if distance < min_distance:
                    min_distance = distance
                    closest_detection = detection
        
        return closest_detection
    
    def associate_face_with_person(self, person_detection: PersonDetection, face_detections: List[FaceDetection]) -> PersonDetection:
        """사람 검출과 얼굴 검출을 연결"""
        px, py, pw, ph = person_detection.bbox
        
        # 사람 바운딩 박스 내부에 있는 얼굴 찾기
        for face in face_detections:
            fx, fy, fw, fh = face.bbox
            
            # 얼굴이 사람 바운딩 박스 내부에 있는지 확인
            if (px <= fx and py <= fy and 
                fx + fw <= px + pw and fy + fh <= py + ph):
                person_detection.face_detection = face
                break
        
        return person_detection
