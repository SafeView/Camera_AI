# SafeView FastAPI – 아키텍처 및 기능 가이드

## 개요
이 서비스는 실시간 영상 비식별화(얼굴/번호판 모자이크)와 녹화를 제공합니다. WebSocket으로 영상을 받아 AI 기반 모자이크를 적용해 다시 스트리밍하고, 조건에 따라 처리본과 원본을 동시에 자동 녹화합니다. 결과물은 S3에 업로드되며, 녹화가 완료되면 Spring 백엔드로 콜백을 전송합니다. 오프라인 영상에서 특정 시점의 얼굴을 추출하는 REST API도 포함됩니다.

## 구성 요소
- FastAPI 앱(`http_video_server.py`)
  - WebSocket 엔드포인트: 입력, 비식별화, 스트리밍
  - 자동 녹화 라이프사이클: 시작/정지, 이중 파일(처리본/원본) 저장, 종료 처리
  - S3 연동: 결과 업로드, S3/로컬 URL 제공
  - Spring 콜백: 종료 시 URL 목록을 POST
  - 얼굴 검출 REST API: 업로드/검출/결과 조회
- AI 프로세서(`AI_processor.py`)
  - MediaPipe 기반 얼굴 검출, 선택적 Haar 검증(정면/프로파일)
  - IoU+TTL 기반 머리 추적(스티키)으로 옆모습도 지속 모자이크
  - HOG 사람 검출 보조로 얼굴 미검출 시 머리 상단 영역 모자이크
  - 번호판 Haar 검출
- 분석(`analytics.py`)
  - 인원 수 카운트/장면 분석(YOLO 기반), 선택 사용

## 데이터 흐름 (WebSocket)
1. 클라이언트가 `/ws/video` 연결 후 프레임을 전송합니다.
2. 서버는 프레임을 디코드 후 `AI_processor.detect_and_blur` 실행:
   - 축소 해상도에서 MediaPipe 얼굴 검출
   - ROI에 대해 Haar(정면/프로파일+좌우반전) 검증으로 오탐 감소
   - 스티키 머리 트랙으로 옆모습 전환 시에도 블러 지속
   - 얼굴 미검출 시 HOG로 머리 상단 영역 추정 후 블러
3. 처리된 프레임을 FPS 제한과 합쳐 보내기(coalesced)로 스트리밍(JPEG)합니다.
4. 자동 녹화: 인원/활동 임계 충족 시 이중 파일 녹화 시작
   - 처리본(모자이크) MP4
   - 원본 MP4
5. 정지/종료 시 파일을 닫고 S3 업로드(설정된 경우) 후 Spring(`SPRING_MAKE_ENTITY_URL`)에 URL 목록을 전송합니다.

## 시간 기반 얼굴 검출 API
- 업로드: `POST /face-detection/upload-video` (multipart 파일)
- 특정 시간부터 얼굴 검출: `POST /face-detection/detect-faces`
  - 파라미터: `time_input`, `filename` | `video_url(blob:, data:, http/https)` | `from_s3`
  - `blob:`은 서버가 직접 접근 불가 → 같은 요청에 파일을 함께 업로드해야 함
  - `data:`는 서버에서 base64 디코드 후 임시 파일로 처리
  - `http(s)`는 다운로드 후 임시 파일로 처리
  - 출력: 감지된 얼굴 이미지 URL 리스트(S3 또는 로컬 파일 URL)
- 결과 목록: `GET /face-detection/results`
- 이미지 다운로드: `GET /face-detection/download-face/{result_id}/{filename}`

### 오탐 감소 로직(API 측)
- YOLO 결과는 클래스명이 "face"인 것만 수용
- 각 후보 박스에 대해 Haar(정면/프로파일+좌우반전)로 2차 검증
- 최소 크기/종횡비 검사 및 NMS로 중복 제거
- face 클래스가 없을 경우 Haar 단독으로 폴백

## 자동 녹화
- 환경변수로 활성화(AUTO_RECORD_ENABLED 등)
- 히스테리시스 임계값으로 잦은 시작/정지 방지
- 처리본/원본을 병렬로 기록
- 종료 시: S3 업로드(선택) + Spring에 URL 목록 콜백

## 환경 변수
주요 항목(일부):
- 서버: `API_HOST`, `API_PORT`
- 스트림 튜닝: `STREAM_TARGET_FPS`, `STREAM_MAX_WIDTH`, `STREAM_JPEG_QUALITY`
- 자동 녹화: `AUTO_RECORD_*` (활성화/임계/타임아웃 등)
- S3: `S3_BUCKET_NAME`, `S3_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
- Spring: `SPRING_MAKE_ENTITY_URL`
- 라이브 AI(`AI_processor.py`):
  - `AI_FACE_HAAR_VALIDATE`(기본 1), `AI_FACE_CONF`, `AI_FACE_MIN_SIZE_PX`,
    `AI_HAAR_MIN_NEIGHBORS`, `AI_FACE_SAMPLE_N`, `AI_BOX_MARGIN`,
    `AI_HEAD_STICKY_FRAMES`, `AI_HEAD_IOU_MATCH`,
    `AI_HEAD_EXPAND_TOP_RATIO`, `AI_HEAD_EXPAND_X_RATIO`,
    `AI_USE_HOG_PERSON_FALLBACK`
- 얼굴 검출 API(오프라인):
  - `FACE_DETECTION_CONFIDENCE_THRESHOLD`, `FACE_SIMILARITY_THRESHOLD`,
    `PROCESSING_DURATION_SECONDS`

## 엔드포인트(발췌)
- WebSocket
  - `ws://{host}:{port}/ws/video` — 프레임 전송/모자이크 프레임 수신
- 얼굴 검출
  - `POST /face-detection/upload-video`
  - `POST /face-detection/detect-faces`
  - `GET  /face-detection/video-info/{filename}`
  - `GET  /face-detection/results`
  - `GET  /face-detection/download-face/{result_id}/{filename}`
- 녹화 관리: start/stop/list(구현 범위에 따라 제공)
