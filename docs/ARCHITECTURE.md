# SafeView FastAPI – 아키텍처 및 기능 가이드

## 개요
이 서비스는 실시간 영상 비식별화(얼굴/번호판 모자이크)와 녹화를 제공합니다. WebSocket으로 영상을 받아 AI 기반 모자이크를 적용해 다시 스트리밍하고, 조건에 따라 처리본과 원본을 동시에 자동 녹화합니다. 결과물은 S3에 업로드되며, 녹화가 완료되면 Spring 백엔드로 콜백을 전송합니다. 오프라인 영상에서 특정 시점의 얼굴을 추출하는 REST API도 포함됩니다.

## 구성 요소
- FastAPI 앱(`server/app.py` + 실행 엔트리: `http_video_server.py`)
  - WebSocket 엔드포인트: 입력, 비식별화, 스트리밍
  - 자동 녹화 라이프사이클: 시작/정지, 이중 파일(처리본/원본) 저장, 종료 처리
  - S3 연동: 결과 업로드, S3/로컬 URL 제공
  - Spring 콜백: 종료 시 URL 목록을 POST
  - 얼굴 검출 REST API: 업로드/검출/결과 조회
- 모자이크 파이프라인(`AI_processor.py`)
  - MediaPipe + Haar/Profile 검증 + HOG fallback
  - IoU+TTL 기반 머리(옆모습) 추적
  - 번호판 Haar 검출
- 인원 수 추정(`server/analytics/person_count.py`)
  - YOLO 기반 AnalyticsEngine → 실패시 HOG Fallback
- 전역 상태(`server/core/state.py`)
  - WebSocket/녹화/검증/디버그 메타 저장 (단일 프로세스 전제)
- 저장소/업로드(`server/storage/s3.py`)

## 의존성 & 실행 (uv)
- 의존성: `pyproject.toml` / 잠금: `uv.lock`
- 설치: `uv sync`
- 실행: `uv run python http_video_server.py` 또는 `uv run uvicorn http_video_server:app --host 0.0.0.0 --port 8000`
- 패키지 추가: `uv add <name>`
> 멀티프로세스(Uvicorn workers>1) 사용 시 전역 state 공유 불가 → Redis 등 외부 상태 스토어 필요.

## 데이터 흐름 (WebSocket)
1. 클라이언트 `/ws/video` 연결 → JPEG 프레임 전송
2. 서버: 디코드 → (필요 시 다운스케일) → 얼굴/번호판/머리 영역 탐지 → 모자이크 적용
3. JPEG 인코드 & FPS 제한/코얼레싱 → 클라이언트 전송
4. 인원 수 추정 → presence 히스토리 업데이트
5. 조건 만족 시 자동 녹화 시작 (처리본/원본 동시 VideoWriter)
6. 부재 타임아웃 충족 시 녹화 종료 & S3 업로드 & Spring 콜백

## 자동 녹화 로직 (현재 구현)
| 단계 | 조건 | 설명 |
|------|------|------|
| 시작 | `pcnt >= AUTO_RECORD_THRESHOLD` AND presence 히트(`hits >= AUTO_PRESENCE_MIN_HITS` 또는 초기 히스토리 전부 1) | 순간 잡음/1프레임 감지 회피
| 진행 | 프레임마다 최대 인원 갱신 | `_recording_max_persons` 유지
| 중단 | 연속 부재 시간(`now - _last_nonzero_person_ts`) ≥ `AUTO_ZERO_TIMEOUT_SEC` | 사람이 다시 감지되면 타이머 리셋
| finalize | VideoWriter 릴리즈 → (S3 업로드/로컬 보존) → WebSocket 이벤트 + Spring POST | 오류 무시 베스트 에포트

## 시간 기반 얼굴 검출 API
- 업로드: `POST /face-detection/upload-video` (multipart)
- 특정 시점 처리: `POST /face-detection/detect-faces` (time_input + 소스 지정)
- 결과 조회/다운로드: `/face-detection/results`, `/face-detection/download-face/...`
- 검출 파이프라인: YOLO(face) → Haar(Profile/정면) 2차 검증 → 크기/종횡비 → NMS → 폴백(Haar 단독)

## 모자이크 파이프라인 요약
1. 샘플링 간격(프레임 N당 1회) + 다운스케일 → MediaPipe 초기 얼굴 후보
2. ROI Haar/Profile & 반전 검사 → 오탐 제거
3. 없을 경우: (a) 풀해상도 재시도 (b) Haar-only 재시도 (c) HOG 머리 상단 fallback
4. IoU 기반 트랙 & TTL로 옆모습 전환 유지
5. NMS 후 Margin 확장 + 픽셀화

## 인원 수 추정
- AnalyticsEngine(YOLO) 활성 시: bounding boxes → len → person count
- 실패/미존재: HOGDescriptor로 사람 후보 → 카운트
- 디버그 정보 state._auto_debug에 기록 (엔진 사용/오류 등)

## 환경 변수 (발췌)
| 그룹 | 변수 | 설명 |
|------|------|------|
| 서버 | `API_HOST`, `API_PORT` | 바인딩 주소/포트 |
| 스트림 | `STREAM_TARGET_FPS`, `STREAM_MAX_WIDTH`, `STREAM_JPEG_QUALITY` | 전송 FPS/해상도/품질 |
| 자동 녹화 | `AUTO_RECORD_ENABLED`, `AUTO_RECORD_THRESHOLD`, `AUTO_PRESENCE_MIN_HITS`, `AUTO_PRESENCE_WINDOW`, `AUTO_ZERO_TIMEOUT_SEC` | 시작/중단 임계 설정 |
| 업로드 | `S3_BUCKET_NAME`, `S3_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | S3 구성 |
| 콜백 | `SPRING_MAKE_ENTITY_URL` | Spring 엔드포인트 |
| AI (실시간) | `AI_FACE_HAAR_VALIDATE`, `AI_FACE_CONF`, `AI_FACE_MIN_SIZE_PX`, `AI_FACE_SAMPLE_N`, `AI_BOX_MARGIN`, `AI_HEAD_STICKY_FRAMES`, `AI_HEAD_IOU_MATCH`, `AI_HEAD_EXPAND_TOP_RATIO`, `AI_HEAD_EXPAND_X_RATIO`, `AI_USE_HOG_PERSON_FALLBACK` | 얼굴/머리 탐지 튜닝 |
| 오프라인 얼굴 | `FACE_DETECTION_CONFIDENCE_THRESHOLD`, `FACE_SIMILARITY_THRESHOLD`, `PROCESSING_DURATION_SECONDS` | REST 얼굴 검출 제어 |

## 상태 관리
- `server/core/state.py` 단일 프로세스 전역 메모리
- 레이스 완화: `recording_lock`, `verification_lock`
- 멀티워커/컨테이너 확장 시 외부 스토어(예: Redis) + Pub/Sub 필요

## 실패/예외 전략
| 영역 | 전략 |
|------|------|
| YOLO 가용성 | 실패 시 HOG fallback 로그 기록 후 계속 진행 |
| VideoWriter 생성 실패 | 다른 fourcc(MJPG) 재시도, 최종 실패 시 이벤트로 reason 전송 |
| S3 업로드 실패 | 로컬 경로 반환 + 오류 배열 포함 |
| Spring 콜백 오류 | 무시(비차단), 재시도 미구현(향후 백오프 큐 고려) |

## 향후 개선 아이디어
- 멀티프로세스 호환(외부 상태/작업 큐)
- 최소 녹화 길이/세그먼트 병합
- YOLO 추론 비동기 파이프라인 & Batch 처리
- WebRTC 전송 대체 (지연/대역폭 절감)
- 얼굴 재식별 해시 기반 개인별 통계(프라이버시 가드 필요)

## 요약 다이어그램 (논리)
```
Client → WS(Frame) → Decode → Detect/Anonymize → JPEG → Throttle → Send
                                 ↓
                            Person Count → Auto Record FSM → Writers → (S3) → Spring
```

---
최신 실행 및 상세 설정 예시는 README 참고.
