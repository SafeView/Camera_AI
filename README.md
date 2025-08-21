# SafeView FastAPI Server

실시간 영상 비식별화(얼굴/번호판 모자이크)와 자동 녹화/업로드를 제공하는 FastAPI 기반 서버입니다. WebSocket으로 영상을 받아 모자이크 처리 후 스트리밍하고, 조건에 따라 처리본/원본을 동시에 녹화합니다. 결과는 S3에 업로드되며 완료 시 Spring 서버에 콜백을 보냅니다. 오프라인 영상에 대해 특정 시간 구간의 얼굴을 추출하는 API도 포함됩니다.

자세한 구조와 동작 흐름은 docs/ARCHITECTURE.md 참고.

## 주요 기능
- WebSocket 실시간 입력 + 모자이크 스트리밍 (JPEG 전송, FPS 제한, 코얼레싱)
- 옆모습 지속 모자이크: 머리 추적(IoU+TTL) + HOG 보조
- 자동 녹화: 처리본/원본 이중 저장, presence 히트 & 부재 타임아웃 기반 시작/중단
- S3 업로드 및 로컬 폴백, Spring 콜백으로 URL 전달
- 시간 기반 얼굴 검출 API: 파일 업로드, blob/data URL, http(s), S3 지원

## 빠른 실행 요약
```bash
git clone https://github.com/SafeView/Python_AI.git
cd Python_AI
curl -LsSf https://astral.sh/uv/install.sh | sh  # 또는 brew install uv
uv sync
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
# 또는
uv run python http_video_server.py
```
> server.app:app 직접 실행 혹은 http_video_server.py 로 진입.

## 기술 스택
- Python 3.10 ~ < 3.13
- FastAPI / Uvicorn
- OpenCV, MediaPipe, YOLO(ultralytics)
- aiohttp, boto3
- uv 패키지/실행 관리

## 사전 준비
1. Python 버전 확인 (`python --version`)
2. uv 설치 (없다면 위 요약 참조)
3. (선택) `.venv/` 자동 생성 (uv sync 시)

## 설치 (uv 권장)
```bash
uv sync
```
패키지 추가: `uv add 패키지명`

### (대안) pip
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 실행
```bash
uv run python http_video_server.py
# 또는
uv run uvicorn http_video_server:app --host 0.0.0.0 --port 8000
```
다중 워커 사용 시 전역 상태 공유 불가.

## WebSocket 사용 흐름
1. `ws://{host}:{port}/ws/video` 연결
2. JPEG 프레임 전송
3. 모자이크 처리 후 JPEG 수신
4. 자동 녹화 이벤트 메시지 수신

## 자동 녹화 요약
| 조건 | 설명 |
|------|------|
| 시작 | `pcnt >= AUTO_RECORD_THRESHOLD` AND presence 히트 충족 |
| 중단 | 연속 부재 시간 ≥ `AUTO_ZERO_TIMEOUT_SEC` |
| 메타 | 최대 인원 `_recording_max_persons` 기록 |

## 주요 엔드포인트
| 메소드 | 경로 | 설명 |
|--------|------|------|
| GET | /health | 헬스 체크 |
| WS | /ws/video | 실시간 스트림 |
| POST | /start_recording | 수동 녹화 요청 |
| POST | /stop_recording | 녹화 중단 |
| GET | /recording_status | 녹화 상태 |
| GET | /recordings | 녹화 목록 |
| POST | /face-detection/upload-video | 오프라인 업로드 |
| POST | /face-detection/detect-faces | 특정 시점 얼굴 추출 |

## 디렉토리 개요
```
server/
  app.py
  websocket_stream.py
  recording.py
  verification.py
  face_time_api.py
  core/state.py
  analytics/person_count.py
AI_processor.py
http_video_server.py
docs/ARCHITECTURE.md
```

## 개발 워크플로
```bash
uv add some-package
uv run python -c "import requests;print(requests.get('http://localhost:8000/health').json())"
```

## 문제 해결
| 증상 | 조치 |
|------|------|
| 자동 녹화 시작 안 됨 | 임계/히트 조정, DETECT_EVERY_N=1 테스트 |
| VideoWriter 실패 | 코덱 지원 확인, MJPG 폴백 로그 확인 |
| S3 업로드 실패 | 자격/권한 및 구성 점검 |
| 사람 수 0 지속 | 조명/해상도/모델 설정 조정 |

## 라이선스
(내부 정책에 따라 지정 예정)

---
이슈나 개선 요청 환영.
