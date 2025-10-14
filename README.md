# SafeView FastAPI Server
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FSafeView%2FCamera_AI.svg?type=shield)](https://app.fossa.com/projects/git%2Bgithub.com%2FSafeView%2FCamera_AI?ref=badge_shield)




실시간 영상 비식별화(얼굴 모자이크)와 자동 녹화/업로드를 제공하는 FastAPI 기반 서버입니다. WebSocket으로 영상을 받아 모자이크 처리 후 스트리밍하고, 조건에 따라 처리본/원본을 동시에 녹화합니다. 결과는 S3에 업로드되며 완료 시 Spring 서버에 콜백을 보냅니다. 오프라인 영상에 대해 특정 시간 구간의 얼굴을 추출하는 API도 포함됩니다.

## 📋 목차

- [프로젝트 개요](#프로젝트-개요)
- [기술 스택](#기술-스택)
- [프로젝트 구조](#프로젝트-구조)
- [주요 기능](#주요-기능)
- [API 문서](#api-문서)
- [설치 및 실행](#설치-및-실행)
- [환경 설정](#환경-설정)
- [데이터베이스](#데이터베이스)
- [개발 가이드](#개발-가이드)
- [라이선스](#라이선스)
- [기여하기](#기여하기)

## 🎯 프로젝트 개요

SafeView AI Server는 실시간 스트림으로 들어오는 프레임을 비식별화(얼굴/번호판 모자이크)한 뒤 스트리밍하고, 조건에 따라 자동으로 녹화하여 S3에 업로드합니다. 또한 오프라인 영상에서 특정 시간 구간의 얼굴을 추출하는 API를 제공합니다. Spring 백엔드와 연동해 키 검증과 메타 등록을 처리합니다.

### 주요 특징

- 🎥 실시간 비식별화 스트리밍: WebSocket으로 JPEG 프레임 수신 → 모자이크 → JPEG 송신
- 🔴 자동 녹화: 인원 수/히스토리 기반 시작·종료, 처리본/원본 이중 저장
- ☁️ S3 업로드: 업로드/목록/서명 URL 발급, 완료 시 Spring 콜백
- 🧠 시간 구간 얼굴 추출: 파일/URL/S3/Blob 입력 지원, 중복 제거 후 결과 반환
- 🔐 키 검증 연동: 백엔드 API + AiApiKey 헤더 기반 검증

## 🛠️ 기술 스택

### Backend
- Python 3.10 ~ < 3.13
- FastAPI, Uvicorn

### Vision/Streaming
- OpenCV, MediaPipe, Ultralytics YOLO
- NumPy

### Networking & Storage
- aiohttp, requests
- AWS S3(boto3)

### Dev/Packaging
- uv 또는 pip
- Docker (python:3.11-slim, ffmpeg/libgl 포함)

## 📁 프로젝트 구조

```
server/
  app.py                # FastAPI 앱/미들웨어/메타 라우터
  websocket_stream.py   # WS 수신, 모자이크, 자동녹화 트리거, 스트리밍
  recording.py          # 녹화 시작/종료, 업로드, 목록, URL 발급
  verification.py       # AiApiKey 기반 키 검증, 사용자 정보 수신
  face_time_api.py      # 시간 기반 얼굴 추출 REST API
  core/state.py         # 런타임 공유 상태
  storage/s3.py         # S3 클라이언트/업로드/리스트/서명 URL
AI_processor.py         # 프레임 비식별화 파이프라인
http_video_server.py    # 실행 스크립트(uvicorn 진입)
Dockerfile              # 컨테이너 빌드/실행 정의
pyproject.toml          # uv 의존성(권장)
requirements.txt        # pip 의존성
```

## 🚀 주요 기능

### 1. 실시간 비식별화 스트리밍
- WebSocket: `ws://{host}:{port}/ws/video`
- JPEG 바이너리 프레임 수신 → 모자이크 처리 → JPEG 송신
- FPS 제한, 코얼레싱으로 최신 프레임 우선 전송

### 2. 자동 녹화/업로드
- 시작: 인원 수 임계치 + presence 히트 충족 시
- 종료: 부재 지속 시간(AUTO_ZERO_TIMEOUT_SEC) 초과 시
- 처리본·원본 동시 저장 → S3 업로드(설정 시) → Spring 콜백

### 3. 시간 구간 얼굴 추출
- 입력: 업로드 파일, HTTP(S) URL, data:/blob:, S3(recordings/{filename})
- YOLO(+Cascade 보정) 기반 탐지, 해시·위치로 중복 제거, 결과 저장/업로드

### 4. 검증/접근 제어
- AiApiKey + 백엔드 키 검증 결과에 따라 원본 저장 완화/강제 모자이크
- 모든 연결에 모자이크 강제 적용 가능

## 📚 API 문서

### 메타/상태
- `GET /health` - 서버 상태/리비전
- `GET /last_detections` - 마지막 스트림 요약
- `GET /auto_recording` - 자동 녹화 설정/상태
- `POST /auto_recording` - {enabled, threshold} 런타임 변경
- `POST /disconnect_ws` - 모든 WebSocket 연결 종료 통보

### 스트림(WebSocket)
- `WS /ws/video`
  - Client → Server: JPEG 프레임 또는 제어 메시지
    - `{ "type": "key_verification", "accessToken": "...", "cameraId": "..." }`
    - `{ "type": "disconnect" }`
  - Server → Client: 모자이크 JPEG, 이벤트(JSON)
    - `auto_recording_started`, `auto_recording_will_finalize`, `auto_recording_finalized`, 오류 등

### 녹화 제어/목록
- `POST /start_recording` - 수동 시작(다음 프레임부터)
- `POST /stop_recording` - 즉시 종료 및 업로드/경로 반환
- `GET /recording_status` - 현재 상태
- `GET /recordings` - S3 객체 목록
- `GET /recordings/{filename}` - 서명 URL 발급

### 검증/강제 모자이크
- `POST /client/user` - `{userId}` 수신/저장(메타 전송용)
- `GET /verification_status` - 검증 연결 수 요약
- `POST /force_mosaic` - 전 연결 모자이크 강제 적용

### 시간 기반 얼굴 검출(prefix: `/face-detection`)
- `GET /face-detection` - 버전/상태
- `POST /face-detection/upload-video` - 폼 업로드 → 로컬 저장
- `POST /face-detection/detect-faces` - 구간 얼굴 추출
  - Query: `time_input`(예: "90" 또는 "1 30"), `filename` | `video_url` | `from_s3` + `file`
- `GET /face-detection/video-info/{filename}` - 로컬 업로드 비디오 메타
- `GET /face-detection/results` - 결과 요약 목록
- `GET /face-detection/results/{result_id}` - 결과 JSON
- `GET /face-detection/download-face/{result_id}/{filename}` - 얼굴 이미지 다운로드

## 🛠️ 설치 및 실행

### 1. 프로젝트 클론
```bash
git clone [repository-url]
cd Camera_AI
```

### 2. 의존성 설치
- uv(권장)
```bash
uv sync
```
- pip(대안)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. 애플리케이션 실행
```bash
# 방법 A
uv run python http_video_server.py
# 방법 B
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 4. 개발 서버 접속
- 앱: http://localhost:8000
- 문서: http://localhost:8000/docs


## 🗄️ 데이터베이스

이 서비스는 별도 데이터베이스를 직접 사용하지 않습니다. 결과와 녹화본은 로컬 디렉토리 또는 S3에 저장됩니다.

### 주요 경로/프리픽스
- 로컬: `uploads/`, `api_results/`
- S3: `recordings/`, `api_results/{result_id}/faces/`

## 🔒 보안

### 인증 및 권한
- 🔑 키 검증: 백엔드 `/api/decryption/keys/verify/ai`와 AiApiKey 헤더로 검증
- 🧱 강제 모자이크: 미검증 연결은 항상 모자이크 적용, 필요 시 전 연결 강제 전환
- 🔐 CORS 허용 범위 설정, 프로덕션 HTTPS 권장

### 데이터 보안
- ☁️ 업로드: S3 서명 URL 발급으로 안전한 접근
- 🧪 입력값 검증: 파일 형식/시간 파라미터 검증, 오류 응답 일관화

## 👨‍💻 개발 가이드

### 코드 컨벤션
- 모듈화된 라우터(server/*) 유지, 공통 상태는 `server/core/state.py`
- 설정은 `server/config.py`의 환경 변수로 제어
- 예외는 사용자 친화적 메시지로 래핑(HTTPException/JSON)

### 개발 워크플로
```bash
uv add <package>
uv run uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 🤝 기여하기

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---
이슈나 개선 요청 환영.


## License
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FSafeView%2FCamera_AI.svg?type=large)](https://app.fossa.com/projects/git%2Bgithub.com%2FSafeView%2FCamera_AI?ref=badge_large)
