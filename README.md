# SafeView FastAPI Server

실시간 영상 비식별화(얼굴/번호판 모자이크)와 자동 녹화/업로드를 제공하는 FastAPI 기반 서버입니다. WebSocket으로 영상을 받아 모자이크 처리 후 스트리밍하고, 조건에 따라 처리본/원본을 동시에 녹화합니다. 결과는 S3에 업로드되며 완료 시 Spring 서버에 콜백을 보냅니다. 오프라인 영상에 대해 특정 시간 구간의 얼굴을 추출하는 API도 포함됩니다.

자세한 구조와 동작 흐름은 docs/ARCHITECTURE.md 참고.

## 주요 기능
- WebSocket 실시간 입력 + 모자이크 스트리밍
- 옆모습 지속 모자이크: 머리 추적(IoU) + HOG 보조
- 자동 녹화: 처리본/원본 이중 저장, 히스테리시스 임계
- S3 업로드 및 로컬 폴백, Spring 콜백으로 URL 전달
- 시간 기반 얼굴 검출 API: 파일 업로드, blob/data URL, http(s), S3 지원

## 빠른 시작
1) 의존성 설치
- pip install -r requirements.txt
- (선택) 분석 모듈: pip install -r requirements-analytics.txt

2) 환경 변수 설정 (S3, 임계값 등) — docs/ARCHITECTURE.md 참고

3) 서버 실행
- uvicorn http_video_server:app --host 0.0.0.0 --port 8000

브라우저에서 http://localhost:8000 접근 후 엔드포인트 사용.