# 1. 의존성 동기화
uv sync

# 2. 가상환경 활성화 (여전히 필요!)
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# 3. 이제 코드 실행
uvicorn http_video_server:app --host 0.0.0.0 --port 8000