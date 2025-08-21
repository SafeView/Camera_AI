# 시간 기반 얼굴 검출 시스템

사용자가 분, 초를 입력하면 그때부터 얼굴 이미지를 저장하는 시스템입니다.

## 📁 폴더 구조

```
time_based_face_detection/
├── README.md                    # 이 파일
├── requirements.txt             # 필요한 패키지들
├── main.py                      # 메인 실행 파일
├── face_detector.py             # 얼굴 검출 엔진
├── video_processor.py           # 비디오 처리
├── api_server.py                # FastAPI 서버
├── test_client.py               # 테스트 클라이언트
├── uploads/                     # 업로드된 비디오 저장
└── results/                     # 검출 결과 저장
```

## 🚀 설치 및 실행

### 1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 2. 서버 실행
```bash
python main.py
```

### 3. 테스트
```bash
python test_client.py
```

## 📡 API 엔드포인트

- `POST /upload-video` - 비디오 업로드
- `POST /detect-faces-interactive` - 시간 입력 얼굴 검출
- `GET /video-info/{filename}` - 비디오 정보 조회
- `GET /results/{result_id}` - 검출 결과 조회

## 💡 사용법

1. 비디오 파일을 업로드
2. 원하는 시간 입력 (예: "1 30" = 1분 30초부터)
3. 해당 시간부터 비디오 끝까지 얼굴 검출
4. 검출된 얼굴 이미지들이 `results/` 폴더에 저장

## 🎯 주요 기능

- ✅ 시간 기반 얼굴 검출
- ✅ 정면/옆모습 구분
- ✅ 신뢰도 기반 필터링
- ✅ 자동 이미지 저장
- ✅ REST API 제공
