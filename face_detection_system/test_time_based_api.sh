#!/bin/bash

# 시간 기반 얼굴 검출 API curl 테스트 스크립트

BASE_URL="http://localhost:8000"
VIDEO_FILE="uploads/20250816_171845_korean_face_detection_test.mp4"

echo "🤖 시간 기반 얼굴 검출 API curl 테스트"
echo "=========================================="

# 1. API 연결 확인
echo "1. API 연결 확인..."
curl -X GET "$BASE_URL/" | jq '.'

echo -e "\n=========================================="

# 2. 비디오 업로드
echo "2. 비디오 업로드..."
if [ -f "$VIDEO_FILE" ]; then
    UPLOAD_RESPONSE=$(curl -X POST "$BASE_URL/upload-video" \
        -F "file=@$VIDEO_FILE" \
        -H "Content-Type: multipart/form-data")
    
    echo "$UPLOAD_RESPONSE" | jq '.'
    
    # 업로드된 파일명 추출
    FILENAME=$(echo "$UPLOAD_RESPONSE" | jq -r '.filename')
    echo "업로드된 파일명: $FILENAME"
else
    echo "❌ 비디오 파일을 찾을 수 없습니다: $VIDEO_FILE"
    exit 1
fi

echo -e "\n=========================================="

# 3. 비디오 정보 조회
echo "3. 비디오 정보 조회..."
curl -X GET "$BASE_URL/video-info/$FILENAME" | jq '.'

echo -e "\n=========================================="

# 4. 0초부터 얼굴 검출 (전체 비디오)
echo "4. 0초부터 얼굴 검출 (전체 비디오)..."
DETECT_RESPONSE=$(curl -X POST "$BASE_URL/detect-faces?filename=$FILENAME&time_input=0%200")

echo "$DETECT_RESPONSE" | jq '.'

# 결과 ID 추출
RESULT_ID=$(echo "$DETECT_RESPONSE" | jq -r '.result_id')
echo "결과 ID: $RESULT_ID"

echo -e "\n=========================================="

# 5. 30초부터 얼굴 검출
echo "5. 30초부터 얼굴 검출..."
curl -X POST "$BASE_URL/detect-faces?filename=$FILENAME&time_input=0%2030" | jq '.'

echo -e "\n=========================================="

# 6. 1분 30초부터 얼굴 검출
echo "6. 1분 30초부터 얼굴 검출..."
curl -X POST "$BASE_URL/detect-faces?filename=$FILENAME&time_input=1%2030" | jq '.'

echo -e "\n=========================================="

# 7. 90초부터 얼굴 검출 (초 단위 입력)
echo "7. 90초부터 얼굴 검출 (초 단위 입력)..."
curl -X POST "$BASE_URL/detect-faces?filename=$FILENAME&time_input=90" | jq '.'

echo -e "\n=========================================="

# 8. 잘못된 시간 입력 테스트
echo "8. 잘못된 시간 입력 테스트..."
curl -X POST "$BASE_URL/detect-faces?filename=$FILENAME&time_input=invalid_time" | jq '.'

echo -e "\n=========================================="

# 9. 결과 조회
echo "9. 결과 조회..."
curl -X GET "$BASE_URL/results/$RESULT_ID" | jq '.'

echo -e "\n=========================================="

# 10. 모든 결과 목록
echo "10. 모든 결과 목록..."
curl -X GET "$BASE_URL/results" | jq '.'

echo -e "\n=========================================="

# 11. 얼굴 이미지 다운로드 (첫 번째 얼굴)
echo "11. 얼굴 이미지 다운로드 테스트..."
if [ ! -z "$RESULT_ID" ]; then
    # 결과에서 첫 번째 얼굴 파일명 가져오기
    FIRST_FACE=$(curl -s -X GET "$BASE_URL/results/$RESULT_ID" | jq -r '.faces[0].filename')
    if [ "$FIRST_FACE" != "null" ] && [ ! -z "$FIRST_FACE" ]; then
        echo "첫 번째 얼굴 이미지 다운로드: $FIRST_FACE"
        curl -X GET "$BASE_URL/download-face/$RESULT_ID/$FIRST_FACE" -o "downloaded_face.jpg"
        echo "✅ 얼굴 이미지가 'downloaded_face.jpg'로 저장되었습니다."
    else
        echo "❌ 다운로드할 얼굴 이미지가 없습니다."
    fi
fi

echo -e "\n=========================================="
echo "✅ 시간 기반 얼굴 검출 API 테스트 완료!"
echo ""
echo "📁 결과 파일 위치:"
echo "   - API 결과: api_results/"
echo "   - 업로드된 비디오: uploads/"
echo "   - 다운로드된 얼굴: downloaded_face.jpg"
