#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시간 기반 얼굴 검출 API 실행 스크립트
"""

import os
import sys
import uvicorn
from pathlib import Path

def check_dependencies():
    """필요한 의존성 확인"""
    required_packages = [
        'fastapi',
        'uvicorn',
        'opencv-python',
        'ultralytics',
        'face_recognition',
        'numpy',
        'scikit-learn'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ 다음 패키지들이 설치되지 않았습니다:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n다음 명령어로 설치하세요:")
        print("pip install -r requirements_time_based_api.txt")
        return False
    
    return True

def create_directories():
    """필요한 디렉토리 생성"""
    directories = [
        "uploads",
        "api_results",
        "face_detection/yolov8_face/weights"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ 디렉토리 생성/확인: {directory}")

def main():
    """메인 실행 함수"""
    print("🤖 시간 기반 얼굴 검출 API 시작")
    print("=" * 50)
    
    # 의존성 확인
    if not check_dependencies():
        sys.exit(1)
    
    # 디렉토리 생성
    create_directories()
    
    print("\n🚀 API 서버를 시작합니다...")
    print("📡 서버 주소: http://localhost:8000")
    print("📚 API 문서: http://localhost:8000/docs")
    print("🔧 ReDoc 문서: http://localhost:8000/redoc")
    print("\n⏹️  서버를 중지하려면 Ctrl+C를 누르세요.")
    print("=" * 50)
    
    try:
        # API 서버 실행
        uvicorn.run(
            "time_based_face_detection_api:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n\n🛑 API 서버가 중지되었습니다.")
    except Exception as e:
        print(f"\n❌ 서버 실행 중 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
