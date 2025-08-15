#!/usr/bin/env python3
"""
얼굴 추적 시스템 메인 실행 파일

사용법:
    python main.py
"""

import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tracking_system import run_app

def main():
    """메인 함수"""
    try:
        run_app()
    except KeyboardInterrupt:
        print("\n🛑 프로그램이 사용자에 의해 중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
