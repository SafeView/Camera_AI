#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
훈련된 모델을 사용한 시간 기반 얼굴 검출 테스트
"""

import requests
import json
import os
from datetime import datetime

class TrainedModelTester:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.video_filename = None
    
    def test_api_connection(self):
        """API 연결 테스트"""
        try:
            response = requests.get(f"{self.base_url}/")
            if response.status_code == 200:
                data = response.json()
                print("✅ API 연결 성공")
                print(f"   메시지: {data['message']}")
                print(f"   버전: {data['version']}")
                print(f"   설명: {data['description']}")
                return True
            else:
                print(f"❌ API 연결 실패: {response.status_code}")
                return False
        except Exception as e:
            print(f"❌ API 연결 오류: {str(e)}")
            return False
    
    def upload_video(self, video_path):
        """비디오 업로드"""
        try:
            if not os.path.exists(video_path):
                print(f"❌ 비디오 파일을 찾을 수 없습니다: {video_path}")
                return False
            
            print(f"📤 비디오 업로드 중: {video_path}")
            with open(video_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(f"{self.base_url}/upload-video", files=files)
            
            if response.status_code == 200:
                result = response.json()
                self.video_filename = result['filename']
                print(f"✅ 비디오 업로드 성공: {self.video_filename}")
                return True
            else:
                print(f"❌ 비디오 업로드 실패: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ 비디오 업로드 오류: {str(e)}")
            return False
    
    def get_video_info(self):
        """비디오 정보 조회"""
        if not self.video_filename:
            print("❌ 업로드된 비디오가 없습니다")
            return None
        
        try:
            response = requests.get(f"{self.base_url}/video-info/{self.video_filename}")
            if response.status_code == 200:
                data = response.json()
                info = data['video_info']
                print(f"📹 비디오 정보:")
                print(f"   - 길이: {info['duration']:.1f}초 ({info['duration_formatted']})")
                print(f"   - 프레임 수: {info['total_frames']}")
                print(f"   - FPS: {info['fps']:.1f}")
                print(f"   - 해상도: {info['width']}x{info['height']}")
                return info
            else:
                print(f"❌ 비디오 정보 조회 실패: {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 비디오 정보 조회 오류: {str(e)}")
            return None
    
    def detect_faces_from_time(self, minutes, seconds):
        """특정 시간부터 얼굴 검출"""
        if not self.video_filename:
            print("❌ 업로드된 비디오가 없습니다")
            return None
        
        try:
            time_input = f"{minutes} {seconds}"
            print(f"🔍 {minutes}분 {seconds}초부터 얼굴 검출 시작...")
            print(f"   훈련된 얼굴 검출 모델 사용 중...")
            
            params = {
                'filename': self.video_filename,
                'time_input': time_input
            }
            
            response = requests.post(f"{self.base_url}/detect-faces", params=params)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ 얼굴 검출 완료!")
                print(f"   - 결과 ID: {result['result_id']}")
                print(f"   - 시작 시간: {result['detection_info']['start_time']}")
                print(f"   - 검출된 얼굴 수: {result['detection_info']['total_faces_detected']}")
                print(f"   - 저장된 얼굴 수: {result['detection_info']['unique_faces_saved']}")
                print(f"   - 처리 시간: {result['detection_info']['processing_duration']}")
                print(f"   - 중복 제거 방법: {result['detection_info']['duplicate_removal_method']}")
                return result
            else:
                print(f"❌ 얼굴 검출 실패: {response.status_code}")
                error_data = response.json()
                if 'error' in error_data:
                    print(f"   오류: {error_data['error']}")
                return None
                
        except Exception as e:
            print(f"❌ 얼굴 검출 오류: {str(e)}")
            return None
    
    def interactive_test(self):
        """인터랙티브 테스트"""
        print("🤖 훈련된 모델을 사용한 시간 기반 얼굴 검출 테스트")
        print("=" * 60)
        
        # 1. API 연결 확인
        if not self.test_api_connection():
            return
        
        # 2. 비디오 파일 찾기
        video_files = [
            "../uploads/20250816_175111_korean_face_detection_test.mp4",
            "test_video.mp4"
        ]
        
        video_path = None
        for vf in video_files:
            if os.path.exists(vf):
                video_path = vf
                break
        
        if not video_path:
            print("❌ 테스트할 비디오 파일을 찾을 수 없습니다")
            print("   다음 파일 중 하나를 준비해주세요:")
            for vf in video_files:
                print(f"   - {vf}")
            return
        
        # 3. 비디오 업로드
        if not self.upload_video(video_path):
            return
        
        # 4. 비디오 정보 조회
        video_info = self.get_video_info()
        if not video_info:
            return
        
        # 5. 사용자 입력 받기
        print("\n" + "=" * 60)
        print("💡 얼굴 검출을 시작할 시간을 입력하세요")
        print("   형식: 분 초 (예: 0 30 = 30초부터, 1 30 = 1분 30초부터)")
        print("   전체 비디오: 0 0")
        print("   종료: q")
        
        while True:
            try:
                user_input = input("\n시간 입력 (분 초): ").strip()
                
                if user_input.lower() == 'q':
                    print("👋 테스트를 종료합니다")
                    break
                
                if not user_input:
                    print("❌ 시간을 입력해주세요")
                    continue
                
                # 시간 파싱
                parts = user_input.split()
                if len(parts) == 2:
                    minutes = int(parts[0])
                    seconds = int(parts[1])
                elif len(parts) == 1:
                    total_seconds = int(parts[0])
                    minutes = total_seconds // 60
                    seconds = total_seconds % 60
                else:
                    print("❌ 잘못된 형식입니다. '분 초' 또는 '초' 형식으로 입력하세요")
                    continue
                
                # 입력 시간 검증
                input_time_seconds = minutes * 60 + seconds
                if input_time_seconds > video_info['duration']:
                    print(f"❌ 입력한 시간({minutes}분 {seconds}초)이 비디오 길이({video_info['duration']:.1f}초)를 초과합니다")
                    continue
                
                # 얼굴 검출 실행
                result = self.detect_faces_from_time(minutes, seconds)
                
                if result:
                    print(f"\n🎉 {minutes}분 {seconds}초부터 얼굴 검출이 완료되었습니다!")
                    print(f"   저장 위치: {result['detection_info']['faces_directory']}")
                    
                    # 검출된 얼굴 정보 표시
                    if result['faces']:
                        print(f"\n📸 검출된 얼굴들:")
                        for i, face in enumerate(result['faces'][:5]):  # 처음 5개만 표시
                            print(f"   {i+1}. 얼굴 {face['face_id']}: {face['detection_time']} (신뢰도: {face['confidence']:.2f})")
                        if len(result['faces']) > 5:
                            print(f"   ... 외 {len(result['faces']) - 5}개 더")
                    
                    # 계속할지 묻기
                    continue_test = input("\n다른 시간으로 테스트하시겠습니까? (y/n): ").strip().lower()
                    if continue_test != 'y':
                        break
                
            except ValueError:
                print("❌ 숫자를 입력해주세요")
            except KeyboardInterrupt:
                print("\n👋 테스트를 종료합니다")
                break
            except Exception as e:
                print(f"❌ 오류 발생: {str(e)}")

def main():
    """메인 함수"""
    tester = TrainedModelTester()
    tester.interactive_test()

if __name__ == "__main__":
    main()
