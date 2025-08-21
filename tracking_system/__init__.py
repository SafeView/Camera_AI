"""
트래킹 시스템 패키지
"""

from .face_tracker import FaceTracker
from .face_detector import FaceDetector
from .person_detector import PersonDetector
from .embedding_processor import EmbeddingProcessor
from .pinecone_manager import PineconeManager
from .server_manager import ServerManager
from .dependencies import Dependencies
from .models import (
    FaceDetection, PersonDetection, TrackingTarget, 
    TrackingState, TrackingResult, TrackingConfig,
    StoredPerson, RecognitionResult
)
from .stored_person_manager import StoredPersonManager
from .video_processor import VideoProcessor
from .video_tracker import VideoTracker

__all__ = [
    'FaceTracker',
    'FaceDetector', 
    'PersonDetector',
    'EmbeddingProcessor',
    'PineconeManager',
    'ServerManager',
    'Dependencies',
    'FaceDetection',
    'PersonDetection', 
    'TrackingTarget',
    'TrackingState',
    'TrackingResult',
    'TrackingConfig',
    'StoredPerson',
    'RecognitionResult',
    'StoredPersonManager',
    'VideoProcessor',
    'VideoTracker',
    'run_app'
]

def run_app():
    """트래킹 시스템 실행"""
    import os
    import sys
    
    # 현재 디렉토리의 비디오 파일들 찾기
    video_files = []
    for file in os.listdir('.'):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_files.append(file)
    
    if not video_files:
        print("❌ 현재 디렉토리에 비디오 파일이 없습니다.")
        print("지원 형식: .mp4, .avi, .mov, .mkv")
        return
    
    print("📹 사용 가능한 비디오 파일:")
    for i, video_file in enumerate(video_files, 1):
        print(f"  {i}. {video_file}")
    
    # 사용자 선택
    while True:
        try:
            choice = input(f"\n비디오 파일을 선택하세요 (1-{len(video_files)}): ").strip()
            if choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(video_files):
                    selected_video = video_files[choice_idx]
                    break
                else:
                    print(f"❌ 1-{len(video_files)} 사이의 숫자를 입력하세요.")
            else:
                print("❌ 숫자를 입력하세요.")
        except KeyboardInterrupt:
            print("\n🛑 취소되었습니다.")
            return
    
    print(f"🎬 선택된 비디오: {selected_video}")
    
    # VideoTracker 초기화 및 실행
    tracker = VideoTracker()
    
    if tracker.load_video(selected_video):
        tracker.run()
    else:
        print("❌ 비디오 로드에 실패했습니다.")
