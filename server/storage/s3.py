from __future__ import annotations
import boto3
from typing import Tuple, Dict, Any
from ..config import S3_BUCKET_NAME, S3_REGION, AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, USE_S3
import os

s3_client = None
if USE_S3:
    try:
        s3_client = boto3.client(
            's3',
            region_name=S3_REGION,
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY
        )
        print("S3 클라이언트 초기화 성공")
    except Exception as e:
        print(f"S3 클라이언트 초기화 실패: {e}")
else:
    print("S3 환경변수가 설정되지 않음")


def _guess_content_type(filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()
    return {
        '.mp4': 'video/mp4',
        '.m4v': 'video/x-m4v',
        '.mov': 'video/quicktime',
        '.avi': 'video/x-msvideo',
        '.mkv': 'video/x-matroska',
        '.webm': 'video/webm',
    }.get(ext, 'application/octet-stream')


def upload_recording(file_path: str, filename: str) -> Tuple[bool, str]:
    if not USE_S3 or not s3_client:
        return False, "S3 not configured"
    try:
        key = f"recordings/{filename}"
        ct = _guess_content_type(filename)
        print(f"[UPLOAD] 시작: bucket={S3_BUCKET_NAME} region={S3_REGION} key={key} path={file_path} contentType={ct}")
        s3_client.upload_file(
            file_path,
            S3_BUCKET_NAME,
            key,
            ExtraArgs={
                "ContentType" : ct,
                "ContentDisposition": f'attachment; filename="{filename}"'
            }
        )
        url = f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{key}"
        print(f"[UPLOAD] 성공: {filename} -> {url}")
        return True, url

    except Exception as e:
        print(f"[UPLOAD] 실패: {filename} error={e}")
        return False, str(e)


def list_recordings() -> Dict[str, Any]:
    if not USE_S3 or not s3_client:
        return {"error": "S3 연결 오류: 환경변수를 확인하세요"}
    try:
        response = s3_client.list_objects_v2(Bucket=S3_BUCKET_NAME, Prefix="recordings/")
        files = []
        for obj in response.get('Contents', []):
            key = obj['Key']
            if not (key.endswith('.mp4')):
                continue
            filename = key.replace('recordings/', '')
            files.append({
                'filename': filename,
                'size': obj['Size'],
                'last_modified': obj['LastModified'].isoformat(),
                'url': f"https://{S3_BUCKET_NAME}.s3.{S3_REGION}.amazonaws.com/{key}",
                'storage': 'S3'
            })
        files.sort(key=lambda x: x['last_modified'], reverse=True)
        return {'recordings': files, 'storage': 'S3'}
    except Exception as e:
        return {'error': f'S3 error: {e}'}


def generate_presigned_url(filename: str):
    if not USE_S3 or not s3_client:
        return {"error": "S3 연결 오류: 환경변수를 확인하세요"}
    try:
        s3_client.head_object(Bucket=S3_BUCKET_NAME, Key=f"recordings/{filename}")
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': S3_BUCKET_NAME, 'Key': f"recordings/{filename}"},
            ExpiresIn=3600
        )
        return {"url": url, "filename": filename, "storage": "S3", "error": "no error"}
    except Exception as e:
        code = getattr(getattr(e, 'response', {}).get('Error', {}), 'get', lambda *_: None)('Code') if hasattr(e, 'response') else None
        if code == '404':
            return {"error": "File not found"}
        return {"error": f"S3 error: {e}"}

__all__ = ['s3_client','upload_recording','list_recordings','generate_presigned_url']
