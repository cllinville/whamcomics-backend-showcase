# app/s3.py
import boto3, datetime, os
from .config import settings

def get_s3():
    kwargs = {
        "aws_access_key_id": settings.s3_access_key,
        "aws_secret_access_key": settings.s3_secret_key,
        "region_name": settings.s3_region,
        "config": boto3.session.Config(signature_version="s3v4"),
    }
    if settings.s3_endpoint:
        kwargs["endpoint_url"] = settings.s3_endpoint
    return boto3.client("s3", **kwargs)

def make_key(prefix: str, filename: str) -> str:
    safe = filename.replace(" ", "_")
    now = datetime.datetime.utcnow()
    return f"{prefix}/{now:%Y/%m/%d}/{now.timestamp():.0f}-{safe}"

def upload_bytes(key: str, data: bytes, content_type: str = "image/png") -> str:
    s3 = get_s3()

    s3.put_object(
        Bucket=settings.s3_bucket,
        Key=key,
        Body=data,
        ContentType=content_type,
    )

    return key

def generate_presigned_get_url(key: str, expires_in: int = 3600) -> str:
    s3 = get_s3()

    return s3.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": settings.s3_bucket,
            "Key": key,
        },
        ExpiresIn=expires_in,
    )

def download_bytes(key: str) -> bytes:
    s3 = get_s3()

    response = s3.get_object(
        Bucket=settings.s3_bucket,
        Key=key,
    )

    return response["Body"].read()