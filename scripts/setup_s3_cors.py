"""
Apply CORS policy to the S3/MinIO bucket so browsers can upload directly.

Works with AWS S3 (production) and newer MinIO (RELEASE.2022+).
For older MinIO, see the fallback instructions printed if this fails.

Usage:
    python scripts/setup_s3_cors.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.s3 import get_s3
from app.config import settings

CORS_CONFIG = {
    "CORSRules": [
        {
            "AllowedOrigins": [
                "https://www.whamcomics.com",
                "https://whamcomics.com",
                "http://localhost:3000",
            ],
            "AllowedMethods": ["GET", "POST", "PUT", "HEAD"],
            "AllowedHeaders": ["*"],
            "ExposeHeaders": ["ETag", "Content-Length"],
            "MaxAgeSeconds": 3000,
        }
    ]
}


def main():
    s3 = get_s3()
    bucket = settings.s3_bucket
    endpoint = settings.s3_endpoint

    print(f"Applying CORS to bucket '{bucket}' at {endpoint}...")

    try:
        s3.put_bucket_cors(Bucket=bucket, CORSConfiguration=CORS_CONFIG)
    except Exception as e:
        if "NotImplemented" in str(e):
            print("")
            print("WARNING: This MinIO version does not support PutBucketCors via S3 API.")
            print("For local dev, set CORS using one of these options:")
            print("")
            print("Option A - MinIO Console (easiest):")
            print(f"  1. Open http://localhost:9001")
            print(f"  2. Go to Buckets -> {bucket} -> Anonymous")
            print(f"  3. Set prefix '/' with read+write access")
            print(f"  Note: Full CORS control requires MinIO RELEASE.2022-05-26 or later.")
            print("")
            print("Option B - Upgrade MinIO to get PutBucketCors support:")
            print("  https://min.io/download")
            print("")
            print("Option C - mc CLI (if installed):")
            print(f"  mc alias set local http://localhost:9000 minioadmin minioadmin")
            print(f"  mc anonymous set download local/{bucket}")
            print("")
            print("For production AWS S3, re-run this script and it will apply correctly.")
            sys.exit(1)
        raise

    result = s3.get_bucket_cors(Bucket=bucket)
    rules = result.get("CORSRules", [])
    print(f"Done. {len(rules)} CORS rule(s) active.")
    for rule in rules:
        print(f"  Origins : {rule['AllowedOrigins']}")
        print(f"  Methods : {rule['AllowedMethods']}")


if __name__ == "__main__":
    main()
