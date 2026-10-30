import base64 as b64lib
import httpx
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from typing import Literal, Optional
from ..config import settings
from ..s3 import get_s3, make_key, upload_bytes

router = APIRouter(prefix="/uploads", tags=["uploads"])

ALLOWED_CT = {
    "image/jpeg",
    "image/png",
    "image/webp",
    "image/heic",
}
MAX_BYTES = 10 * 1024 * 1024  # 10 MB

class SignReq(BaseModel):
    filename: str
    content_type: str
    prefix: Optional[Literal["raw","characters","pets","misc"]] = "raw"
    max_bytes: Optional[int] = MAX_BYTES

@router.post("/sign")
def sign_upload(body: SignReq):
    if body.content_type not in ALLOWED_CT:
        raise HTTPException(400, f"content_type not allowed: {body.content_type}")

    s3 = get_s3()
    key = make_key(body.prefix or "raw", body.filename)
    conditions = [
        {"bucket": settings.s3_bucket},
        ["starts-with", "$key", ""],
        ["content-length-range", 1, body.max_bytes or MAX_BYTES],
        {"Content-Type": body.content_type},
    ]
    fields = {"Content-Type": body.content_type}
    try:
        presigned = s3.generate_presigned_post(
            Bucket=settings.s3_bucket,
            Key=key,
            Fields=fields,
            Conditions=conditions,
            ExpiresIn=60 * 5,
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to create presigned POST: {e}")

    return {
        "upload_url": presigned["url"],
        "fields": presigned["fields"],
        "key": key,
        "public_url": f"{settings.s3_endpoint}/{settings.s3_bucket}/{key}",
    }


class FromUrlReq(BaseModel):
    url: str
    filename: str = "photo.jpg"


@router.post("/from-url")
def upload_from_url(body: FromUrlReq):
    """Download an image from a URL (e.g. Wix media) and store it in S3."""
    try:
        response = httpx.get(
            body.url,
            timeout=30,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "image/webp,image/apng,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.whamcomics.com/upload",
            },
        )
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch image from URL: {e}")

    content_type = response.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    if content_type not in ALLOWED_CT:
        content_type = "image/jpeg"

    if len(response.content) > MAX_BYTES:
        raise HTTPException(400, "Image exceeds 10MB limit")

    key = make_key("characters", body.filename)
    upload_bytes(key=key, data=response.content, content_type=content_type)

    return {"key": key}


@router.post("/raw")
async def upload_raw(request: Request, filename: str = "photo.jpg"):
    """Accept raw image bytes from a Wix backend proxy (no multipart, no CORS issues)."""
    content_type = request.headers.get("content-type", "image/jpeg").split(";")[0].strip()
    if content_type not in ALLOWED_CT:
        content_type = "image/jpeg"

    data = await request.body()
    if len(data) == 0:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(400, "Image exceeds 10MB")

    key = make_key("characters", filename)
    upload_bytes(key=key, data=data, content_type=content_type)
    return {"key": key}


@router.post("/direct")
async def upload_direct(file: UploadFile = File(...)):
    """Accept a raw file upload from the browser and store it in S3."""
    content_type = (file.content_type or "image/jpeg").split(";")[0].strip()
    if content_type not in ALLOWED_CT:
        content_type = "image/jpeg"

    data = await file.read()
    if len(data) == 0:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(400, "Image exceeds 10MB limit")

    key = make_key("characters", file.filename or "photo.jpg")
    upload_bytes(key=key, data=data, content_type=content_type)

    return {"key": key}


class Base64Req(BaseModel):
    data: str          # base64 string (no data URL prefix)
    filename: str = "photo.jpg"
    content_type: str = "image/jpeg"


@router.post("/base64")
def upload_base64(body: Base64Req):
    """Accept a base64-encoded image — used by Wix Velo frontend (avoids FormData/Blob issues)."""
    content_type = body.content_type.split(";")[0].strip()
    if content_type not in ALLOWED_CT:
        content_type = "image/jpeg"

    try:
        data = b64lib.b64decode(body.data)
    except Exception:
        raise HTTPException(400, "Invalid base64 data")

    if len(data) == 0:
        raise HTTPException(400, "Empty file")
    if len(data) > MAX_BYTES:
        raise HTTPException(400, "Image exceeds 10MB")

    key = make_key("characters", body.filename)
    upload_bytes(key=key, data=data, content_type=content_type)
    return {"key": key}

