import base64
import hashlib
import hmac
import time

import httpx

from app.config import settings

_TOKEN_URL_PATH = "/auth/realms/glasstree/protocol/openid-connect/token"

# Cached in-process; re-fetched a bit before real expiry.
_token_cache: dict = {"access_token": None, "expires_at": 0}


class LuluAPIError(RuntimeError):
    def __init__(self, message: str, status_code: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


def _basic_auth_header() -> str:
    raw = f"{settings.lulu_client_key}:{settings.lulu_client_secret}".encode()
    return "Basic " + base64.b64encode(raw).decode()


def get_access_token(force_refresh: bool = False) -> str:
    if not settings.lulu_client_key or not settings.lulu_client_secret:
        raise LuluAPIError("Lulu client key/secret are not configured (LULU_CLIENT_KEY / LULU_CLIENT_SECRET)")

    if not force_refresh and _token_cache["access_token"] and time.time() < _token_cache["expires_at"]:
        return _token_cache["access_token"]

    with httpx.Client(timeout=30) as client:
        resp = client.post(
            f"{settings.lulu_base_url}{_TOKEN_URL_PATH}",
            data={"grant_type": "client_credentials"},
            headers={
                "Authorization": _basic_auth_header(),
                "Content-Type": "application/x-www-form-urlencoded",
            },
        )
    if resp.status_code >= 400:
        raise LuluAPIError(f"Lulu token request failed: {resp.status_code}", resp.status_code, resp.text)

    data = resp.json()
    token = data["access_token"]
    expires_in = data.get("expires_in", 3600)
    _token_cache["access_token"] = token
    _token_cache["expires_at"] = time.time() + expires_in - 30  # refresh 30s early
    return token


def _headers() -> dict:
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json",
    }


def _request(method: str, path: str, retry_on_401: bool = True, **kwargs) -> dict:
    with httpx.Client(timeout=30) as client:
        resp = client.request(method, f"{settings.lulu_base_url}{path}", headers=_headers(), **kwargs)

    if resp.status_code == 401 and retry_on_401:
        get_access_token(force_refresh=True)
        return _request(method, path, retry_on_401=False, **kwargs)

    if resp.status_code >= 400:
        raise LuluAPIError(f"Lulu API {method} {path} failed: {resp.status_code}", resp.status_code, resp.text)

    if resp.status_code == 204 or not resp.content:
        return {}
    return resp.json()


def build_shipping_address(shipping: dict, fallback_email: str) -> dict:
    """Map our stored shipping_address (Stripe-shaped) to Lulu's schema."""
    return {
        "name": shipping.get("name") or "",
        "street1": shipping.get("line1") or "",
        "street2": shipping.get("line2") or "",
        "city": shipping.get("city") or "",
        "state_code": shipping.get("state") or "",
        "postcode": shipping.get("postal_code") or "",
        "country_code": (shipping.get("country") or "US").upper(),
        "phone_number": shipping.get("phone") or "",
        "email": shipping.get("email") or fallback_email or "",
    }


def get_cover_dimensions(interior_page_count: int, pod_package_id: str | None = None, unit: str = "pt") -> dict:
    """Returns {"width": ..., "height": ..., "unit": ...} for the wrap cover PDF."""
    payload = {
        "pod_package_id": pod_package_id or settings.lulu_pod_package_id,
        "interior_page_count": interior_page_count,
        "unit": unit,
    }
    return _request("POST", "/cover-dimensions/", json=payload)


def create_print_job(
    *,
    external_id: str,
    title: str,
    quantity: int,
    cover_url: str,
    interior_url: str,
    shipping: dict,
    customer_email: str,
    pod_package_id: str | None = None,
) -> dict:
    payload = {
        "contact_email": settings.lulu_contact_email,
        "external_id": external_id,
        "shipping_level": settings.lulu_shipping_level,
        "production_delay": settings.lulu_production_delay,
        "shipping_address": build_shipping_address(shipping, customer_email),
        "line_items": [
            {
                "external_id": external_id,
                "title": title[:255],
                "quantity": quantity,
                "printable_normalization": {
                    "pod_package_id": pod_package_id or settings.lulu_pod_package_id,
                    "cover": {"source_url": cover_url},
                    "interior": {"source_url": interior_url},
                },
            }
        ],
    }
    return _request("POST", "/print-jobs/", json=payload)


def get_print_job(lulu_job_id: str) -> dict:
    return _request("GET", f"/print-jobs/{lulu_job_id}/")


def cancel_print_job(lulu_job_id: str) -> dict:
    return _request("PUT", f"/print-jobs/{lulu_job_id}/status/", json={"name": "CANCELED"})


def verify_webhook_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """Verify the Lulu-HMAC-SHA256 header against the raw request body."""
    if not signature_header or not settings.lulu_client_secret:
        return False
    key = settings.lulu_client_secret.encode()
    digest = hmac.new(key, raw_body, hashlib.sha256)
    candidates = {digest.hexdigest(), base64.b64encode(digest.digest()).decode()}
    return signature_header.strip() in candidates


def register_webhook(url: str, topics: list[str] | None = None) -> dict:
    payload = {"topics": topics or ["PRINT_JOB_STATUS_CHANGED"], "url": url}
    return _request("POST", "/webhooks/", json=payload)
