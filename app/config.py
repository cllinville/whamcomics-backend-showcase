from pydantic import BaseModel
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

def _fix_db_url(url: str) -> str:
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg2://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg2://", 1)
    return url

class Settings(BaseModel):
    app_env: str = os.getenv("APP_ENV", "dev")
    secret_key: str = os.getenv("SECRET_KEY", "dev-secret")
    database_url: str = _fix_db_url(os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/wham"))
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    s3_endpoint: str = os.getenv("S3_ENDPOINT", "")
    s3_region: str = os.getenv("S3_REGION", "us-east-1")
    s3_access_key: str = os.getenv("S3_ACCESS_KEY", "minioadmin")
    s3_secret_key: str = os.getenv("S3_SECRET_KEY", "minioadmin")
    s3_bucket: str = os.getenv("S3_BUCKET", "wham-dev")
    stripe_secret: str = os.getenv("STRIPE_SECRET_KEY", "")
    stripe_webhook_secret: str = os.getenv("STRIPE_WEBHOOK_SECRET", "")
    stripe_success_url: str = os.getenv("STRIPE_SUCCESS_URL", "http://localhost:3000/success?session_id={CHECKOUT_SESSION_ID}")
    stripe_cancel_url: str = os.getenv("STRIPE_CANCEL_URL", "http://localhost:3000/cancel")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    fal_api_key: str = os.getenv("FAL_API_KEY", "")
    use_lora: bool = os.getenv("USE_LORA", "false").lower() == "true"
    use_pulid: bool = os.getenv("USE_PULID", "false").lower() == "true"
    use_instantid: bool = os.getenv("USE_INSTANTID", "false").lower() == "true"
    style_lora_url: str = os.getenv("STYLE_LORA_URL", "")
    style_lora_trigger: str = os.getenv("STYLE_LORA_TRIGGER", "")
    base_url: str = os.getenv("BASE_URL", "https://whamcomics-backend-production.up.railway.app")
    brevo_api_key: str = os.getenv("BREVO_API_KEY", "")
    owner_email: str = os.getenv("OWNER_EMAIL", "organizedfuturellc@gmail.com")
    llamagen_api_key: str = os.getenv("LLAMAGEN_API_KEY", "")
    use_llamagen: bool = os.getenv("USE_LLAMAGEN", "false").lower() == "true"
    llamagen_preset: str = os.getenv("LLAMAGEN_PRESET", "western comic")
    llamagen_max_concurrent: int = int(os.getenv("LLAMAGEN_MAX_CONCURRENT", "4"))

    # Lulu Print API (print-on-demand fulfillment)
    lulu_client_key: str = os.getenv("LULU_CLIENT_KEY", "")
    lulu_client_secret: str = os.getenv("LULU_CLIENT_SECRET", "")
    lulu_base_url: str = os.getenv("LULU_BASE_URL", "https://api.sandbox.lulu.com")
    # Placeholder default: 6.625" x 10.25" comic trim, full color, standard quality,
    # saddle stitch, 60# uncoated white paper, gloss cover. VERIFY against Lulu's
    # sandbox / price calculator before submitting any real production order.
    lulu_pod_package_id: str = os.getenv("LULU_POD_PACKAGE_ID", "0663X1025.FC.STD.SS.060UW444.GXX")
    lulu_shipping_level: str = os.getenv("LULU_SHIPPING_LEVEL", "MAIL")
    lulu_contact_email: str = os.getenv("LULU_CONTACT_EMAIL", "") or owner_email
    lulu_production_delay: int = int(os.getenv("LULU_PRODUCTION_DELAY", "120"))

settings = Settings()
