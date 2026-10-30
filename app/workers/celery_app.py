from celery import Celery
import os

CELERY_BROKER_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
CELERY_BACKEND_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery = Celery(
    "wham",
    broker=CELERY_BROKER_URL,
    backend=CELERY_BACKEND_URL,
)
