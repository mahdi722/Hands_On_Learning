from celery import Celery
from opentelemetry.instrumentation.celery import CeleryInstrumentor
import os

CeleryInstrumentor().instrument()
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

celery_app = Celery(
    "agent_tasks",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=3600,  
    worker_concurrency=4,  
    task_track_started=True,
    broker_heartbeat=10,
    include=["tasks"],
)
