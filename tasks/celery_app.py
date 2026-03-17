from celery import Celery
from config.settings import settings

app = Celery(
    "video_edit_agent",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)
app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    task_acks_late=True,
    worker_max_tasks_per_child=10,
    task_track_started=True,
)
