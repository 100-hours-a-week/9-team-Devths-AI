"""
Celery Application Configuration — ADR-094, ADR-096.

Celery Beat 스케줄러 설정 및 앱 초기화.
Redis를 메시지 브로커 및 결과 저장소로 사용.
"""

from celery import Celery
from celery.schedules import crontab

from app.config.settings import get_settings

# settings.py에서 설정 로드 (ADR-096: 설정 일관성)
_settings = get_settings()

CELERY_BROKER_URL = _settings.celery_broker_url
CELERY_RESULT_BACKEND = _settings.celery_result_backend

# 트렌드 크롤링 스케줄 설정
TREND_CRAWL_CRON_MINUTE = _settings.trend_crawl_cron_minute
TREND_CRAWL_CRON_HOUR = _settings.trend_crawl_cron_hour
TREND_CRAWL_CRON_DAY_OF_WEEK = _settings.trend_crawl_cron_day_of_week

celery_app = Celery(
    "devths_ai",
    broker=CELERY_BROKER_URL,
    backend=CELERY_RESULT_BACKEND,
    include=["app.tasks.trend_tasks"],
)

# Celery 설정
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Seoul",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,  # 1시간 타임아웃
    worker_prefetch_multiplier=1,
    task_acks_late=True,
)

# Celery Beat 스케줄 설정 (ADR-094: 1주 주기)
celery_app.conf.beat_schedule = {
    "crawl-trend-weekly": {
        "task": "app.tasks.trend_tasks.crawl_trend_urls_task",
        "schedule": crontab(
            minute=TREND_CRAWL_CRON_MINUTE,
            hour=TREND_CRAWL_CRON_HOUR,
            day_of_week=TREND_CRAWL_CRON_DAY_OF_WEEK,
        ),
        "options": {"queue": "trend_crawl"},
    },
}
