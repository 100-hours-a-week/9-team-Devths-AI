"""
Celery Application Configuration — ADR-094.

Celery Beat 스케줄러 설정 및 앱 초기화.
"""

import os

from celery import Celery
from celery.schedules import crontab

# 환경변수에서 설정 로드
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

# 트렌드 크롤링 스케줄 (기본: 매주 월요일 오전 9시)
TREND_CRAWL_CRON_MINUTE = os.getenv("TREND_CRAWL_CRON_MINUTE", "0")
TREND_CRAWL_CRON_HOUR = os.getenv("TREND_CRAWL_CRON_HOUR", "9")
TREND_CRAWL_CRON_DAY_OF_WEEK = os.getenv("TREND_CRAWL_CRON_DAY_OF_WEEK", "1")  # 월요일

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
