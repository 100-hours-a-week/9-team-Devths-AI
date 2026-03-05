"""
Celery Application Configuration — ADR-094, ADR-096, ADR-101, ADR-102, ADR-117.

Celery Beat 스케줄러 설정 및 앱 초기화.
Redis를 메시지 브로커 및 결과 저장소로 사용.

Beat 스케줄:
- crawl-trend-weekly : Phase 1 (ADR-094) — URL 기반 WebBaseLoader 크롤링
- search-trend-weekly: Phase 2 (ADR-101) — Tavily 검색 쿼리 기반 자동 발견 (TAVILY_API_KEY 필요)
- sync-training-data-weekly: ADR-117 — VectorDB → S3 학습 데이터 동기화

ADR-102 태스크:
- text_extract_tasks : 이력서/채용공고 텍스트 추출 + 임베딩
- masking_tasks      : PII 마스킹 처리
- evaluation_tasks   : 면접 Q&A VectorDB 적재

ADR-117 태스크:
- sagemaker_sync_tasks: VectorDB → S3 동기화 + SageMaker Pipeline 트리거
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
    include=[
        "app.tasks.trend_tasks",
        "app.tasks.text_extract_tasks",  # ADR-102
        "app.tasks.masking_tasks",  # ADR-102
        "app.tasks.evaluation_tasks",  # ADR-102
        "app.tasks.sagemaker_sync_tasks",  # ADR-117
    ],
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
    # ADR-102: 태스크 라우팅 설정
    task_routes={
        "app.tasks.trend_tasks.*": {"queue": "trend_crawl"},
        "app.tasks.text_extract_tasks.*": {"queue": "text_extract"},
        "app.tasks.masking_tasks.*": {"queue": "masking"},
        "app.tasks.evaluation_tasks.*": {"queue": "evaluation"},
        "app.tasks.sagemaker_sync_tasks.*": {"queue": "sagemaker_sync"},  # ADR-117
    },
    # 기본 큐 설정
    task_default_queue="celery",
)

# Celery Beat 스케줄 설정
celery_app.conf.beat_schedule = {
    # Phase 1 (ADR-094): URL 기반 WebBaseLoader 크롤링
    "crawl-trend-weekly": {
        "task": "app.tasks.trend_tasks.crawl_trend_urls_task",
        "schedule": crontab(
            minute=TREND_CRAWL_CRON_MINUTE,
            hour=TREND_CRAWL_CRON_HOUR,
            day_of_week=TREND_CRAWL_CRON_DAY_OF_WEEK,
        ),
        "options": {"queue": "trend_crawl"},
    },
    # Phase 2 (ADR-101): Tavily 검색 쿼리 기반 자동 발견
    # TREND_CRAWL_QUERIES 환경변수 비설정 시 태스크 내부에서 "skipped" 반환 (에러 없음)
    # TAVILY_API_KEY 미설정 시 태스크 실패 → retry 3회
    "search-trend-weekly": {
        "task": "app.tasks.trend_tasks.search_trend_queries_task",
        "schedule": crontab(
            minute="30",  # URL 크롤링 30분 후 실행
            hour=TREND_CRAWL_CRON_HOUR,
            day_of_week=TREND_CRAWL_CRON_DAY_OF_WEEK,
        ),
        "options": {"queue": "trend_crawl"},
    },
    # ADR-117: VectorDB → S3 학습 데이터 동기화 (트렌드 크롤링 1시간 후 실행)
    "sync-training-data-weekly": {
        "task": "app.tasks.sagemaker_sync_tasks.sync_training_data_task",
        "schedule": crontab(
            minute="0",
            hour=str((int(TREND_CRAWL_CRON_HOUR) + 1) % 24),  # 크롤링 1시간 후 (23→0 wrap)
            day_of_week=TREND_CRAWL_CRON_DAY_OF_WEEK,
        ),
        "options": {"queue": "sagemaker_sync"},
    },
}
