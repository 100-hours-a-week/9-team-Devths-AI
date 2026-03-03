"""
Celery Tasks Package — ADR-094 채용 트렌드 크롤링 스케줄러.

Celery Beat를 통한 주기적 크롤링 태스크 관리.
"""

from app.tasks.celery_app import celery_app

__all__ = ["celery_app"]
