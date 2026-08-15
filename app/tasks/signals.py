"""
Celery Prometheus 계측 — celery_tasks_active_gauge / celery_task_wait_time_seconds /
ai_background_job_count_total 지표를 실제로 채워 넣는 시그널 핸들러.

app/core/monitoring.py에 지표만 정의돼 있고 이를 갱신하는 코드가 없어서
계속 빈 값이었던 문제를 해결한다 (Celery 자체는 정상 동작 중이었음 — 관측만 안 됐음).

동시성(-c) > 1인 워커는 자식 프로세스마다 별도 인메모리 레지스트리를 가지므로,
prometheus_client의 multiprocess 모드(PROMETHEUS_MULTIPROC_DIR)로 부모 프로세스에서
전체 자식 프로세스 값을 합산해 /metrics로 노출한다.
"""

import os
import shutil
import threading
import time
from wsgiref.simple_server import make_server

from celery.signals import (
    before_task_publish,
    task_failure,
    task_postrun,
    task_prerun,
    task_success,
    worker_init,
)
from prometheus_client import CollectorRegistry, make_wsgi_app, multiprocess

from app.core.monitoring import (
    AI_BACKGROUND_JOB_COUNT,
    CELERY_TASK_WAIT_TIME,
    CELERY_TASKS_ACTIVE,
)

METRICS_PORT = int(os.environ.get("CELERY_METRICS_PORT", "9200"))


@worker_init.connect
def _setup_multiproc_metrics_server(**kwargs):
    """부모 프로세스에서 1회 — multiproc 디렉터리 초기화 + /metrics 서버 기동."""
    multiproc_dir = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not multiproc_dir:
        return

    if os.path.isdir(multiproc_dir):
        shutil.rmtree(multiproc_dir)
    os.makedirs(multiproc_dir, exist_ok=True)

    def _app(environ, start_response):
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)
        return make_wsgi_app(registry)(environ, start_response)

    httpd = make_server("0.0.0.0", METRICS_PORT, _app)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()


@before_task_publish.connect
def _stamp_enqueue_time(headers=None, **kwargs):
    """태스크가 큐에 들어간 시각을 헤더에 기록 — 워커가 나중에 대기시간 계산에 씀."""
    if headers is not None:
        headers["enqueued_at"] = time.time()


@task_prerun.connect
def _on_task_prerun(task=None, **kwargs):
    if task is None:
        return
    CELERY_TASKS_ACTIVE.labels(task_name=task.name).inc()

    enqueued_at = getattr(task.request, "enqueued_at", None)
    if enqueued_at is not None:
        CELERY_TASK_WAIT_TIME.labels(task_name=task.name).observe(time.time() - enqueued_at)


@task_postrun.connect
def _on_task_postrun(task=None, **kwargs):
    if task is None:
        return
    CELERY_TASKS_ACTIVE.labels(task_name=task.name).dec()


@task_success.connect
def _on_task_success(sender=None, **kwargs):
    if sender is None:
        return
    AI_BACKGROUND_JOB_COUNT.labels(job_type=sender.name, status="success").inc()


@task_failure.connect
def _on_task_failure(sender=None, **kwargs):
    if sender is None:
        return
    AI_BACKGROUND_JOB_COUNT.labels(job_type=sender.name, status="failure").inc()
