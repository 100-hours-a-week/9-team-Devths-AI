from prometheus_client import Counter, Gauge, Histogram

# ============================================================================
# 1. HTTP 인프라 지표 (FastAPI Middleware)
# ============================================================================

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP 요청 처리 소요 시간 (초 단위)",
    ["method", "path", "status"],
)

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total", "HTTP 요청 총 누적 건수", ["method", "path", "status"]
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress", "현재 처리 중인 동시 HTTP 요청 수", ["method", "path"]
)

# ============================================================================
# 2. AI (LLM) 특화 지표 (Domain / Service)
# ============================================================================

AI_TIME_TO_FIRST_TOKEN = Histogram(
    "ai_time_to_first_token_seconds",
    "AI 모델이 첫 번째 토큰을 생성하기까지 걸린 대기 시간 (TTFT)",
    ["model", "endpoint"],
)

AI_GENERATION_DURATION = Histogram(
    "ai_generation_duration_seconds",
    "첫 토큰 이후 전체 응답이 완성될 때까지 걸린 생성 소요 시간",
    ["model", "endpoint"],
)

AI_BACKGROUND_JOB_COUNT = Counter(
    "ai_background_job_count_total", "백그라운드 AI 작업 누적 처리 건수", ["job_type", "status"]
)

# ============================================================================
# 3. 추가 커스텀 인프라/API 통합 지표 (Custom Metrics)
# ============================================================================

VECTOR_DB_INSERT_DURATION = Histogram(
    "vector_db_insert_duration_seconds",
    "문서 파싱 후 VectorDB 인서트에 걸린 지연 시간",
    ["collection_name"],
)

EXTERNAL_API_ERRORS = Counter(
    "external_api_errors_total",
    "외부 AI 연동 API 실패 횟수 (LLM, VQA 등)",
    ["provider", "error_type"],
)

CELERY_TASK_WAIT_TIME = Histogram(
    "celery_task_wait_time_seconds", "Celery 작업이 큐에서 워커 실행까지 대기한 시간", ["task_name"]
)

CELERY_TASKS_ACTIVE = Gauge(
    "celery_tasks_active_gauge", "동시에 실행 중인 Celery Task 개수", ["task_name"]
)

AI_PROMPT_INJECTION_BLOCKED = Counter(
    "ai_prompt_injection_blocked_total", "가드레일에 의해 차단된 프롬프트 인젝션 횟수", ["endpoint"]
)
