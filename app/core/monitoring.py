from prometheus_client import Counter, Gauge, Histogram

# ============================================================================
# 1. HTTP 인프라 지표 (FastAPI Middleware)
# ============================================================================

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP 요청 처리 소요 시간 (초 단위)",
    ["method", "path", "status"]
)

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "HTTP 요청 총 누적 건수",
    ["method", "path", "status"]
)

HTTP_REQUESTS_IN_PROGRESS = Gauge(
    "http_requests_in_progress",
    "현재 처리 중인 동시 HTTP 요청 수",
    ["method", "path"]
)

# ============================================================================
# 2. AI (LLM) 특화 지표 (Domain / Service)
# ============================================================================

AI_TIME_TO_FIRST_TOKEN = Histogram(
    "ai_time_to_first_token_seconds",
    "AI 모델이 첫 번째 토큰을 생성하기까지 걸린 대기 시간 (TTFT)",
    ["model", "endpoint"]
)

AI_GENERATION_DURATION = Histogram(
    "ai_generation_duration_seconds",
    "첫 토큰 이후 전체 응답이 완성될 때까지 걸린 생성 소요 시간",
    ["model", "endpoint"]
)

AI_BACKGROUND_JOB_COUNT = Counter(
    "ai_background_job_count_total",
    "백그라운드 AI 작업 누적 처리 건수",
    ["job_type", "status"]
)
