import logging
import os
import sys

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

import app.core.monitoring  # Register custom metrics
from app.api.middleware.metrics import PrometheusMiddleware
from app.api.routes import v2
from app.api.routes.v1 import ai as v1_ai
from app.api.routes.v1 import masking as v1_masking
from app.config.settings import get_settings

# ============================================================================
# 로깅 설정 (운영 서버 호환)
# ============================================================================

class EndpointFilter(logging.Filter):
    """특정 엔드포인트(/health, /metrics 등)의 접근 로그를 무시하는 필터"""
    def filter(self, record: logging.LogRecord) -> bool:
        # uvicorn.access 로그 레코드의 args 속성은 다음과 같은 튜플 형태입니다:
        # (client_addr, method, path, http_version, status_code)
        if record.args and len(record.args) >= 3:
            path = record.args[2]
            if path in ["/health", "/metrics", "/docs", "/redoc", "/openapi.json"]:
                return False
        return True


def setup_logging(settings):
    """운영 환경과 로컬 환경 모두에서 로그가 출력되도록 설정"""
    # 환경에 따른 로그 레벨 설정
    # SIM108: Use ternary operator
    default_level = "DEBUG" if settings.environment == "development" else "INFO"

    log_level = os.getenv("LOG_LEVEL", default_level).upper()
    log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    # 루트 로거 설정
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))

    # 기존 핸들러 제거 (중복 방지)
    root_logger.handlers.clear()

    # stdout 핸들러 (uvicorn이 캡처)
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    stdout_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    root_logger.addHandler(stdout_handler)

    # uvicorn 로거도 동일하게 설정
    for logger_name in ["uvicorn", "uvicorn.error", "uvicorn.access"]:
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.addHandler(stdout_handler)
        logger.setLevel(logging.INFO)
        # uvicorn.access 로그 중 무의미한 헬스체크 등은 필터링
        if logger_name == "uvicorn.access":
            logger.addFilter(EndpointFilter())

    # 앱 로거 설정
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)

    return root_logger


settings = get_settings()

# 로깅 초기화
setup_logging(settings)

logger = logging.getLogger(__name__)
logger.info("=" * 60)
logger.info("🚀 AI Server logging initialized")
logger.info(f"   Environment: {settings.environment}")
logger.info(f"   Log level: {os.getenv('LOG_LEVEL', 'INFO')}")
logger.info(f"   Debug mode: {settings.debug}")
logger.info("=" * 60)


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="AI Server API",
    description="FastAPI 기반 AI Server API",
    version="1.0.0",
    docs_url="/docs"
    if settings.environment == "development"
    else None,  # Dev에서만 Docs 노출 권장 (선택사항)
    redoc_url="/redoc" if settings.environment == "development" else None,
    openapi_url="/openapi.json" if settings.environment == "development" else None,
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus 모니터링 미들웨어 등록 (CORS 뒤에 붙여야 동작이 정확함)
app.add_middleware(PrometheusMiddleware)

# Prometheus Metrics 엔드포인트 마운트
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# 422 에러 핸들러 (디버깅용 상세 로그)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """422 Validation Error 발생 시 요청 body 로깅"""
    try:
        body = await request.body()
        body_str = body.decode("utf-8")[:2000]  # 최대 2000자
    except Exception:
        body_str = "[body 읽기 실패]"

    logger.error("=" * 80)
    logger.error("❌ [422 Validation Error]")
    logger.error(f"   URL: {request.url}")
    logger.error(f"   Method: {request.method}")
    logger.error(f"   Body: {body_str}")
    logger.error(f"   Errors: {exc.errors()}")
    logger.error("=" * 80)

    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


# 라우터 등록 (v2: 기본, v1: 하위 호환)
app.include_router(v2.router)  # v2: /ai/...
app.include_router(v1_ai.router)  # v1: /ai/v1/...
app.include_router(v1_masking.router)  # v1: /ai/v1/masking/...


@app.get("/", tags=["Root"])
async def root():
    """
    루트 엔드포인트

    API 정보를 반환합니다.
    """
    return {
        "message": "Welcome to AI Server API",
        "version": "2.0.0",
        "docs": "/docs",
        "redoc": "/redoc",
        "api_versions": {
            "v2 (current)": [
                {"endpoint": "POST /ai/text/extract", "type": "async"},
                {"endpoint": "GET /ai/task/{task_id}", "type": "sync"},
                {"endpoint": "POST /ai/chat", "type": "streaming"},
                {"endpoint": "POST /ai/calendar/parse", "type": "sync"},
                {"endpoint": "POST /ai/masking/draft", "type": "async"},
                {"endpoint": "GET /ai/masking/task/{task_id}", "type": "sync"},
                {"endpoint": "GET /ai/masking/health", "type": "sync"},
            ],
            "v1 (legacy)": [
                {"endpoint": "POST /ai/v1/text/extract", "type": "async"},
                {"endpoint": "GET /ai/v1/task/{task_id}", "type": "sync"},
                {"endpoint": "POST /ai/v1/chat", "type": "streaming"},
                {"endpoint": "POST /ai/v1/calendar/parse", "type": "sync"},
                {"endpoint": "POST /ai/v1/masking/draft", "type": "async"},
                {"endpoint": "GET /ai/v1/masking/task/{task_id}", "type": "sync"},
                {"endpoint": "GET /ai/v1/masking/health", "type": "sync"},
            ],
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    헬스 체크 엔드포인트

    서버 상태를 확인합니다.
    """
    return {"status": "healthy", "service": "ai-server", "version": "1.0.0"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
