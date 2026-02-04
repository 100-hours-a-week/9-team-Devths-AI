import logging
import os
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import ai, masking
from app.middlewares.cloudwatch_middleware import CloudWatchMiddleware
from app.services.cloudwatch_service import CloudWatchService

# .env 파일 로드
load_dotenv()

# ============================================================================
# 로깅 설정 (운영 서버 호환)
# ============================================================================


def setup_logging():
    """운영 환경과 로컬 환경 모두에서 로그가 출력되도록 설정"""
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
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

    # 앱 로거 설정
    app_logger = logging.getLogger("app")
    app_logger.setLevel(logging.INFO)

    return root_logger


# 로깅 초기화
setup_logging()

logger = logging.getLogger(__name__)
logger.info("=" * 60)
logger.info("🚀 AI Server logging initialized")
logger.info(f"   Log level: {os.getenv('LOG_LEVEL', 'INFO')}")
logger.info("=" * 60)


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="AI Server API",
    description="FastAPI 기반 AI Server API",
    version="1.0.0",
    contact={
        "name": "AI Server Support",
        "email": "ai-support@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc
    openapi_url="/openapi.json",  # OpenAPI 스키마
)

# CORS 미들웨어 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# 라우터 등록
app.include_router(ai.router)
app.include_router(masking.router)


app.add_middleware(CloudWatchMiddleware)

@app.on_event("startup")
async def startup_event():
    logger.info("🔧 Initializing CloudWatch Service...")
    CloudWatchService.get_instance()

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 Flushing CloudWatch metrics...")
    cw_service = CloudWatchService.get_instance()
    await cw_service.flush()


@app.get("/", tags=["Root"])
async def root():
    """
    루트 엔드포인트

    API 정보를 반환합니다.
    """
    return {
        "message": "Welcome to AI Server API",
        "version": "1.0.0",
        "total_apis": 9,
        "docs": "/docs",
        "redoc": "/redoc",
        "api_list": [
            {"id": 1, "endpoint": "POST /ai/ocr/extract", "type": "async"},
            {"id": 2, "endpoint": "POST /ai/file/embed", "type": "sync"},
            {"id": 3, "endpoint": "POST /ai/analyze", "type": "streaming"},
            {"id": 4, "endpoint": "POST /ai/interview/question", "type": "sync"},
            {"id": 5, "endpoint": "POST /ai/interview/save", "type": "sync"},
            {"id": 6, "endpoint": "POST /ai/interview/report", "type": "streaming"},
            {"id": 7, "endpoint": "POST /ai/chat", "type": "streaming"},
            {"id": 8, "endpoint": "POST /ai/calendar/parse", "type": "sync"},
            {"id": 9, "endpoint": "POST /ai/masking/draft", "type": "async"},
        ],
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
