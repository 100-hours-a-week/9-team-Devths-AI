import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import ai, masking

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:     %(message)s'
)


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

# 라우터 등록
app.include_router(ai.router)
app.include_router(masking.router)


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
        ]
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """
    헬스 체크 엔드포인트

    서버 상태를 확인합니다.
    """
    return {
        "status": "healthy",
        "service": "ai-server",
        "version": "1.0.0"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
