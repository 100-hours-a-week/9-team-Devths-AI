import logging

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import ai, masking

# .env 파일 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")


# FastAPI 애플리케이션 생성
app = FastAPI(
    title="AI Server API",
    description="""
    FastAPI 기반 AI Server API입니다. Backend(Spring Boot)가 이 API들을 호출합니다.

    ## 📋 API 목록 (총 9개)

    ### 🔍 OCR 및 임베딩
    1. **POST /ai/ocr/extract** - OCR 텍스트 추출 + 임베딩 저장 (비동기)
    2. **POST /ai/file/embed** - 텍스트 직접 입력 시 임베딩 저장 (동기)

    ### 📊 분석 및 매칭
    3. **POST /ai/analyze** - 이력서/채용공고 분석 + 매칭도 (스트리밍)

    ### 🎤 모의 면접
    4. **POST /ai/interview/question** - 면접 질문/꼬리질문 생성 (동기)
    5. **POST /ai/interview/save** - 면접 Q&A 개별 저장 (동기)
    6. **POST /ai/interview/report** - 면접 평가 및 피드백 (스트리밍)

    ### 💬 채팅
    7. **POST /ai/chat** - 대화 처리 (RAG + 에이전트) (스트리밍)

    ### 📅 캘린더
    8. **POST /ai/calendar/parse** - 일정 정보 파싱 (동기)

    ### 🔒 개인정보 마스킹
    9. **POST /ai/masking/draft** - 게시판 첨부파일 1차 마스킹 (비동기)

    ### 🔄 비동기 작업 조회
    - **GET /ai/task/{task_id}** - 비동기 작업 상태 조회

    ## 🔧 처리 방식

    | 방식 | 아이콘 | 설명 | 사용 API |
    |------|-------|------|----------|
    | **동기** | ⚡ | 즉시 응답 반환 | 2, 4, 5, 8 |
    | **비동기** | 🔄 | task_id 반환 → 폴링 필요 | 1, 9 |
    | **스트리밍** | 📡 | SSE로 실시간 응답 전송 | 3, 6, 7 |

    ## 🔐 인증

    **API Key 기반 인증**
    - Header: `X-API-Key: your-api-key-here`

    ## 📚 기술 스택

    - **LLM/VLM:** Google Gemini 3 Flash Preview (face detection), Gemini 1.5 Flash (main), Gemini 1.5 Pro (fallback)
    - **OCR/PII:** datalab-to/chandra (text PII detection)
    - **Embedding:** Google text-embedding-004
    - **OCR:** PaddleOCR (local), Tesseract (fallback)
    - **VectorDB:** ChromaDB
    - **Framework:** FastAPI
    - **Processing:** LangChain (RAG), LangGraph (Agent)

    ## ⚠️ Rate Limits

    - 동기 API: 100 requests/min
    - 비동기 API: 50 requests/min
    - 스트리밍 API: 20 connections/min
    """,
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
