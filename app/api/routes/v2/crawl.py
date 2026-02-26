"""
Trend Data Crawling API — ADR-094/078

채용 트렌드 URL 크롤링 → 메타데이터 자동 태깅 → trend_data 컬렉션 적재.
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

router = APIRouter()


class CrawlRequest(BaseModel):
    """크롤링 요청 모델."""

    urls: list[str] = Field(
        ..., min_length=1, max_length=20, description="크롤링할 URL 목록 (최대 20개)"
    )
    ingest_multi_vector: bool = Field(
        default=False,
        description="True일 경우 크롤링 후 MultiVector 적재도 함께 수행 (ADR-078)",
    )


class CrawlResult(BaseModel):
    """크롤링 결과 모델."""

    total_urls: int
    success_count: int
    documents: list[dict]
    multi_vector_ingested: int = 0


@router.post("/crawl/trend", response_model=CrawlResult)
async def crawl_trend_data(request: CrawlRequest):
    """채용 트렌드 URL 크롤링 + trend_data 컬렉션 적재.

    - URL에서 텍스트 추출 (WebBaseLoader)
    - LLM으로 메타데이터 자동 태깅 (year, company, position, category, recruitment_type)
    - trend_data 컬렉션에 저장
    - ingest_multi_vector=True 시 요약/가설 쿼리 임베딩도 생성 (ADR-078)
    """
    from app.services.vectordb_service import VectorDBService
    from app.services.web_loader_service import TrendCrawlService

    settings = get_settings()

    try:
        # VectorDB 초기화
        vectordb = VectorDBService(
            api_key=settings.google_api_key,
            persist_directory=settings.chroma_persist_dir,
            chroma_server_host=settings.chroma_server_host,
            chroma_server_port=settings.chroma_server_port,
        )

        # LLM 초기화 (메타데이터 추출용)
        llm = None
        try:
            from app.domain.chat.chains import RAGChain

            rag_chain = RAGChain(settings)
            if hasattr(rag_chain, "llm"):
                llm = rag_chain.llm
        except Exception:
            logger.warning("[CrawlAPI] LLM 초기화 실패, 메타데이터 없이 진행")

        # 크롤링 실행
        crawl_service = TrendCrawlService(vectordb=vectordb, llm=llm)
        crawl_results = await crawl_service.crawl_trend_urls(request.urls)

        # ADR-078: MultiVector 적재
        multi_vector_count = 0
        if request.ingest_multi_vector and settings.rag_use_multi_vector and crawl_results:
            try:
                from app.services.rag_service import RAGService

                rag = RAGService(vectordb_service=vectordb, settings=settings)
                multi_vector_count = await rag.ingest_trend_documents(crawl_results)
            except Exception as e:
                logger.warning("[CrawlAPI] MultiVector 적재 실패: %s", e)

        # SAST: request 기반 값은 로그에 넣지 않음
        logger.info(
            "[CrawlAPI] 크롤링 완료: %d/%d URL 성공, MultiVector %d건",
            len(crawl_results),
            len(request.urls),
            multi_vector_count,
        )

        return CrawlResult(
            total_urls=len(request.urls),
            success_count=len(crawl_results),
            documents=crawl_results,
            multi_vector_ingested=multi_vector_count,
        )

    except Exception as e:
        logger.error("[CrawlAPI] 크롤링 API 오류: %s", e)
        raise HTTPException(status_code=500, detail="크롤링 처리 중 오류가 발생했습니다.") from e


class TriggerTaskResponse(BaseModel):
    """Celery 태스크 트리거 응답."""

    task_id: str
    message: str


class TaskStatusResponse(BaseModel):
    """Celery 태스크 상태 응답."""

    task_id: str
    status: str
    result: dict | None = None


@router.post("/crawl/trigger", response_model=TriggerTaskResponse)
async def trigger_trend_crawl():
    """Celery 태스크를 수동으로 트리거하여 트렌드 크롤링 실행.

    환경변수 TREND_CRAWL_URLS에 설정된 URL 목록을 크롤링합니다.
    """
    from app.services.trend_crawler_service import TrendCrawlerService

    try:
        service = TrendCrawlerService()
        task_id = service.trigger_celery_task()

        return TriggerTaskResponse(
            task_id=task_id,
            message="트렌드 크롤링 태스크가 시작되었습니다.",
        )

    except Exception as e:
        logger.error("[CrawlAPI] 태스크 트리거 실패: %s", e)
        raise HTTPException(status_code=500, detail="태스크 트리거에 실패했습니다.") from e


@router.get("/crawl/task/{task_id}", response_model=TaskStatusResponse)
async def get_crawl_task_status(task_id: str):
    """Celery 태스크 상태 조회.

    Args:
        task_id: 태스크 ID
    """
    from app.services.trend_crawler_service import TrendCrawlerService

    try:
        service = TrendCrawlerService()
        status = service.get_celery_task_status(task_id)

        return TaskStatusResponse(**status)

    except Exception as e:
        logger.error("[CrawlAPI] 태스크 상태 조회 실패: %s", e)
        raise HTTPException(status_code=500, detail="태스크 상태 조회에 실패했습니다.") from e


@router.get("/crawl/urls")
async def get_configured_urls():
    """환경변수에 설정된 크롤링 URL 목록 조회."""
    from app.services.trend_crawler_service import TrendCrawlerService

    service = TrendCrawlerService()
    urls = service.get_configured_urls()

    return {
        "total": len(urls),
        "urls": urls,
    }
