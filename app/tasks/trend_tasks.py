"""
Trend Crawling Tasks — ADR-094.

Celery 태스크: 채용 트렌드 URL 크롤링 + VectorDB 적재.
"""

import asyncio
import logging
import os

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# 환경변수에서 크롤링 URL 목록 로드 (쉼표 구분)
TREND_CRAWL_URLS = os.getenv("TREND_CRAWL_URLS", "")


def _get_trend_urls() -> list[str]:
    """환경변수에서 트렌드 크롤링 URL 목록을 파싱."""
    if not TREND_CRAWL_URLS:
        return []
    return [url.strip() for url in TREND_CRAWL_URLS.split(",") if url.strip()]


@celery_app.task(bind=True, name="app.tasks.trend_tasks.crawl_trend_urls_task")
def crawl_trend_urls_task(self):
    """채용 트렌드 URL 크롤링 태스크 (Celery Beat 스케줄러용).

    환경변수 TREND_CRAWL_URLS에 설정된 URL 목록을 크롤링하여
    trend_data 컬렉션에 적재합니다.
    """
    urls = _get_trend_urls()
    if not urls:
        logger.warning("[TrendTask] TREND_CRAWL_URLS 환경변수가 비어있습니다.")
        return {"status": "skipped", "reason": "no_urls_configured"}

    logger.info("[TrendTask] 트렌드 크롤링 시작: %d개 URL", len(urls))

    try:
        result = asyncio.run(_crawl_urls_async(urls))
        logger.info(
            "[TrendTask] 트렌드 크롤링 완료: %d/%d 성공",
            result.get("success_count", 0),
            len(urls),
        )
        return result

    except Exception as e:
        logger.error("[TrendTask] 트렌드 크롤링 실패: %s", e)
        raise self.retry(exc=e, countdown=300, max_retries=3) from e


async def _crawl_urls_async(urls: list[str]) -> dict:
    """비동기 크롤링 실행."""
    from app.config.settings import get_settings
    from app.services.vectordb_service import VectorDBService
    from app.services.web_loader_service import TrendCrawlService

    settings = get_settings()

    vectordb = VectorDBService(
        api_key=settings.google_api_key,
        persist_directory=settings.chroma_persist_dir,
        chroma_server_host=settings.chroma_server_host,
        chroma_server_port=settings.chroma_server_port,
    )

    llm = None
    try:
        from app.domain.chat.chains import RAGChain

        rag_chain = RAGChain(settings)
        if hasattr(rag_chain, "llm"):
            llm = rag_chain.llm
    except Exception:
        logger.warning("[TrendTask] LLM 초기화 실패, 메타데이터 없이 진행")

    crawl_service = TrendCrawlService(vectordb=vectordb, llm=llm)
    results = await crawl_service.crawl_trend_urls(urls)

    return {
        "status": "completed",
        "total_urls": len(urls),
        "success_count": len(results),
        "documents": [{"id": r["id"], "text_length": r["text_length"]} for r in results],
    }
