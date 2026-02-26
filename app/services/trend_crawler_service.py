"""
Trend Crawler Service — ADR-094.

채용 트렌드 크롤링 서비스: Celery Beat 스케줄러 연동 및 수동 크롤링 지원.
"""

import logging
from dataclasses import dataclass

from app.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass
class CrawlStats:
    """크롤링 통계."""

    total_urls: int
    success_count: int
    failed_count: int
    documents: list[dict]


class TrendCrawlerService:
    """ADR-094: 채용 트렌드 크롤링 서비스.

    - Celery Beat 스케줄러와 연동하여 주기적 크롤링
    - 수동 크롤링 API 지원
    - VectorDB trend_data 컬렉션에 적재
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()

    def get_configured_urls(self) -> list[str]:
        """환경변수에서 설정된 크롤링 URL 목록 반환."""
        urls_str = self.settings.trend_crawl_urls
        if not urls_str:
            return []
        return [url.strip() for url in urls_str.split(",") if url.strip()]

    async def crawl_configured_urls(self) -> CrawlStats:
        """환경변수에 설정된 URL 목록을 크롤링.

        Returns:
            CrawlStats: 크롤링 통계
        """
        urls = self.get_configured_urls()
        if not urls:
            logger.warning("[TrendCrawler] 설정된 URL이 없습니다.")
            return CrawlStats(
                total_urls=0,
                success_count=0,
                failed_count=0,
                documents=[],
            )

        return await self.crawl_urls(urls)

    async def crawl_urls(self, urls: list[str]) -> CrawlStats:
        """지정된 URL 목록을 크롤링.

        Args:
            urls: 크롤링할 URL 목록

        Returns:
            CrawlStats: 크롤링 통계
        """
        from app.services.vectordb_service import VectorDBService
        from app.services.web_loader_service import TrendCrawlService

        logger.info("[TrendCrawler] 크롤링 시작: %d개 URL", len(urls))

        vectordb = VectorDBService(
            api_key=self.settings.google_api_key,
            persist_directory=self.settings.chroma_persist_dir,
            chroma_server_host=self.settings.chroma_server_host,
            chroma_server_port=self.settings.chroma_server_port,
        )

        llm = self._get_llm()

        crawl_service = TrendCrawlService(vectordb=vectordb, llm=llm)
        results = await crawl_service.crawl_trend_urls(urls)

        stats = CrawlStats(
            total_urls=len(urls),
            success_count=len(results),
            failed_count=len(urls) - len(results),
            documents=results,
        )

        logger.info(
            "[TrendCrawler] 크롤링 완료: %d/%d 성공",
            stats.success_count,
            stats.total_urls,
        )

        return stats

    def _get_llm(self):
        """LLM 인스턴스 반환 (메타데이터 추출용)."""
        try:
            from app.domain.chat.chains import RAGChain

            rag_chain = RAGChain(self.settings)
            if hasattr(rag_chain, "llm"):
                return rag_chain.llm
        except Exception as e:
            logger.warning("[TrendCrawler] LLM 초기화 실패: %s", e)
        return None

    def trigger_celery_task(self) -> str:
        """Celery 태스크를 수동으로 트리거.

        Returns:
            str: 태스크 ID
        """
        from app.tasks.trend_tasks import crawl_trend_urls_task

        task = crawl_trend_urls_task.delay()
        logger.info("[TrendCrawler] Celery 태스크 트리거됨: %s", task.id)
        return task.id

    def get_celery_task_status(self, task_id: str) -> dict:
        """Celery 태스크 상태 조회.

        Args:
            task_id: 태스크 ID

        Returns:
            dict: 태스크 상태 정보
        """
        from app.tasks.celery_app import celery_app

        result = celery_app.AsyncResult(task_id)
        return {
            "task_id": task_id,
            "status": result.status,
            "result": result.result if result.ready() else None,
        }
