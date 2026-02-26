"""
Tavily Search Service — ADR-101.

Phase 2 채용 트렌드 검색: 검색 쿼리 기반 자동 URL 발견 + 본문 추출.

Phase 1 (WebBaseLoader) 대비 장점:
- 검색 쿼리로 관련 URL 자동 발견 (URL 수동 관리 불필요)
- JS 렌더링 지원 (채용 사이트 SPA 대응)
- 메타데이터(제목, URL, score) 자동 제공 → LLM 메타데이터 추출 비용 절감
- 중복 URL 자동 필터링
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass
class TavilySearchResult:
    """Tavily 검색 결과 단위."""

    title: str
    url: str
    content: str  # Tavily가 추출한 본문 (raw_content 우선, content 폴백)
    score: float  # relevance score (0.0 ~ 1.0)
    published_date: str | None = None
    query: str = ""  # 어느 쿼리에서 발견됐는지


@dataclass
class TavilySearchStats:
    """Tavily 검색 통계."""

    total_queries: int
    total_results: int
    deduplicated_results: int
    failed_queries: list[str] = field(default_factory=list)


class TavilySearchService:
    """ADR-101: Tavily API 기반 검색 + 본문 추출 서비스.

    - `search(query)` — 단일 쿼리 검색
    - `search_multiple(queries)` — 복수 쿼리 병렬 검색 + 중복 URL 제거

    TAVILY_API_KEY 환경변수 필수.
    미설정 시 ValueError 발생 → /crawl/trend/search에서 503으로 처리.
    """

    def __init__(self, settings: Settings | None = None):
        from app.config.settings import get_settings

        self.settings = settings or get_settings()
        if not self.settings.tavily_api_key:
            raise ValueError(
                "TAVILY_API_KEY 환경변수가 설정되지 않았습니다. "
                "Tavily 홈페이지(https://tavily.com)에서 API 키를 발급받아 설정하세요."
            )

    async def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[TavilySearchResult]:
        """단일 쿼리로 관련 웹 페이지 검색 + 본문 추출.

        Args:
            query: 검색 쿼리 (예: "2026 백엔드 채용 트렌드")
            max_results: 최대 결과 수 (기본값: settings.tavily_max_results)

        Returns:
            TavilySearchResult 리스트 (content 없는 결과 제외)
        """
        from langchain_tavily import TavilySearch

        n = max_results or self.settings.tavily_max_results

        try:
            tool = TavilySearch(
                api_key=self.settings.tavily_api_key,
                max_results=n,
                search_depth=self.settings.tavily_search_depth,
                include_raw_content=True,
            )

            # SAST: 쿼리는 사용자 입력이므로 로그에 직접 포함하지 않음
            logger.info(
                "[Tavily] 검색 실행 (depth=%s, max=%d)", self.settings.tavily_search_depth, n
            )

            raw = await tool.ainvoke({"query": query})
            items = raw if isinstance(raw, list) else raw.get("results", [])

            results = []
            for r in items:
                content = r.get("raw_content") or r.get("content", "")
                if not content:
                    continue
                results.append(
                    TavilySearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        content=content,
                        score=float(r.get("score", 0.0)),
                        published_date=r.get("published_date"),
                        query=query,
                    )
                )

            logger.info("[Tavily] 검색 완료: %d건 반환", len(results))
            return results

        except Exception as e:
            logger.error("[Tavily] 검색 실패: %s", e)
            raise

    async def search_multiple(
        self,
        queries: list[str],
    ) -> tuple[list[TavilySearchResult], TavilySearchStats]:
        """복수 쿼리 병렬 검색 + 중복 URL 제거.

        Args:
            queries: 검색 쿼리 목록

        Returns:
            (중복 제거된 결과 리스트, 검색 통계)
        """
        logger.info("[Tavily] 병렬 검색 시작: %d개 쿼리", len(queries))

        tasks = [self.search(q) for q in queries]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        seen_urls: set[str] = set()
        merged: list[TavilySearchResult] = []
        failed_queries: list[str] = []
        total_raw = 0

        for query, results in zip(queries, results_list, strict=False):
            if isinstance(results, Exception):
                logger.warning("[Tavily] 쿼리 실패 (계속 진행): %s", results)
                failed_queries.append(query)
                continue

            total_raw += len(results)
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    merged.append(r)

        stats = TavilySearchStats(
            total_queries=len(queries),
            total_results=total_raw,
            deduplicated_results=len(merged),
            failed_queries=failed_queries,
        )

        logger.info(
            "[Tavily] 병렬 검색 완료: 총 %d건 → 중복 제거 후 %d건 (실패 쿼리 %d개)",
            total_raw,
            len(merged),
            len(failed_queries),
        )
        return merged, stats
