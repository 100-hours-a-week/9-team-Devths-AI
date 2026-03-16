"""
Tavily Search Service — ADR-101, ADR-131.

Phase 2 채용 트렌드 검색: 검색 쿼리 기반 자동 URL 발견 + 본문 추출.

Phase 1 (WebBaseLoader) 대비 장점:
- 검색 쿼리로 관련 URL 자동 발견 (URL 수동 관리 불필요)
- JS 렌더링 지원 (채용 사이트 SPA 대응)
- 메타데이터(제목, URL, score) 자동 제공 → LLM 메타데이터 추출 비용 절감
- 중복 URL 자동 필터링

ADR-131 추가 신뢰성 검증:
- days 파라미터로 API 레벨 날짜 필터 (최신 데이터만 수신)
- score 임계값 미달 결과 제거 (저품질 문서 차단)
- 최소 본문 길이 미달 결과 제거 (스텁 페이지 차단)
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
    filtered_by_score: int = 0         # ADR-131: score 임계값 미달로 제거된 수
    filtered_by_length: int = 0        # ADR-131: 최소 본문 길이 미달로 제거된 수
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

    async def _search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> tuple[list[TavilySearchResult], int, int]:
        """단일 쿼리 검색 내부 구현 — 필터 통계 포함 반환 (ADR-101, ADR-131).

        Returns:
            (results, filtered_by_score, filtered_by_length)

        Filters (ADR-131):
            1. API 레벨: days 파라미터 → 오래된 문서 수신 차단
            2. 빈 콘텐츠 제거
            3. 최소 본문 길이(tavily_min_content_length) 미달 제거
            4. 최소 score(tavily_min_score) 미달 제거
        """
        from langchain_tavily import TavilySearch

        n = max_results or self.settings.tavily_max_results

        try:
            # ADR-131: API 레벨 날짜 필터 — 오래된 문서를 수신 전 차단 (가장 효율적)
            tool_kwargs: dict = {
                "api_key": self.settings.tavily_api_key,
                "max_results": n,
                "search_depth": self.settings.tavily_search_depth,
                "include_raw_content": True,
            }
            if self.settings.tavily_days > 0:
                tool_kwargs["days"] = self.settings.tavily_days

            tool = TavilySearch(**tool_kwargs)

            # SAST: 쿼리는 사용자 입력이므로 로그에 직접 포함하지 않음
            logger.info(
                "[Tavily] 검색 실행 (depth=%s, max=%d, days=%s, min_score=%.2f)",
                self.settings.tavily_search_depth,
                n,
                self.settings.tavily_days or "무제한",
                self.settings.tavily_min_score,
            )

            raw = await tool.ainvoke({"query": query})
            items = raw if isinstance(raw, list) else raw.get("results", [])

            results = []
            filtered_score = 0
            filtered_length = 0

            for r in items:
                content = r.get("raw_content") or r.get("content", "")
                score = float(r.get("score", 0.0))

                # 필터 1: 빈 콘텐츠
                if not content:
                    logger.debug("[Tavily] 빈 콘텐츠 제거: %s", r.get("url", ""))
                    continue

                # 필터 2: 최소 본문 길이 (ADR-131)
                if (
                    self.settings.tavily_min_content_length > 0
                    and len(content) < self.settings.tavily_min_content_length
                ):
                    logger.debug(
                        "[Tavily] 짧은 콘텐츠 제거 (%d자 < %d자): %s",
                        len(content),
                        self.settings.tavily_min_content_length,
                        r.get("url", ""),
                    )
                    filtered_length += 1
                    continue

                # 필터 3: 최소 relevance score (ADR-131)
                if (
                    self.settings.tavily_min_score > 0.0
                    and score < self.settings.tavily_min_score
                ):
                    logger.debug(
                        "[Tavily] 낮은 score 제거 (%.2f < %.2f): %s",
                        score,
                        self.settings.tavily_min_score,
                        r.get("url", ""),
                    )
                    filtered_score += 1
                    continue

                results.append(
                    TavilySearchResult(
                        title=r.get("title", ""),
                        url=r.get("url", ""),
                        content=content,
                        score=score,
                        published_date=r.get("published_date"),
                        query=query,
                    )
                )

            logger.info(
                "[Tavily] 검색 완료: %d건 반환 (원본 %d건, score 필터 %d건, 길이 필터 %d건)",
                len(results),
                len(items),
                filtered_score,
                filtered_length,
            )
            return results, filtered_score, filtered_length

        except Exception as e:
            logger.error("[Tavily] 검색 실패: %s", e)
            raise

    async def search(
        self,
        query: str,
        max_results: int | None = None,
    ) -> list[TavilySearchResult]:
        """단일 쿼리로 관련 웹 페이지 검색 + 본문 추출 (ADR-101, ADR-131).

        Args:
            query: 검색 쿼리 (예: "2026 백엔드 채용 트렌드")
            max_results: 최대 결과 수 (기본값: settings.tavily_max_results)

        Returns:
            TavilySearchResult 리스트 (신뢰성 필터 통과한 결과만)
        """
        results, _, _ = await self._search(query, max_results)
        return results

    async def search_multiple(
        self,
        queries: list[str],
    ) -> tuple[list[TavilySearchResult], TavilySearchStats]:
        """복수 쿼리 병렬 검색 + 중복 URL 제거.

        Args:
            queries: 검색 쿼리 목록

        Returns:
            (중복 제거된 결과 리스트, 검색 통계 — filtered_by_score/length 포함)
        """
        logger.info("[Tavily] 병렬 검색 시작: %d개 쿼리", len(queries))

        tasks = [self._search(q) for q in queries]
        results_list = await asyncio.gather(*tasks, return_exceptions=True)

        seen_urls: set[str] = set()
        merged: list[TavilySearchResult] = []
        failed_queries: list[str] = []
        total_raw = 0
        total_filtered_score = 0
        total_filtered_length = 0

        for query, result in zip(queries, results_list, strict=False):
            if isinstance(result, Exception):
                logger.warning("[Tavily] 쿼리 실패 (계속 진행): %s", result)
                failed_queries.append(query)
                continue

            results, filtered_score, filtered_length = result
            total_raw += len(results)
            total_filtered_score += filtered_score
            total_filtered_length += filtered_length
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    merged.append(r)

        stats = TavilySearchStats(
            total_queries=len(queries),
            total_results=total_raw,
            deduplicated_results=len(merged),
            filtered_by_score=total_filtered_score,
            filtered_by_length=total_filtered_length,
            failed_queries=failed_queries,
        )

        logger.info(
            "[Tavily] 병렬 검색 완료: 총 %d건 → 중복 제거 후 %d건"
            " (실패 %d개, score 필터 %d건, 길이 필터 %d건, days=%s)",
            total_raw,
            len(merged),
            len(failed_queries),
            total_filtered_score,
            total_filtered_length,
            self.settings.tavily_days or "무제한",
        )
        return merged, stats
