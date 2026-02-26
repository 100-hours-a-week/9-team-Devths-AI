"""
Web Loader Service — URL에서 텍스트 추출 (LangChain WebBaseLoader)

용도:
1. 첫 화면: 이력서/채용공고 URL 입력 → 텍스트 추출 → VectorDB 저장
2. 채팅 중: 메시지 내 URL 감지 → 웹 내용 읽기 → 컨텍스트 주입

ADR-059 참조
"""

import logging
import re
from asyncio import get_event_loop

from langchain_community.document_loaders import WebBaseLoader

logger = logging.getLogger(__name__)

# URL 감지 정규식
URL_PATTERN = re.compile(r"https?://[^\s<>\"')\]]+")

# 텍스트 최대 길이 (채팅 컨텍스트용 — 약 750 토큰)
MAX_CHAT_CONTEXT_LENGTH = 3000

# 텍스트 최대 길이 (문서 추출용 — 약 25,000 토큰)
MAX_DOCUMENT_LENGTH = 100000


class WebLoaderService:
    """URL에서 텍스트를 추출하는 서비스"""

    @staticmethod
    async def extract_text_from_url(url: str, max_length: int = MAX_DOCUMENT_LENGTH) -> str:
        """URL에서 텍스트를 추출합니다.

        Args:
            url: 웹 페이지 URL
            max_length: 추출 텍스트 최대 길이

        Returns:
            추출된 텍스트 (실패 시 빈 문자열)
        """
        try:
            # SAST: url은 사용자 입력값이므로 로그에 넣지 않음
            logger.info("[WebLoader] URL 텍스트 추출 시작")
            loop = get_event_loop()
            docs = await loop.run_in_executor(None, _load_url, url)

            if not docs:
                logger.warning("[WebLoader] URL에서 문서를 추출할 수 없습니다")
                return ""

            text = "\n".join(doc.page_content for doc in docs)
            text = text.strip()

            if len(text) > max_length:
                text = text[:max_length]
                logger.info("[WebLoader] 텍스트 길이 제한 적용: %d자 → %d자", len(text), max_length)

            logger.info("[WebLoader] 추출 완료: %d자", len(text))
            return text

        except Exception as e:
            logger.error("[WebLoader] URL 텍스트 추출 실패: %s", e)
            return ""

    @staticmethod
    def detect_urls(text: str) -> list[str]:
        """텍스트에서 URL을 감지합니다.

        Args:
            text: 사용자 메시지 텍스트

        Returns:
            감지된 URL 리스트
        """
        return URL_PATTERN.findall(text)

    @staticmethod
    async def extract_chat_context(user_message: str) -> tuple[str, str | None]:
        """채팅 메시지에서 URL을 감지하고, 웹 내용을 컨텍스트로 추출합니다.

        Args:
            user_message: 사용자 메시지

        Returns:
            (원본 메시지, 웹 컨텍스트 텍스트 또는 None)
        """
        urls = WebLoaderService.detect_urls(user_message)
        if not urls:
            return user_message, None

        url = urls[0]
        # SAST: url은 사용자 입력값이므로 로그에 직접 넣지 않음
        logger.info("[WebLoader] 채팅 메시지에서 URL 감지됨 (1건)")

        web_text = await WebLoaderService.extract_text_from_url(
            url, max_length=MAX_CHAT_CONTEXT_LENGTH
        )
        if not web_text:
            return user_message, None

        return user_message, web_text


def _load_url(url: str):
    """WebBaseLoader를 동기적으로 호출합니다 (run_in_executor용)."""
    loader = WebBaseLoader(url)
    return loader.load()


# ADR-094/078: 메타데이터 추출용 프롬프트
_METADATA_EXTRACTION_TEMPLATE = """다음 채용/면접 관련 문서에서 메타데이터를 추출하세요.
반드시 아래 JSON 형식으로만 응답하세요. 확인할 수 없는 필드는 null로 적으세요.

{{
  "year": 연도(정수) 또는 null,
  "company": "기업명" 또는 null,
  "position": "직무/포지션" 또는 null,
  "category": "기술질문|인성질문|채용공고|면접후기|채용트렌드" 중 하나 또는 null,
  "recruitment_type": "공채|수시|인턴" 중 하나 또는 null
}}

문서 내용:
{doc}"""


class TrendCrawlService:
    """ADR-094/078: 채용 트렌드 URL 크롤링 + 메타데이터 자동 태깅 서비스."""

    def __init__(self, vectordb=None, llm=None):
        self.vectordb = vectordb
        self.llm = llm

    async def crawl_trend_urls(
        self,
        urls: list[str],
    ) -> list[dict]:
        """URL 목록을 크롤링하여 메타데이터 태깅 후 trend_data 컬렉션에 적재.

        Args:
            urls: 크롤링할 URL 목록

        Returns:
            적재된 문서 정보 리스트
        """
        import uuid

        results = []

        for url in urls:
            try:
                # SAST: url은 사용자 입력값이므로 로그에 넣지 않음
                logger.info("[TrendCrawl] URL 크롤링 시작 (%d/%d)", len(results) + 1, len(urls))
                text = await WebLoaderService.extract_text_from_url(url)

                if not text:
                    logger.warning("[TrendCrawl] URL에서 텍스트 추출 실패")
                    continue

                # LLM으로 메타데이터 자동 추출
                metadata = await self._extract_metadata_with_llm(text)
                metadata["source"] = "webbaseloader"

                # trend_data 컬렉션에 적재
                doc_id = str(uuid.uuid4())
                if self.vectordb:
                    await self.vectordb.add_document(
                        document_id=doc_id,
                        text=text,
                        collection_type="trend_data",
                        metadata=metadata,
                    )

                results.append(
                    {
                        "id": doc_id,
                        "text_length": len(text),
                        "metadata": metadata,
                    }
                )
                logger.info(
                    "[TrendCrawl] 문서 적재 완료: %d자, 메타데이터 %d개 필드",
                    len(text),
                    len([v for v in metadata.values() if v is not None]),
                )

            except Exception as e:
                logger.error("[TrendCrawl] 크롤링 실패: %s", e)
                continue

        logger.info("[TrendCrawl] 총 %d/%d URL 적재 완료", len(results), len(urls))
        return results

    async def _extract_metadata_with_llm(self, text: str) -> dict:
        """LLM으로 문서 내용에서 메타데이터 자동 추출.

        Returns:
            추출된 메타데이터 dict (null 값은 제거)
        """
        import json as _json

        if not self.llm:
            logger.warning("[TrendCrawl] LLM 미설정, 기본 메타데이터 반환")
            return {}

        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            # 문서 앞부분만 사용 (토큰 절약)
            snippet = text[:3000]
            prompt = ChatPromptTemplate.from_template(_METADATA_EXTRACTION_TEMPLATE)
            chain = prompt | self.llm | StrOutputParser()

            result_text = await chain.ainvoke({"doc": snippet})

            # JSON 파싱
            result_text = result_text.strip()
            if result_text.startswith("```"):
                result_text = result_text.split("\n", 1)[-1].rsplit("```", 1)[0]

            metadata = _json.loads(result_text)
            # None 값 제거 (ChromaDB 호환)
            metadata = {k: v for k, v in metadata.items() if v is not None}
            return metadata

        except Exception as e:
            logger.warning("[TrendCrawl] 메타데이터 추출 실패 (기본값 사용): %s", e)
            return {}
