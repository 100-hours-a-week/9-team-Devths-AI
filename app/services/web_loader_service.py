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
            logger.info("[WebLoader] URL 텍스트 추출 시작: %s", url[:80])
            loop = get_event_loop()
            docs = await loop.run_in_executor(None, _load_url, url)

            if not docs:
                logger.warning("[WebLoader] URL에서 문서를 추출할 수 없습니다: %s", url[:80])
                return ""

            text = "\n".join(doc.page_content for doc in docs)
            text = text.strip()

            if len(text) > max_length:
                text = text[:max_length]
                logger.info("[WebLoader] 텍스트 길이 제한 적용: %d자 → %d자", len(text), max_length)

            logger.info("[WebLoader] 추출 완료: %d자", len(text))
            return text

        except Exception as e:
            logger.error("[WebLoader] URL 텍스트 추출 실패 (%s): %s", url[:80], e)
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
        logger.info("[WebLoader] 채팅 메시지에서 URL 감지: %s", url[:80])

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
