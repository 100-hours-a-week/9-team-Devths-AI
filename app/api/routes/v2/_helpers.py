"""
v2 라우트 공유 헬퍼 함수 및 서비스 초기화
"""

import asyncio
import json
import logging
import os
import re
from collections.abc import AsyncGenerator

from fastapi import Header, HTTPException, status

from app.config.settings import get_settings
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.vectordb_service import VectorDBService
from app.services.vllm_service import VLLMService

logger = logging.getLogger(__name__)

# Initialize services
_llm_service = None
_vllm_service = None
_vectordb_service = None
_langchain_gateway = None
_rag_chain = None
_rag_service = None


def get_services():
    """Get or initialize AI services (설정은 config/settings 사용)"""
    global \
        _llm_service, \
        _vllm_service, \
        _vectordb_service, \
        _langchain_gateway, \
        _rag_chain, \
        _rag_service

    if _llm_service is None:
        settings = get_settings()
        raw_key = (
            settings.google_api_key or os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        )
        api_key = (raw_key.strip() if isinstance(raw_key, str) and raw_key else raw_key) or None
        if api_key == "":
            api_key = None
        _llm_service = LLMService(api_key=api_key)
        _vectordb_service = VectorDBService(
            api_key=api_key,
            chroma_server_host=settings.chroma_server_host,
            chroma_server_port=settings.chroma_server_port,
        )

        # LangChain LCEL Gateway (면접/QnA 체인용, API 키 분산·폴백 지원)
        try:
            raw_keys = settings.all_google_api_keys or ([api_key] if api_key else [])
            keys = [k.strip() for k in raw_keys if k and isinstance(k, str) and k.strip()]
            if keys:
                from app.infrastructure.llm.langchain_wrapper import LangChainLLMGateway

                _langchain_gateway = LangChainLLMGateway(
                    google_api_keys=keys if len(keys) > 1 else None,
                    google_api_key=keys[0] if len(keys) == 1 else None,
                    model_name=settings.gemini_model,
                    embedding_model=settings.gemini_embedding_model,
                )
                logger.info("✅ LangChain LCEL Gateway initialized")
            else:
                _langchain_gateway = None
        except Exception as e:
            logger.warning(f"LangChain Gateway initialization failed: {e}")
            _langchain_gateway = None

        # Initialize vLLM service (GCP GPU server)
        gcp_vllm_url = settings.gcp_vllm_base_url or os.getenv("GCP_VLLM_BASE_URL")

        try:
            if gcp_vllm_url:
                logger.info(f"🌐 GCP vLLM 서버 연결: {gcp_vllm_url}")
                _vllm_service = VLLMService()
                logger.info("✅ vLLM service initialized (GCP GPU server)")
            else:
                # GCP URL 없으면 OCR 전용 모드
                logger.info("💰 GCP URL 없음 - OCR 전용 모드로 초기화")
                _vllm_service = VLLMService(ocr_only=True)
                logger.info("✅ vLLM service initialized (OCR-only mode)")
        except Exception as e:
            logger.warning(f"vLLM service initialization failed: {e}")
            _vllm_service = None

        _rag_chain = None
        if _langchain_gateway is not None:
            try:
                from langchain_chroma import Chroma

                from app.domain.chat.chains import COLLECTION_NAME_MAP, RAGChain

                chroma_client = _vectordb_service.chroma_client
                embedding_function = _langchain_gateway.embeddings
                vectorstores = {}
                for collection_type, collection_name in COLLECTION_NAME_MAP.items():
                    vectorstores[collection_type] = Chroma(
                        client=chroma_client,
                        collection_name=collection_name,
                        embedding_function=embedding_function,
                    )
                _rag_chain = RAGChain(
                    llm_gateway=_langchain_gateway,
                    vectorstores=vectorstores,
                    max_context_length=settings.rag_max_context_length,
                    retrieval_k=settings.rag_retrieval_k,
                    fetch_k=settings.rag_fetch_k,
                    lambda_mult=settings.rag_lambda_mult,
                )
                logger.info("RAGChain (MMR) initialized for retrieve_context")
            except Exception as e:
                logger.warning("RAGChain initialization failed, using VectorDB fallback: %s", e)
                _rag_chain = None

        _rag_service = RAGService(
            _llm_service,
            _vectordb_service,
            _vllm_service,
            langchain_gateway=_langchain_gateway,
            rag_chain=_rag_chain,
        )

    return _rag_service


def get_session_key(user_id: int, interview_id: int | None) -> str:
    """면접 세션 캐시 키 생성"""
    return f"interview:{user_id}:{interview_id or 'default'}"


def format_analysis_text(
    resume_analysis: dict | None,
    posting_analysis: dict | None,
    summary: str | None,
) -> str:
    """분석 결과를 plain text로 포맷팅 (마크다운 없이)

    백엔드에서 바로 화면에 표시할 수 있도록 포맷팅된 텍스트를 생성합니다.
    """
    lines = []

    # 회사/직무
    if summary:
        lines.append(f"지원 회사 및 직무 : {summary}")
        lines.append("")

    # 이력서 분석
    if resume_analysis:
        lines.append("이력서 분석")
        lines.append("")
        lines.append("장점")
        strengths = resume_analysis.get("strengths", [])
        for i, strength in enumerate(strengths[:5], 1):
            lines.append(f"{i}. {strength}")
        lines.append("")
        lines.append("단점")
        weaknesses = resume_analysis.get("weaknesses", [])
        for i, weakness in enumerate(weaknesses[:5], 1):
            lines.append(f"{i}. {weakness}")
        lines.append("")

    # 채용공고 분석
    if posting_analysis:
        lines.append("채용 공고 분석")
        lines.append("")
        company = posting_analysis.get("company", "")
        position = posting_analysis.get("position", "")
        lines.append("기업 / 포지션")
        lines.append(f"{company} / {position}")
        lines.append("")
        lines.append("필수 역량")
        for skill in posting_analysis.get("required_skills", [])[:5]:
            lines.append(f"- {skill}")
        lines.append("")
        lines.append("우대 사항")
        for skill in posting_analysis.get("preferred_skills", [])[:5]:
            lines.append(f"- {skill}")
        lines.append("")

    # 매칭도
    if resume_analysis and posting_analysis:
        lines.append("매칭도")
        lines.append("")
        lines.append("나와 지원 직무에 맞는 점")
        matches = resume_analysis.get("matches", resume_analysis.get("strengths", [])[:3])
        for match in matches[:3] if matches else []:
            lines.append(f"- {match}")
        lines.append("")
        lines.append("나와 지원 직무에 맞지 않는 점")
        gaps = resume_analysis.get("gaps", resume_analysis.get("weaknesses", [])[:3])
        for gap in gaps[:3] if gaps else []:
            lines.append(f"- {gap}")

    return "\n".join(lines)


async def verify_api_key(x_api_key: str | None = Header(None)):
    """API 키 검증"""
    if x_api_key != "your-api-key-here":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return x_api_key


# ── SSE 스트리밍 상수 ──────────────────────────────────────────
SSE_CHAR_DELAY = 0.015  # 글자당 딜레이(초) — SSE 타이핑 효과


def extract_json_from_llm_response(text: str) -> dict | None:
    """LLM 응답 텍스트에서 JSON 객체를 추출한다.

    마크다운 코드 펜스(```json ... ```)를 제거한 뒤 첫 번째 {...} 블록을 파싱.
    파싱 실패 시 None 반환.
    """
    clean = re.sub(r"```json\s*", "", text)
    clean = re.sub(r"```\s*$", "", clean).strip()
    s = clean.find("{")
    e = clean.rfind("}") + 1
    if s != -1 and e > s:
        try:
            return json.loads(clean[s:e])
        except json.JSONDecodeError:
            return None
    return None


async def stream_text_chars(text: str, sse_end: str) -> AsyncGenerator[str, None]:
    """텍스트를 한 글자씩 SSE 청크로 스트리밍하는 비동기 제너레이터."""
    for char in text:
        yield f"data: {json.dumps({'chunk': char}, ensure_ascii=False)}{sse_end}"
        await asyncio.sleep(SSE_CHAR_DELAY)
