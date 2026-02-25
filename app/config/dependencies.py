"""
FastAPI Dependency Injection Configuration.

Provides dependency injection for services and infrastructure components.
"""

from functools import lru_cache
from typing import TYPE_CHECKING

from fastapi import Depends

from .settings import Settings, get_settings

if TYPE_CHECKING:
    from app.infrastructure.llm.base import BaseLLMProvider
    from app.infrastructure.ocr.base import BaseOCRProvider
    from app.infrastructure.queue.base import BaseTaskQueue
    from app.infrastructure.session.base import BaseSessionStore
    from app.infrastructure.vectordb.base import BaseVectorStore


# ============================================
# Infrastructure Dependencies
# ============================================


@lru_cache
def get_llm_provider(
    settings: Settings = Depends(get_settings),
) -> "BaseLLMProvider":
    """Get LLM provider instance.

    Returns Gemini provider by default.
    In production, this can be swapped for different providers.
    """
    from app.infrastructure.llm.gemini import GeminiProvider

    return GeminiProvider(
        api_key=settings.google_api_key,
        model_name=settings.gemini_model,
    )


@lru_cache
def get_vllm_provider(
    settings: Settings = Depends(get_settings),
) -> "BaseLLMProvider | None":
    """Get vLLM provider instance (optional).

    Returns None if vLLM is not configured.
    """
    if not settings.vllm_available:
        return None

    from app.infrastructure.llm.vllm import VLLMProvider

    return VLLMProvider(
        base_url=settings.gcp_vllm_base_url,
        model_name=settings.vllm_model_name,
    )


@lru_cache
def get_vectordb(
    settings: Settings = Depends(get_settings),
) -> "BaseVectorStore":
    """Get VectorDB instance.

    Returns ChromaDB by default.
    """
    from app.infrastructure.vectordb.chroma import ChromaVectorStore

    return ChromaVectorStore(
        persist_directory=settings.chroma_persist_dir,
        api_key=settings.google_api_key,
        embedding_model=settings.gemini_embedding_model,
    )


@lru_cache
def get_ocr_provider(
    settings: Settings = Depends(get_settings),
) -> "BaseOCRProvider":
    """Get OCR provider instance.

    Returns CLOVA OCR if configured, otherwise Gemini Vision.
    """
    if settings.clova_available:
        from app.infrastructure.ocr.clova import ClovaOCRProvider

        return ClovaOCRProvider(
            api_url=settings.clova_ocr_api_url,
            secret_key=settings.clova_ocr_secret_key,
        )

    from app.infrastructure.ocr.gemini_vision import GeminiVisionOCRProvider

    return GeminiVisionOCRProvider(
        api_key=settings.google_api_key,
        model_name=settings.gemini_model,
    )


# 세션 스토어 싱글톤 (InMemory 시 동일 인스턴스 공유)
_session_store_instance: "BaseSessionStore | None" = None


def get_session_store(
    settings: Settings = Depends(get_settings),
) -> "BaseSessionStore":
    """Get session store instance.

    Returns Redis in production, in-memory otherwise.
    In-memory store is a singleton so interview sessions are shared across requests.
    """
    global _session_store_instance
    if _session_store_instance is not None:
        return _session_store_instance

    if settings.is_production:
        from app.infrastructure.session.redis import RedisSessionStore

        _session_store_instance = RedisSessionStore(
            redis_url=settings.redis_url,
            default_ttl=settings.redis_session_ttl,
        )
    else:
        from app.infrastructure.session.memory import InMemorySessionStore

        _session_store_instance = InMemorySessionStore(
            default_ttl=settings.redis_session_ttl,
        )
    return _session_store_instance


def get_task_queue(
    settings: Settings = Depends(get_settings),
) -> "BaseTaskQueue":
    """Get task queue instance.

    Returns Celery in production, file-based otherwise.
    """
    if settings.is_production:
        from app.infrastructure.queue.celery_queue import CeleryTaskQueue

        return CeleryTaskQueue(
            broker_url=settings.celery_broker_url,
            backend_url=settings.celery_result_backend,
        )

    from app.infrastructure.queue.file_queue import FileTaskQueue

    return FileTaskQueue(storage_dir=settings.task_storage_dir)


# ============================================
# Legacy Task Storage (save/get dict - ai.py 호환)
# ============================================

_task_storage_instance = None


def get_legacy_task_storage(
    settings: Settings = Depends(get_settings),
):
    """Get legacy file-based task storage (save/get dict).

    Used by ai.py for text_extract and task status polling.
    Same API as utils.task_store.FileTaskStore for drop-in replacement.
    """
    from app.utils.task_store import FileTaskStore

    global _task_storage_instance
    if _task_storage_instance is None:
        _task_storage_instance = FileTaskStore(storage_dir=settings.task_storage_dir)
    return _task_storage_instance


# ============================================
# Domain Service Dependencies
# ============================================


def get_chat_service(
    vectordb: "BaseVectorStore" = Depends(get_vectordb),
    llm_provider: "BaseLLMProvider" = Depends(get_llm_provider),
    vllm_provider: "BaseLLMProvider | None" = Depends(get_vllm_provider),
):
    """Get Chat service instance."""
    from app.domain.chat.services import ChatService

    return ChatService(
        vectordb=vectordb,
        llm_provider=llm_provider,
        vllm_provider=vllm_provider,
    )


def get_interview_service(
    session_store: "BaseSessionStore" = Depends(get_session_store),
    llm_provider: "BaseLLMProvider" = Depends(get_llm_provider),
    vectordb: "BaseVectorStore" = Depends(get_vectordb),
    settings: Settings = Depends(get_settings),
):
    """Get Interview service instance."""
    from app.domain.interview.services import InterviewService

    return InterviewService(
        session_store=session_store,
        llm_provider=llm_provider,
        vectordb=vectordb,
        max_questions=settings.interview_max_questions,
        max_followup_depth=settings.interview_max_followup_depth,
    )


def get_ocr_service(
    ocr_provider: "BaseOCRProvider" = Depends(get_ocr_provider),
    llm_provider: "BaseLLMProvider" = Depends(get_llm_provider),
):
    """Get OCR service instance."""
    from app.domain.ocr.services import OCRService

    return OCRService(
        ocr_provider=ocr_provider,
        fallback_provider=llm_provider,  # Gemini Vision as fallback
    )


def get_masking_service(
    settings: Settings = Depends(get_settings),
):
    """Get Masking service instance."""
    from app.domain.masking.services import MaskingService

    return MaskingService(
        google_api_key=settings.google_api_key,
    )


def get_openai_provider(
    settings: Settings = Depends(get_settings),
) -> "BaseLLMProvider | None":
    """Get OpenAI provider instance (optional).

    Returns None if OpenAI is not configured.
    Used for evaluation debate feature.
    """
    if not settings.openai_available:
        return None

    from app.infrastructure.llm.openai_provider import OpenAIProvider

    return OpenAIProvider(
        api_key=settings.openai_api_key,
        model_name=settings.eval_gpt_model,
    )


def get_evaluation_analyzer(
    settings: Settings = Depends(get_settings),
):
    """Get Evaluation Analyzer instance (Gemini 3 Pro)."""
    from app.domain.evaluation.analyzer import InterviewAnalyzer

    return InterviewAnalyzer(
        api_key=settings.google_api_key,
        model_name=settings.eval_gemini_model,
        thinking_level=settings.eval_thinking_level,
    )


def get_debate_service(
    settings: Settings = Depends(get_settings),
):
    """Get Debate service instance (LangGraph).

    Returns None if debate is not available.
    """
    if not settings.debate_available:
        return None

    from app.domain.evaluation.debate_graph import DebateService

    return DebateService(
        google_api_key=settings.google_api_key,
        openai_api_key=settings.openai_api_key,
        gemini_model=settings.eval_gemini_model,
        gpt_model=settings.eval_gpt_model,
        thinking_level=settings.eval_thinking_level,
    )


# ============================================
# Utility Dependencies
# ============================================


def get_current_settings() -> Settings:
    """Get current settings (non-cached for testing)."""
    return get_settings()
