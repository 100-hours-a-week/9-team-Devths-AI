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
    """Get vLLM 8B provider instance (평시 질의응답).

    Returns None if vLLM 8B is not configured.
    """
    if not settings.vllm_available:
        return None

    from app.infrastructure.llm.vllm import VLLMProvider

    return VLLMProvider(
        base_url=settings.gcp_vllm_base_url,
        model_name=settings.vllm_model_name,
    )


@lru_cache
def get_vllm_32b_provider(
    settings: Settings = Depends(get_settings),
) -> "BaseLLMProvider | None":
    """Get vLLM 32B provider instance (면접 질문 생성).

    Returns None if vLLM 32B is not configured.
    RunPod 서버리스 또는 GCP GPU 서버에서 구동.
    """
    if not settings.vllm_32b_available:
        return None

    from app.infrastructure.llm.vllm import VLLMProvider

    return VLLMProvider(
        base_url=settings.vllm_32b_base_url,
        model_name=settings.vllm_32b_model_name,
    )


@lru_cache
def get_vectordb(
    settings: Settings = Depends(get_settings),
) -> "BaseVectorStore":
    """Get VectorDB instance.

    Returns ChromaDB. Uses server mode (HttpClient) when chroma_server_host is set (v2).
    """
    from app.infrastructure.vectordb.chroma import ChromaVectorStore

    return ChromaVectorStore(
        persist_directory=settings.chroma_persist_dir,
        api_key=settings.google_api_key,
        embedding_model=settings.gemini_embedding_model,
        chroma_server_host=settings.chroma_server_host,
        chroma_server_port=settings.chroma_server_port,
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
            socket_timeout=settings.redis_socket_timeout,
            socket_connect_timeout=settings.redis_connect_timeout,
            retry_on_timeout=settings.redis_retry_on_timeout,
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

_task_storage_instance = None        # FileTaskStore fallback 싱글톤
_redis_task_storage_instance = None  # RedisTaskStore 싱글톤 (ADR-130)


def get_legacy_task_storage(
    settings: Settings = Depends(get_settings),
):
    """Task 저장소 — Redis 우선, Redis 미설정 시 파일 폴백 (ADR-130).

    Redis URL이 설정된 환경(dev 포함)에서는 RedisTaskStore를 반환한다.
    Redis가 없는 순수 로컬 환경에서만 FileTaskStore로 폴백한다.

    k8s 분리 Deployment(FastAPI·Celery Worker 별도 파드) 환경에서
    /tmp 파일 공유 불가 문제를 해결하기 위해 Redis로 통일.

    Note:
        Celery 태스크에서 인수 없이 호출 시 settings = Depends(get_settings) 객체가
        전달되므로, 싱글톤 확인을 먼저 수행해 settings 접근을 최소화한다.
    """
    global _redis_task_storage_instance, _task_storage_instance

    # ── Fast path: 싱글톤이 이미 있으면 settings 접근 없이 즉시 반환 ──────────────
    # Celery 태스크는 인수 없이 호출하므로 settings가 Depends 객체일 수 있음.
    # 싱글톤 확인을 먼저 수행해 AttributeError를 방지한다.
    if _redis_task_storage_instance is not None:
        return _redis_task_storage_instance
    if _task_storage_instance is not None:
        return _task_storage_instance

    # ── 최초 초기화: 실제 Settings 객체 확보 ──────────────────────────────────────
    # FastAPI DI 없이 직접 호출(Celery 등)하면 settings = Depends 마커 객체이므로
    # get_settings()를 직접 호출해 실제 설정을 가져온다.
    actual_settings: Settings = settings if hasattr(settings, "redis_url") else get_settings()

    if actual_settings.redis_url:
        from app.utils.task_store import RedisTaskStore

        _redis_task_storage_instance = RedisTaskStore(
            redis_url=actual_settings.redis_url,  # DB 0, prefix "legacy_task:"으로 세션과 네임스페이스 분리
            ttl=actual_settings.redis_task_ttl,   # 기본 86400초 (24시간)
        )
        return _redis_task_storage_instance

    # Redis 없는 순수 로컬 환경 fallback
    from app.utils.task_store import FileTaskStore

    _task_storage_instance = FileTaskStore(storage_dir=actual_settings.task_storage_dir)
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
