"""
Langfuse 클라이언트 유틸리티

Langfuse를 사용하여 LLM 호출을 추적하고 모니터링합니다.
"""

import logging
import os
from typing import Any, TypedDict

from langfuse import Langfuse

logger = logging.getLogger(__name__)

# 일부 langfuse 버전에서는 decorators 모듈이 없을 수 있음
try:
    from langfuse.decorators import observe  # type: ignore
except Exception:  # pragma: no cover
    observe = None

# 전역 Langfuse 클라이언트
_langfuse_client: Langfuse | None = None


class LangfuseTraceContext(TypedDict):
    client: Langfuse
    trace_id: str
    trace_name: str
    user_id: str | None
    metadata: dict[str, Any]


def get_langfuse_client() -> Langfuse | None:
    """
    Langfuse 클라이언트 싱글톤 인스턴스 반환

    환경 변수:
        LANGFUSE_PUBLIC_KEY: Langfuse Public Key
        LANGFUSE_SECRET_KEY: Langfuse Secret Key
        LANGFUSE_HOST: Langfuse Host URL (기본값: http://localhost:3000)

    Returns:
        Langfuse 클라이언트 인스턴스 또는 None (설정되지 않은 경우)
    """
    global _langfuse_client

    if _langfuse_client is not None:
        return _langfuse_client

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    # LANGFUSE_HOST 또는 LANGFUSE_BASE_URL 지원 (둘 다 있으면 LANGFUSE_HOST 우선)
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "http://localhost:3001"
    if host:
        host = host.strip()

    if not public_key or not secret_key:
        logger.warning(
            "Langfuse credentials not found. Set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY "
            "environment variables to enable Langfuse tracking."
        )
        return None

    try:
        _langfuse_client = Langfuse(public_key=public_key, secret_key=secret_key, host=host)
        logger.info(f"Langfuse client initialized (host: {host})")
        return _langfuse_client
    except Exception as e:
        logger.error(f"Failed to initialize Langfuse client: {e}")
        return None


def trace_llm_call(name: str, user_id: str | None = None, metadata: dict | None = None):
    """
    LLM 호출 추적을 위한 trace 생성

    Args:
        name: Trace 이름
        user_id: 사용자 ID (선택)
        metadata: 추가 메타데이터 (선택)

    Returns:
        Langfuse trace context(dict) 또는 None
    """
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        trace_id = client.create_trace_id()
        return {
            "client": client,
            "trace_id": trace_id,
            "trace_name": name,
            "user_id": user_id,
            "metadata": dict(metadata or {}),
        }
    except Exception as e:
        logger.error(f"Failed to create Langfuse trace: {e}")
        return None


def create_generation(
    trace: LangfuseTraceContext | None,
    name: str,
    model: str,
    input_text: str,
    output_text: str | None = None,
    metadata: dict | None = None,
):
    """
    Generation 생성 (LLM 호출 기록)

    Args:
        trace: Langfuse trace context(dict)
        name: Generation 이름
        model: 사용한 모델 이름
        input_text: 입력 텍스트
        output_text: 출력 텍스트 (선택)
        metadata: 추가 메타데이터 (선택)

    Returns:
        Langfuse generation 객체 또는 None
    """
    if trace is None:
        return None

    try:
        client = trace["client"]
        trace_id = trace["trace_id"]

        generation = client.start_generation(
            name=name,
            trace_context={"trace_id": trace_id},
            model=model,
            input=input_text,
            output=output_text,
            metadata=metadata or {},
        )
        # trace 메타/유저 정보 업데이트
        generation.update_trace(
            name=trace.get("trace_name"),
            user_id=trace.get("user_id"),
            metadata=trace.get("metadata") or {},
        )
        generation.end()
        return generation
    except Exception as e:
        logger.error(f"Failed to create Langfuse generation: {e}")
        return None


# 데코레이터를 사용한 간편한 추적
def observe_llm_call(name: str | None = None):
    """
    LLM 호출을 자동으로 추적하는 데코레이터

    사용 예:
        @observe_llm_call(name="chat_completion")
        async def chat(user_message: str):
            # LLM 호출 코드
            pass
    """
    if observe is None:
        # decorators가 없는 환경에서는 no-op 데코레이터 반환
        def _noop_decorator(fn):
            return fn

        return _noop_decorator

    return observe(name=name)
