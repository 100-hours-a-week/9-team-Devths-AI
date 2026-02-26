"""
Langfuse 클라이언트 유틸리티

Langfuse를 사용하여 LLM 호출을 추적하고 모니터링합니다.
"""

import contextlib
import logging
import os
from typing import Any, TypedDict

logger = logging.getLogger(__name__)

# langfuse는 선택 의존성: 설치되어 있지 않아도 서비스가 동작해야 함
try:
    from langfuse import Langfuse  # type: ignore

    LANGFUSE_AVAILABLE = True
except Exception:  # pragma: no cover
    Langfuse = Any  # type: ignore
    LANGFUSE_AVAILABLE = False

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

    if not LANGFUSE_AVAILABLE:
        return None

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
    usage_details: dict[str, int] | None = None,
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
        usage_details: 토큰 사용량 (선택) - Langfuse v3 `usage_details` 형식

    Returns:
        Langfuse generation 객체 또는 None
    """
    if trace is None:
        return None

    try:
        client = trace["client"]
        trace_id = trace["trace_id"]

        # Langfuse SDK v3에서는 start_observation(as_type="generation", usage_details=...) 사용을 권장.
        # 하위 호환을 위해 start_observation이 없거나 실패하면 start_generation로 fallback.
        generation = None
        if hasattr(client, "start_observation"):
            try:
                generation = client.start_observation(
                    name=name,
                    as_type="generation",
                    trace_context={"trace_id": trace_id},
                    model=model,
                    input=input_text,
                    output=output_text,
                    metadata=metadata or {},
                    usage_details=usage_details,
                )
            except TypeError:
                # 일부 버전에서 usage_details 시그니처가 다를 수 있어 fallback
                generation = None

        if generation is None:
            generation = client.start_generation(
                name=name,
                trace_context={"trace_id": trace_id},
                model=model,
                input=input_text,
                output=output_text,
                metadata=metadata or {},
            )
            # start_generation 경로에서도 update()가 지원되면 usage_details를 반영
            if usage_details and hasattr(generation, "update"):
                with contextlib.suppress(Exception):
                    generation.update(usage_details=usage_details)
        # trace 메타/유저 정보 업데이트
        if hasattr(generation, "update_trace"):
            generation.update_trace(
                name=trace.get("trace_name"),
                user_id=trace.get("user_id"),
                metadata=trace.get("metadata") or {},
            )
        if hasattr(generation, "end"):
            generation.end()
        return generation
    except Exception as e:
        logger.error(f"Failed to create Langfuse generation: {e}")
        return None


def record_score(
    trace: "LangfuseTraceContext | None",
    name: str,
    value: float,
    comment: str | None = None,
) -> None:
    """ADR-091: Judge LLM이 채점한 점수를 Langfuse trace에 기록.

    Langfuse 대시보드에서 모델/파이프라인 단계별 품질 점수를 확인할 수 있다.
    Langfuse 미설정 또는 오류 시 no-op으로 동작 (서비스 중단 없음).

    Args:
        trace: trace_llm_call()로 생성한 컨텍스트
        name: 채점 기준명 ("relevance", "accuracy", "fluency", "completeness", "overall")
        value: 점수 (1.0 ~ 5.0)
        comment: 채점 근거 텍스트 (선택)
    """
    if trace is None:
        return
    try:
        client = trace["client"]
        client.score(
            trace_id=trace["trace_id"],
            name=name,
            value=value,
            comment=comment,
        )
    except Exception as e:
        logger.error("Langfuse score 기록 실패 (name=%s): %s", name, e)


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


def get_langfuse_callback_handler(
    session_id: str | None = None,
    user_id: str | None = None,
    trace_name: str | None = None,
) -> Any | None:
    """LangChain CallbackHandler 반환 (ADR-068).

    LCEL chain / LangGraph ainvoke의 config={"callbacks": [...]}에 주입한다.
    Langfuse 미설치·미설정 시 None 반환 → 호출부에서 조건부 처리.

    Args:
        session_id: 사용자 세션 ID (Langfuse Session으로 기록)
        user_id: 사용자 ID
        trace_name: Langfuse trace 이름 (미지정 시 "langchain-trace")

    Returns:
        LangfuseCallbackHandler 인스턴스 또는 None
    """
    if not LANGFUSE_AVAILABLE:
        return None
    try:
        from langfuse.callback import CallbackHandler as LangfuseCallbackHandler  # type: ignore
    except Exception:
        return None

    public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = os.getenv("LANGFUSE_SECRET_KEY")
    host = os.getenv("LANGFUSE_HOST") or os.getenv("LANGFUSE_BASE_URL") or "http://localhost:3001"
    if host:
        host = host.strip()

    if not public_key or not secret_key:
        return None

    try:
        return LangfuseCallbackHandler(
            public_key=public_key,
            secret_key=secret_key,
            host=host,
            session_id=str(session_id) if session_id else None,
            user_id=str(user_id) if user_id else None,
            trace_name=trace_name or "langchain-trace",
        )
    except Exception as e:
        logger.error("Failed to create LangfuseCallbackHandler: %s", e)
        return None
