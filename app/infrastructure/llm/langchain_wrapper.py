"""
LangChain LLM Gateway Implementation.

Provides unified LLM interface using LangChain for standardized LLM interactions.
"""

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from langchain_community.cache import InMemoryCache
from langchain_core.embeddings import Embeddings
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from app.config.settings import get_settings
from app.prompts.loader import load_prompt_yaml

logger = logging.getLogger(__name__)


class _EmbeddingsWithKeyFallback(Embeddings):
    """여러 API 키로 생성한 임베딩을 순서대로 시도 (실패 시 다음 키)."""

    def __init__(
        self,
        embeddings_instances: list[GoogleGenerativeAIEmbeddings],
    ):
        self._instances = embeddings_instances

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        for i, emb in enumerate(self._instances):
            try:
                return emb.embed_documents(texts)
            except Exception as e:
                logger.warning("Embedding (documents) key %d failed: %s", i + 1, e)
                if i == len(self._instances) - 1:
                    raise
        raise RuntimeError("No embedding instance succeeded")

    def embed_query(self, text: str) -> list[float]:
        for i, emb in enumerate(self._instances):
            try:
                return emb.embed_query(text)
            except Exception as e:
                logger.warning("Embedding (query) key %d failed: %s", i + 1, e)
                if i == len(self._instances) - 1:
                    raise
        raise RuntimeError("No embedding instance succeeded")

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        for i, emb in enumerate(self._instances):
            try:
                return await emb.aembed_documents(texts)
            except Exception as e:
                logger.warning("Embedding (aembed_documents) key %d failed: %s", i + 1, e)
                if i == len(self._instances) - 1:
                    raise
        raise RuntimeError("No embedding instance succeeded")

    async def aembed_query(self, text: str) -> list[float]:
        for i, emb in enumerate(self._instances):
            try:
                return await emb.aembed_query(text)
            except Exception as e:
                logger.warning("Embedding (aembed_query) key %d failed: %s", i + 1, e)
                if i == len(self._instances) - 1:
                    raise
        raise RuntimeError("No embedding instance succeeded")


class LangChainLLMGateway:
    """Unified LLM Gateway using LangChain.

    Provides a standardized interface for different LLM providers
    through LangChain abstractions.
    Supports multiple API keys with random selection + fallback for rate limit handling.
    """

    def __init__(
        self,
        google_api_key: str | None = None,
        google_api_keys: list[str] | None = None,
        model_name: str | None = None,
        embedding_model: str = "gemini-embedding-001",
        temperature: float = 0.7,
    ):
        """Initialize LangChain LLM Gateway.

        Args:
            google_api_key: Google API key (단일 키, 하위 호환).
            google_api_keys: Google API keys (다중 키, 분산 처리용).
            model_name: Default model name for text generation.
            embedding_model: Model name for embeddings.
            temperature: Default temperature for generation.
        """
        model_name = model_name or get_settings().gemini_model
        import random

        # API 키 수집
        keys = []
        if google_api_keys:
            keys.extend(google_api_keys)
        if google_api_key and google_api_key not in keys:
            keys.append(google_api_key)
        if not keys:
            env_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
            if env_key:
                keys.append(env_key)

        if not keys:
            raise ValueError("GOOGLE_API_KEY 또는 GEMINI_API_KEY 환경 변수가 필요합니다.")

        # 랜덤 순서로 섞기 (분산 처리)
        random.shuffle(keys)

        # 각 키별 LLM 인스턴스 생성
        llm_instances = [
            ChatGoogleGenerativeAI(
                model=model_name,
                google_api_key=k,
                temperature=temperature,
                convert_system_message_to_human=True,
            )
            for k in keys
        ]

        # Primary LLM + fallbacks 설정
        if len(llm_instances) > 1:
            self._llm = llm_instances[0].with_fallbacks(llm_instances[1:])
            logger.info(f"LangChain Gateway: {len(keys)}개 API 키 분산 처리 (with_fallbacks)")
        else:
            self._llm = llm_instances[0]

        # Embeddings: 설정된 동일 API 키 사용. 다중 키 시 실패하면 다음 키로 시도
        if len(keys) > 1:
            embedding_list = [
                GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=k) for k in keys
            ]
            self._embeddings = _EmbeddingsWithKeyFallback(embedding_list)
            logger.info("Embeddings: %d개 API 키 (fallback 지원)", len(keys))
        else:
            self._embeddings = GoogleGenerativeAIEmbeddings(
                model=embedding_model,
                google_api_key=keys[0],
            )

        self._model_name = model_name
        self._output_parser = StrOutputParser()

        set_llm_cache(InMemoryCache())
        logger.info(
            f"LangChainLLMGateway initialized with model: {model_name} (InMemoryCache enabled)"
        )

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Get the underlying LLM instance."""
        return self._llm

    @property
    def embeddings(self) -> Embeddings:
        """Get the embeddings instance (단일 키 또는 다중 키 fallback)."""
        return self._embeddings

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    def _convert_messages(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> list[Any]:
        """Convert message dicts to LangChain message objects.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            system_prompt: Optional system prompt to prepend.

        Returns:
            List of LangChain message objects.
        """
        lc_messages = []

        # Add system message if provided
        if system_prompt:
            lc_messages.append(SystemMessage(content=system_prompt))

        # Convert messages
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:  # user
                lc_messages.append(HumanMessage(content=content))

        return lc_messages

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        """Generate a response using LangChain.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature (overrides default).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters.

        Returns:
            Generated text response.
        """
        lc_messages = self._convert_messages(messages, system_prompt)

        # Override temperature and max_tokens if provided
        llm = self._llm
        bind_kwargs = {}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None:
            bind_kwargs["max_output_tokens"] = max_tokens
        if bind_kwargs:
            llm = llm.bind(**bind_kwargs)

        response = await llm.ainvoke(lc_messages)
        return response.content

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        output_schema: type,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """구조화 출력을 강제하여 Pydantic 모델 인스턴스를 반환합니다.

        Args:
            messages: 메시지 딕셔너리 리스트.
            output_schema: 출력 Pydantic 모델 클래스.
            system_prompt: 선택적 시스템 프롬프트.
            temperature: 샘플링 온도.
            max_tokens: 최대 토큰 수.

        Returns:
            output_schema 타입의 Pydantic 모델 인스턴스.
        """
        lc_messages = self._convert_messages(messages, system_prompt)

        llm = self._llm
        bind_kwargs = {}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None:
            bind_kwargs["max_output_tokens"] = max_tokens
        if bind_kwargs:
            llm = llm.bind(**bind_kwargs)

        structured_llm = llm.with_structured_output(output_schema)
        return await structured_llm.ainvoke(lc_messages)

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> AsyncIterator[str]:
        """Generate a streaming response using LangChain.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            system_prompt: Optional system instructions.
            temperature: Sampling temperature (overrides default).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters.

        Yields:
            Chunks of generated text.
        """
        lc_messages = self._convert_messages(messages, system_prompt)

        # Override temperature and max_tokens if provided
        llm = self._llm
        bind_kwargs = {}
        if temperature is not None:
            bind_kwargs["temperature"] = temperature
        if max_tokens is not None:
            bind_kwargs["max_output_tokens"] = max_tokens
        if bind_kwargs:
            llm = llm.bind(**bind_kwargs)

        async for chunk in llm.astream(lc_messages):
            if chunk.content:
                yield chunk.content

    async def create_embedding(self, text: str) -> list[float]:
        """Create an embedding using LangChain.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector.
        """
        return await self._embeddings.aembed_query(text)

    async def create_embeddings(self, texts: list[str]) -> list[list[float]]:
        """Create embeddings for multiple texts.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.
        """
        return await self._embeddings.aembed_documents(texts)

    def create_chain(
        self,
        prompt_template: str,
        *,
        input_variables: list[str] | None = None,  # noqa: ARG002
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Create a simple LangChain chain (prompt | llm | StrOutputParser).

        Args:
            prompt_template: Prompt template string (e.g. with {resume}, {posting}).
            input_variables: List of input variable names (inferred from template if not set).
            system_prompt: Optional system instructions.
            temperature: Optional sampling temperature (binds to LLM).
            max_tokens: Optional max output tokens (binds to LLM).

        Returns:
            LangChain chain (Runnable). ainvoke(input_dict) returns str.
        """
        messages = []

        if system_prompt:
            messages.append(("system", system_prompt))

        messages.append(("human", prompt_template))

        prompt = ChatPromptTemplate.from_messages(messages)

        llm = self._llm
        if temperature is not None or max_tokens is not None:
            bind_kwargs = {}
            if temperature is not None:
                bind_kwargs["temperature"] = temperature
            if max_tokens is not None:
                bind_kwargs["max_output_tokens"] = max_tokens
            llm = llm.bind(**bind_kwargs)

        return prompt | llm | self._output_parser

    def create_chain_from_yaml(
        self,
        domain: str,
        name: str,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Create a chain from a YAML prompt file (system + human, optional ai).

        Loads templates/{domain}/{name}.yaml and builds ChatPromptTemplate
        from system/human/ai keys. Variable substitution is done via ainvoke(input_dict).

        Args:
            domain: Subdirectory name (e.g. interview, chat).
            name: File name without extension.
            temperature: Optional sampling temperature.
            max_tokens: Optional max output tokens.

        Returns:
            LangChain chain (Runnable). ainvoke(input_dict) returns str.
        """
        d = load_prompt_yaml(domain, name)
        messages = []
        if d.get("system"):
            messages.append(("system", d["system"]))
        messages.append(("human", d["human"]))
        if d.get("ai"):
            messages.append(("ai", d["ai"]))

        prompt = ChatPromptTemplate.from_messages(messages)

        llm = self._llm
        if temperature is not None or max_tokens is not None:
            bind_kwargs = {}
            if temperature is not None:
                bind_kwargs["temperature"] = temperature
            if max_tokens is not None:
                bind_kwargs["max_output_tokens"] = max_tokens
            llm = llm.bind(**bind_kwargs)

        return prompt | llm | self._output_parser

    def create_chat_chain(
        self,
        system_prompt: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Create a chat chain with message history support.

        Args:
            system_prompt: Optional system instructions.
            temperature: Optional sampling temperature (binds to LLM).
            max_tokens: Optional max output tokens (binds to LLM).

        Returns:
            LangChain chat chain (Runnable). ainvoke({"messages": [...]}) / astream({"messages": [...]}).
        """
        messages = []

        if system_prompt:
            messages.append(("system", system_prompt))

        messages.append(MessagesPlaceholder(variable_name="messages"))

        prompt = ChatPromptTemplate.from_messages(messages)

        llm = self._llm
        if temperature is not None or max_tokens is not None:
            bind_kwargs = {}
            if temperature is not None:
                bind_kwargs["temperature"] = temperature
            if max_tokens is not None:
                bind_kwargs["max_output_tokens"] = max_tokens
            llm = llm.bind(**bind_kwargs)

        return prompt | llm | self._output_parser

    # ============================================
    # ADR-106 Phase 2: LCEL 고급 패턴
    # ============================================

    def get_llm_with_retry(
        self,
        *,
        stop_after_attempt: int = 3,
        wait_exponential_jitter: bool = True,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """ADR-106: 자동 재시도가 적용된 LLM 반환.

        429 (Rate Limit), 500 등 일시적 오류 시 자동 재시도.

        Returns:
            Runnable LLM with retry.
        """
        llm = self._llm
        if temperature is not None or max_tokens is not None:
            bind_kwargs = {}
            if temperature is not None:
                bind_kwargs["temperature"] = temperature
            if max_tokens is not None:
                bind_kwargs["max_output_tokens"] = max_tokens
            llm = llm.bind(**bind_kwargs)

        return llm.with_retry(
            stop_after_attempt=stop_after_attempt,
            wait_exponential_jitter=wait_exponential_jitter,
        )

    def get_llm_with_fallbacks(
        self,
        fallback_llms: list,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """ADR-106: 폴백 LLM이 연결된 LLM 반환 (Gemini → vLLM 등).

        Args:
            fallback_llms: 폴백으로 사용할 LLM 리스트.
            temperature: 샘플링 온도.
            max_tokens: 최대 토큰 수.

        Returns:
            Runnable LLM with fallbacks.
        """
        llm = self._llm
        if temperature is not None or max_tokens is not None:
            bind_kwargs = {}
            if temperature is not None:
                bind_kwargs["temperature"] = temperature
            if max_tokens is not None:
                bind_kwargs["max_output_tokens"] = max_tokens
            llm = llm.bind(**bind_kwargs)

        if fallback_llms:
            return llm.with_fallbacks(fallback_llms)
        return llm

    def create_rag_chain(
        self,
        retrieve_fn,
        *,
        system_prompt: str | None = None,
        prompt_template: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        use_retry: bool = True,
    ):
        """ADR-106: LCEL RAG 체인 — 검색 → 프롬프트 → LLM → 파싱을 단일 체인으로 구성.

        Args:
            retrieve_fn: 검색 함수 (async callable). dict → str 반환.
                         입력: {"question": str, ...}, 출력: context str.
            system_prompt: 시스템 프롬프트.
            prompt_template: 사용자 프롬프트 템플릿 ({context}, {question} 포함).
            temperature: 샘플링 온도.
            max_tokens: 최대 토큰 수.
            use_retry: 자동 재시도 적용 여부.

        Returns:
            LangChain Runnable. astream({"question": ..., ...}) / ainvoke({"question": ..., ...}).
        """
        from langchain_core.runnables import RunnableLambda, RunnablePassthrough

        # 검색 함수를 Runnable로 래핑
        retriever_runnable = RunnableLambda(retrieve_fn)

        # 프롬프트 구성
        if prompt_template is None:
            prompt_template = (
                "관련 정보:\n{context}\n\n"
                "질문: {question}\n\n"
                "위 관련 정보를 참고하여 질문에 답변해주세요. "
                "관련 정보가 없으면 일반적인 지식으로 답변해주세요."
            )

        messages = []
        if system_prompt:
            messages.append(("system", system_prompt))
        messages.append(("human", prompt_template))
        prompt = ChatPromptTemplate.from_messages(messages)

        # LLM (재시도 옵션)
        if use_retry:
            llm = self.get_llm_with_retry(temperature=temperature, max_tokens=max_tokens)
        else:
            llm = self._llm
            if temperature is not None or max_tokens is not None:
                bind_kwargs = {}
                if temperature is not None:
                    bind_kwargs["temperature"] = temperature
                if max_tokens is not None:
                    bind_kwargs["max_output_tokens"] = max_tokens
                llm = llm.bind(**bind_kwargs)

        # LCEL 파이프라인: 검색 → context 주입 → 프롬프트 → LLM → 파싱
        chain = (
            RunnablePassthrough.assign(context=retriever_runnable)
            | prompt
            | llm
            | self._output_parser
        )

        logger.debug("ADR-106: LCEL RAG chain created (retry=%s)", use_retry)
        return chain

    async def health_check(self) -> bool:
        """Check if the gateway is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            response = await self.generate(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=10,
            )
            return bool(response)
        except Exception:
            return False
