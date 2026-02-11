"""
OpenAI LLM Provider Implementation.

Implements the BaseLLMProvider interface for OpenAI API.
Used primarily for evaluation debate feature.
"""

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from openai import AsyncOpenAI

from .base import BaseLLMProvider, EmbeddingResponse, LLMResponse

logger = logging.getLogger(__name__)


class OpenAIProvider(BaseLLMProvider):
    """OpenAI LLM provider implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gpt-4o",
    ):
        """Initialize OpenAI provider.

        Args:
            api_key: OpenAI API key (uses OPENAI_API_KEY env var if not provided).
            model_name: OpenAI model name for text generation.
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for OpenAI provider")

        self.client = AsyncOpenAI(api_key=api_key)
        self._model_name = model_name

        logger.info(f"OpenAIProvider initialized with model: {model_name}")

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "openai"

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **_kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            system_prompt: System instructions.
            **kwargs: Additional parameters.

        Returns:
            LLMResponse with generated content.
        """
        openai_messages = self._build_messages(messages, system_prompt)

        response = await self.client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens or 4096,
        )

        content = response.choices[0].message.content or ""
        usage = None
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        return LLMResponse(
            content=content,
            model=self._model_name,
            usage=usage,
        )

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **_kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from OpenAI.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            system_prompt: System instructions.
            **kwargs: Additional parameters.

        Yields:
            Chunks of generated text.
        """
        openai_messages = self._build_messages(messages, system_prompt)

        stream = await self.client.chat.completions.create(
            model=self._model_name,
            messages=openai_messages,
            temperature=temperature,
            max_tokens=max_tokens or 4096,
            stream=True,
        )

        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    async def create_embedding(
        self,
        text: str,
        **_kwargs: Any,
    ) -> EmbeddingResponse:
        """Create an embedding using OpenAI.

        Args:
            text: Text to embed.
            **kwargs: Additional parameters.

        Returns:
            EmbeddingResponse with embedding vector.
        """
        response = await self.client.embeddings.create(
            model="text-embedding-3-small",
            input=text,
        )

        return EmbeddingResponse(
            embedding=response.data[0].embedding,
            model="text-embedding-3-small",
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            if response.usage
            else None,
        )

    def _build_messages(
        self,
        messages: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> list[dict[str, str]]:
        """Build OpenAI message format.

        Args:
            messages: List of message dicts.
            system_prompt: Optional system prompt.

        Returns:
            OpenAI-formatted messages list.
        """
        openai_messages = []

        if system_prompt:
            openai_messages.append({"role": "system", "content": system_prompt})

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            # Normalize role names
            if role == "model":
                role = "assistant"
            openai_messages.append({"role": role, "content": content})

        return openai_messages
