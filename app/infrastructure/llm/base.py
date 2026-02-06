"""
Abstract Base Class for LLM Providers.

Defines the interface for all LLM provider implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    """Response from LLM provider."""

    content: str
    model: str
    usage: dict[str, int] | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class EmbeddingResponse:
    """Response from embedding generation."""

    embedding: list[float]
    model: str
    usage: dict[str, int] | None = None


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers.

    All LLM provider implementations must inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Returns:
            LLMResponse with generated content.
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Yields:
            Chunks of generated text.
        """
        pass

    @abstractmethod
    async def create_embedding(
        self,
        text: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Create an embedding for the given text.

        Args:
            text: Text to embed.
            **kwargs: Additional provider-specific parameters.

        Returns:
            EmbeddingResponse with embedding vector.
        """
        pass

    async def health_check(self) -> bool:
        """Check if the provider is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            await self.generate(
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
            )
            return True
        except Exception:
            return False

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the model name."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass
