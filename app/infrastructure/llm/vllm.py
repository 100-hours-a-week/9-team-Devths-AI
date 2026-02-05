"""
vLLM Provider Implementation.

Implements the BaseLLMProvider interface for vLLM server (OpenAI-compatible API).
"""

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

import httpx

from .base import BaseLLMProvider, EmbeddingResponse, LLMResponse

logger = logging.getLogger(__name__)


class VLLMProvider(BaseLLMProvider):
    """vLLM provider implementation (OpenAI-compatible API)."""

    def __init__(
        self,
        base_url: str | None = None,
        model_name: str = "MLP-KTLim/llama-3-Korean-Bllossom-8B",
        timeout: float = 120.0,
    ):
        """Initialize vLLM provider.

        Args:
            base_url: vLLM server base URL (uses GCP_VLLM_BASE_URL env var if not provided).
            model_name: Model name on vLLM server.
            timeout: Request timeout in seconds.
        """
        self.base_url = base_url or os.getenv("GCP_VLLM_BASE_URL")
        if not self.base_url:
            raise ValueError("GCP_VLLM_BASE_URL environment variable is required")

        self._model_name = model_name
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)

        logger.info(f"VLLMProvider initialized with server: {self.base_url}")

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "vllm"

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from vLLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters.

        Returns:
            LLMResponse with generated content.
        """
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 2048,
            "stream": False,
        }

        response = await self._client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        content = ""
        if "choices" in data and len(data["choices"]) > 0:
            content = data["choices"][0].get("message", {}).get("content", "")

        return LLMResponse(
            content=content,
            model=self._model_name,
            usage=data.get("usage"),
        )

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from vLLM.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional parameters.

        Yields:
            Chunks of generated text.
        """
        url = f"{self.base_url}/v1/chat/completions"

        payload = {
            "model": self._model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens or 2048,
            "stream": True,
        }

        async with self._client.stream("POST", url, json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        import json

                        data = json.loads(data_str)
                        if "choices" in data and len(data["choices"]) > 0:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                    except Exception:
                        continue

    async def create_embedding(
        self,
        text: str,
        **kwargs: Any,
    ) -> EmbeddingResponse:
        """Create an embedding (not supported by default vLLM setup).

        Note: vLLM doesn't typically support embeddings.
        This will raise NotImplementedError.

        Args:
            text: Text to embed.
            **kwargs: Additional parameters.

        Raises:
            NotImplementedError: vLLM doesn't support embeddings.
        """
        raise NotImplementedError(
            "vLLM doesn't support embedding generation. Use Gemini for embeddings."
        )

    async def health_check(self) -> bool:
        """Check if the vLLM server is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            url = f"{self.base_url}/health"
            response = await self._client.get(url)
            return response.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
