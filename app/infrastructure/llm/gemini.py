"""
Google Gemini LLM Provider Implementation.

Implements the BaseLLMProvider interface for Google Gemini API.
"""

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from google import genai
from google.genai import types

from .base import BaseLLMProvider, EmbeddingResponse, LLMResponse

logger = logging.getLogger(__name__)


class GeminiProvider(BaseLLMProvider):
    """Google Gemini LLM provider implementation."""

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-2.0-flash",
        embedding_model: str = "text-embedding-004",
    ):
        """Initialize Gemini provider.

        Args:
            api_key: Google API key (uses GOOGLE_API_KEY env var if not provided).
            model_name: Gemini model name for text generation.
            embedding_model: Gemini model name for embeddings.
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        self.client = genai.Client(api_key=api_key)
        self._model_name = model_name
        self._embedding_model = embedding_model

        logger.info(f"GeminiProvider initialized with model: {model_name}")

    @property
    def model_name(self) -> str:
        """Get the model name."""
        return self._model_name

    @property
    def provider_name(self) -> str:
        """Get the provider name."""
        return "gemini"

    async def generate(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """Generate a response from Gemini.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            system_prompt: System instructions.
            **kwargs: Additional parameters.

        Returns:
            LLMResponse with generated content.
        """
        # Convert messages to Gemini format
        contents = self._convert_messages(messages)

        # Create config
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 40),
            max_output_tokens=max_tokens or 2048,
            system_instruction=system_prompt,
        )

        # Generate response
        response = self.client.models.generate_content(
            model=self._model_name,
            contents=contents,
            config=config,
        )

        # Extract text from response
        content = ""
        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                content = "".join(
                    part.text for part in candidate.content.parts if hasattr(part, "text")
                )

        return LLMResponse(
            content=content,
            model=self._model_name,
            usage=self._extract_usage(response),
        )

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Generate a streaming response from Gemini.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            system_prompt: System instructions.
            **kwargs: Additional parameters.

        Yields:
            Chunks of generated text.
        """
        # Convert messages to Gemini format
        contents = self._convert_messages(messages)

        # Create config
        config = types.GenerateContentConfig(
            temperature=temperature,
            top_p=kwargs.get("top_p", 0.9),
            top_k=kwargs.get("top_k", 40),
            max_output_tokens=max_tokens or 2048,
            system_instruction=system_prompt,
        )

        # Generate streaming response
        response = self.client.models.generate_content_stream(
            model=self._model_name,
            contents=contents,
            config=config,
        )

        for chunk in response:
            if chunk.candidates and len(chunk.candidates) > 0:
                candidate = chunk.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            yield part.text

    async def create_embedding(
        self,
        text: str,
        **kwargs: Any,  # noqa: ARG002
    ) -> EmbeddingResponse:
        """Create an embedding using Gemini.

        Args:
            text: Text to embed.
            **kwargs: Additional parameters.

        Returns:
            EmbeddingResponse with embedding vector.
        """
        response = self.client.models.embed_content(
            model=self._embedding_model,
            contents=text,
        )

        return EmbeddingResponse(
            embedding=response.embeddings[0].values,
            model=self._embedding_model,
        )

    def _convert_messages(
        self, messages: list[dict[str, str]]
    ) -> list[types.Content]:
        """Convert messages to Gemini format.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.

        Returns:
            List of Gemini Content objects.
        """
        contents = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            # Gemini uses 'user' and 'model' roles
            gemini_role = "model" if role == "assistant" else "user"

            contents.append(
                types.Content(
                    role=gemini_role,
                    parts=[types.Part.from_text(text=content)],
                )
            )

        return contents

    def _extract_usage(self, response: Any) -> dict[str, int] | None:
        """Extract usage info from response.

        Args:
            response: Gemini response object.

        Returns:
            Usage dict or None.
        """
        try:
            if hasattr(response, "usage_metadata"):
                metadata = response.usage_metadata
                return {
                    "prompt_tokens": getattr(metadata, "prompt_token_count", 0),
                    "completion_tokens": getattr(metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(metadata, "total_token_count", 0),
                }
        except Exception:
            pass
        return None
