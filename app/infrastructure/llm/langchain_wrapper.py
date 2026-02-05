"""
LangChain LLM Gateway Implementation.

Provides unified LLM interface using LangChain for standardized LLM interactions.
"""

import logging
import os
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

logger = logging.getLogger(__name__)


class LangChainLLMGateway:
    """Unified LLM Gateway using LangChain.

    Provides a standardized interface for different LLM providers
    through LangChain abstractions.
    """

    def __init__(
        self,
        google_api_key: str | None = None,
        model_name: str = "gemini-2.0-flash",
        embedding_model: str = "models/text-embedding-004",
        temperature: float = 0.7,
    ):
        """Initialize LangChain LLM Gateway.

        Args:
            google_api_key: Google API key (uses GOOGLE_API_KEY env var if not provided).
            model_name: Default model name for text generation.
            embedding_model: Model name for embeddings.
            temperature: Default temperature for generation.
        """
        api_key = google_api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required")

        # Initialize Gemini LLM
        self._llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            convert_system_message_to_human=True,
        )

        # Initialize embeddings
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model,
            google_api_key=api_key,
        )

        self._model_name = model_name
        self._output_parser = StrOutputParser()

        logger.info(f"LangChainLLMGateway initialized with model: {model_name}")

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        """Get the underlying LLM instance."""
        return self._llm

    @property
    def embeddings(self) -> GoogleGenerativeAIEmbeddings:
        """Get the embeddings instance."""
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
        **kwargs: Any,
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

        # Override temperature if provided
        llm = self._llm
        if temperature is not None:
            llm = llm.with_config({"temperature": temperature})

        response = await llm.ainvoke(lc_messages)
        return response.content

    async def generate_stream(
        self,
        messages: list[dict[str, str]],
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
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

        # Override temperature if provided
        llm = self._llm
        if temperature is not None:
            llm = llm.with_config({"temperature": temperature})

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
        input_variables: list[str] | None = None,
        system_prompt: str | None = None,
    ):
        """Create a simple LangChain chain.

        Args:
            prompt_template: Prompt template string.
            input_variables: List of input variable names.
            system_prompt: Optional system instructions.

        Returns:
            LangChain chain (Runnable).
        """
        messages = []

        if system_prompt:
            messages.append(("system", system_prompt))

        messages.append(("human", prompt_template))

        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt | self._llm | self._output_parser

    def create_chat_chain(
        self,
        system_prompt: str | None = None,
    ):
        """Create a chat chain with message history support.

        Args:
            system_prompt: Optional system instructions.

        Returns:
            LangChain chat chain (Runnable).
        """
        messages = []

        if system_prompt:
            messages.append(("system", system_prompt))

        messages.append(MessagesPlaceholder(variable_name="messages"))

        prompt = ChatPromptTemplate.from_messages(messages)
        return prompt | self._llm | self._output_parser

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
