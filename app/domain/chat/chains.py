"""
LangChain RAG Chain Implementation.

Provides RAG (Retrieval-Augmented Generation) functionality using LangChain.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any

from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from app.infrastructure.llm.langchain_wrapper import LangChainLLMGateway

logger = logging.getLogger(__name__)


# RAG System Prompts
SYSTEM_RAG_CHAT = """당신은 사용자의 이력서와 채용공고를 기반으로 맞춤형 도움을 제공하는 AI 어시스턴트입니다.

다음 지침을 따라주세요:
1. 제공된 컨텍스트(이력서, 채용공고)를 참고하여 답변하세요.
2. 컨텍스트에 관련 정보가 없으면 일반적인 지식으로 답변하되, 이를 명시하세요.
3. 친절하고 전문적인 톤을 유지하세요.
4. 구체적이고 실용적인 조언을 제공하세요.
5. 답변은 한국어로 작성하세요."""

SYSTEM_FOLLOWUP = """당신은 사용자의 답변을 분석하고 관련 후속 질문을 생성하는 AI 어시스턴트입니다.

다음 지침을 따라주세요:
1. 사용자의 답변을 분석하고 더 깊은 이해를 위한 후속 질문을 생성하세요.
2. 질문은 구체적이고 관련성이 있어야 합니다.
3. 질문은 한국어로 작성하세요."""


class RAGChain:
    """RAG Chain using LangChain.

    Provides context-aware chat functionality with document retrieval.
    """

    def __init__(
        self,
        llm_gateway: LangChainLLMGateway,
        vectorstore: Chroma | None = None,
        max_context_length: int = 4000,
        retrieval_k: int = 3,
    ):
        """Initialize RAG Chain.

        Args:
            llm_gateway: LangChain LLM Gateway instance.
            vectorstore: LangChain Chroma vectorstore (optional).
            max_context_length: Maximum context length in characters.
            retrieval_k: Number of documents to retrieve.
        """
        self._llm_gateway = llm_gateway
        self._vectorstore = vectorstore
        self._max_context_length = max_context_length
        self._retrieval_k = retrieval_k
        self._output_parser = StrOutputParser()

        # Create RAG prompt template
        self._rag_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_RAG_CHAT),
            ("human", """관련 정보:
{context}

질문: {question}

위 관련 정보를 참고하여 질문에 답변해주세요. 관련 정보가 없으면 일반적인 지식으로 답변해주세요."""),
        ])

        # Create chat prompt template (without retrieval)
        self._chat_prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_RAG_CHAT),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{question}"),
        ])

        logger.info("RAGChain initialized")

    def set_vectorstore(self, vectorstore: Chroma) -> None:
        """Set the vectorstore for retrieval.

        Args:
            vectorstore: LangChain Chroma vectorstore.
        """
        self._vectorstore = vectorstore

    async def retrieve_context(
        self,
        query: str,
        user_id: str,
        collection_types: list[str] | None = None,
    ) -> str:
        """Retrieve relevant context from vectorstore.

        Args:
            query: Query text for retrieval.
            user_id: User ID for filtering documents.
            collection_types: Types of collections to search.

        Returns:
            Formatted context string.
        """
        if self._vectorstore is None:
            return ""

        if collection_types is None:
            collection_types = ["resume", "job_posting"]

        try:
            # Build retriever with user filter
            retriever = self._vectorstore.as_retriever(
                search_kwargs={
                    "k": self._retrieval_k,
                    "filter": {"user_id": user_id},
                }
            )

            # Retrieve documents
            docs = await retriever.aget_relevant_documents(query)

            # Format context
            context_parts = []
            total_length = 0

            for doc in docs:
                source = doc.metadata.get("collection_type", "document")
                source_name = {
                    "resume": "이력서",
                    "job_posting": "채용공고",
                    "portfolio": "포트폴리오",
                }.get(source, source)

                doc_text = doc.page_content

                # Truncate if needed
                remaining = self._max_context_length - total_length
                if len(doc_text) > remaining:
                    doc_text = doc_text[:remaining] + "..."

                context_parts.append(f"[{source_name}]\n{doc_text}")
                total_length += len(doc_text)

                if total_length >= self._max_context_length:
                    break

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

    async def chat(
        self,
        question: str,
        user_id: str,
        history: list[dict[str, str]] | None = None,
        use_retrieval: bool = True,
    ) -> str:
        """Generate a chat response.

        Args:
            question: User's question.
            user_id: User ID for context retrieval.
            history: Chat history.
            use_retrieval: Whether to use RAG retrieval.

        Returns:
            Generated response text.
        """
        if use_retrieval:
            # Retrieve context
            context = await self.retrieve_context(question, user_id)

            # Create chain with retrieval
            chain = self._rag_prompt | self._llm_gateway.llm | self._output_parser

            response = await chain.ainvoke({
                "context": context,
                "question": question,
            })
        else:
            # Create chain without retrieval
            chain = self._chat_prompt | self._llm_gateway.llm | self._output_parser

            # Convert history to LangChain messages
            from langchain_core.messages import AIMessage, HumanMessage
            lc_history = []
            if history:
                for msg in history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "assistant":
                        lc_history.append(AIMessage(content=content))
                    else:
                        lc_history.append(HumanMessage(content=content))

            response = await chain.ainvoke({
                "history": lc_history,
                "question": question,
            })

        return response

    async def chat_stream(
        self,
        question: str,
        user_id: str,
        history: list[dict[str, str]] | None = None,
        use_retrieval: bool = True,
    ) -> AsyncIterator[str]:
        """Generate a streaming chat response.

        Args:
            question: User's question.
            user_id: User ID for context retrieval.
            history: Chat history.
            use_retrieval: Whether to use RAG retrieval.

        Yields:
            Chunks of generated response.
        """
        if use_retrieval:
            # Retrieve context
            context = await self.retrieve_context(question, user_id)

            # Create chain with retrieval
            chain = self._rag_prompt | self._llm_gateway.llm | self._output_parser

            async for chunk in chain.astream({
                "context": context,
                "question": question,
            }):
                yield chunk
        else:
            # Create chain without retrieval
            chain = self._chat_prompt | self._llm_gateway.llm | self._output_parser

            # Convert history
            from langchain_core.messages import AIMessage, HumanMessage
            lc_history = []
            if history:
                for msg in history:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    if role == "assistant":
                        lc_history.append(AIMessage(content=content))
                    else:
                        lc_history.append(HumanMessage(content=content))

            async for chunk in chain.astream({
                "history": lc_history,
                "question": question,
            }):
                yield chunk

    async def generate_followup_question(
        self,
        original_question: str,
        user_answer: str,
        context: str | None = None,
    ) -> str:
        """Generate a follow-up question based on user's answer.

        Args:
            original_question: The original question asked.
            user_answer: User's answer to analyze.
            context: Additional context (optional).

        Returns:
            Generated follow-up question.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_FOLLOWUP),
            ("human", """원래 질문: {original_question}

사용자 답변: {user_answer}

{context_section}

사용자의 답변을 분석하고, 더 깊은 이해를 위한 후속 질문을 하나 생성해주세요."""),
        ])

        chain = prompt | self._llm_gateway.llm | self._output_parser

        context_section = f"관련 컨텍스트:\n{context}" if context else ""

        response = await chain.ainvoke({
            "original_question": original_question,
            "user_answer": user_answer,
            "context_section": context_section,
        })

        return response

    async def generate_analysis(
        self,
        resume_text: str,
        posting_text: str,
    ) -> dict[str, Any]:
        """Generate analysis of resume and job posting match.

        Args:
            resume_text: Resume text.
            posting_text: Job posting text.

        Returns:
            Analysis result as dictionary.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 이력서와 채용공고를 분석하는 AI 전문가입니다.
주어진 이력서와 채용공고를 비교 분석하여 JSON 형식으로 결과를 제공하세요.

다음 항목을 포함해주세요:
- resume_analysis: 이력서 분석 (strengths, weaknesses 배열)
- posting_analysis: 채용공고 분석 (company, position, required_skills, preferred_skills)
- matching: 매칭 분석 (score, matches, gaps)"""),
            ("human", """이력서:
{resume_text}

채용공고:
{posting_text}

위 이력서와 채용공고를 분석하여 JSON 형식으로 결과를 제공해주세요."""),
        ])

        chain = prompt | self._llm_gateway.llm | self._output_parser

        response = await chain.ainvoke({
            "resume_text": resume_text,
            "posting_text": posting_text,
        })

        # Parse JSON response
        try:
            import json
            # Try to extract JSON from response
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(response[start:end])
        except Exception as e:
            logger.warning(f"Failed to parse analysis JSON: {e}")

        return {
            "resume_analysis": {"strengths": [], "weaknesses": []},
            "posting_analysis": {},
            "matching": {"score": 0, "matches": [], "gaps": []},
            "raw_response": response,
        }
