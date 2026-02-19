"""
RAG (Retrieval-Augmented Generation) Service

Implements the RAG pipeline following the architecture diagram:
1. User asks Question/Interview request
2. Query VectorDB for relevant context (Resume/Result)
3. Send Question + Context to LLM
4. Generate and stream Answer

LCEL: 면접 질문 생성·일반 QnA는 LangChain 체인(prompt | llm | StrOutputParser)으로 처리 가능.
"""

import asyncio
import json
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, HumanMessage

from app.config.settings import get_settings
from app.domain.chat.chains import RAGChain
from app.prompts import (
    SYSTEM_FOLLOWUP,
    SYSTEM_GENERAL_CHAT,
    SYSTEM_RAG_CHAT,
    create_followup_prompt,
)
from app.prompts.interview import create_feedback_prompt

from .example_selector import get_few_shot_for_general
from .interview_templates import InterviewTemplateService
from .llm_service import LLMService
from .ocr_service import OCRService
from .vectordb_service import VectorDBService
from .vllm_service import VLLMService

if TYPE_CHECKING:
    from app.infrastructure.llm.langchain_wrapper import LangChainLLMGateway

logger = logging.getLogger(__name__)

# LCEL 면접 질문용 프롬프트 템플릿 (create_chain 변수 바인딩)
INTERVIEW_SINGLE_PROMPT = """이력서:
{resume}

채용공고:
{posting}

{feedback_block}

위 내용을 바탕으로 {interview_type} 면접 질문 1개를 JSON 형식으로 생성해주세요. 다른 설명 없이 JSON만 출력하세요.
{{"question": "질문 내용", "difficulty": "easy|medium|hard", "category": "{interview_type}", "follow_up": false}}"""

INTERVIEW_BATCH_PROMPT = """이력서:
{resume}

채용공고:
{posting}

{feedback_block}

위 내용을 바탕으로 {interview_type} 면접 질문을 {count}개 생성해주세요. 서로 다른 주제와 난이도(easy, medium, hard)를 섞어주세요.
반드시 아래 JSON 배열 형식으로만 응답하세요. 다른 설명 없이 JSON만 출력하세요:
[
  {{"question": "질문1", "difficulty": "easy", "category": "{interview_type}", "follow_up": false}},
  {{"question": "질문2", "difficulty": "medium", "category": "{interview_type}", "follow_up": false}},
  ...
]"""

# LCEL 일반 QnA용 프롬프트 (RAG context + 질문)
RAG_QNA_PROMPT = """관련 정보:
{context}

질문: {question}

위 관련 정보를 참고하여 질문에 답변해주세요. 관련 정보가 없으면 일반적인 지식으로 답변해주세요."""


def _parse_interview_json(text: str) -> dict[str, Any] | None:
    """Parse single interview question JSON from LLM output."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass
    return None


def _parse_interview_json_list(text: str) -> list[dict[str, Any]]:
    """Parse list of interview questions (JSON array) from LLM output."""
    if not text:
        return []
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
    if text.endswith("```"):
        text = text[:-3]
    start = text.find("[")
    end = text.rfind("]") + 1
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end])
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            pass
    return []


class RAGService:
    """RAG Service for chatbot with VectorDB context retrieval"""

    def __init__(
        self,
        llm_service: LLMService,
        vectordb_service: VectorDBService,
        vllm_service: VLLMService | None = None,
        ocr_service: OCRService | None = None,
        langchain_gateway: "LangChainLLMGateway | None" = None,
        rag_chain: RAGChain | None = None,
    ):
        """
        Initialize RAG Service

        Args:
            llm_service: LLM service instance (Gemini)
            vectordb_service: VectorDB service instance
            vllm_service: vLLM service instance (optional)
            ocr_service: OCR service instance (EasyOCR + Gemini Fallback)
            langchain_gateway: LCEL 체인용 Gateway (있으면 면접에서 체인 사용)
            rag_chain: RAGChain for MMR retrieval (optional; else VectorDB query fallback)
        """
        self.llm = llm_service
        self.vllm = vllm_service
        self.vectordb = vectordb_service
        self._langchain_gateway = langchain_gateway
        self._rag_chain = rag_chain
        # OCRService: 전달받거나 자동 생성
        self.ocr = ocr_service or OCRService(llm_service=llm_service)
        # 템플릿 기반 면접 질문 서비스
        self.interview_templates = InterviewTemplateService()
        logger.info(
            "RAG Service initialized (OCR + InterviewTemplate; LCEL=%s)",
            "on" if langchain_gateway else "off",
        )

    async def retrieve_all_documents(self, user_id: str, context_types: list[str] = None) -> str:
        """
        Retrieve ALL documents for a user (for analysis mode)

        Args:
            user_id: User ID
            context_types: List of collection types to retrieve

        Returns:
            Formatted context string with all documents (truncated if needed)
        """
        if context_types is None:
            context_types = ["resume", "job_posting"]
        try:
            # Portfolio 컬렉션 제외 (분석 시에는 사용자 데이터만)
            types_to_fetch = [ct for ct in context_types if ct != "portfolio"]
            if not types_to_fetch:
                return ""

            results_per_type = await asyncio.gather(
                *[
                    self.vectordb.get_all_documents_by_user(user_id=user_id, collection_type=ct)
                    for ct in types_to_fetch
                ]
            )
            all_results = []
            for collection_type, docs in zip(types_to_fetch, results_per_type, strict=True):
                all_results.extend([(collection_type, doc) for doc in docs])

            # Format context
            if not all_results:
                return ""

            # 청크 정렬: 같은 문서의 청크를 원본 순서대로 배치 (ADR-060 Phase 2)
            all_results.sort(key=lambda x: (
                x[1].get("metadata", {}).get("parent_document_id", ""),
                x[1].get("metadata", {}).get("chunk_index", 0),
            ))

            context_parts = []
            total_length = 0
            settings = get_settings()
            max_context_length = settings.rag_max_context_length

            for collection_type, doc in all_results:
                source = {
                    "resume": "이력서",
                    "job_posting": "채용공고",
                    "portfolio": "포트폴리오",
                }.get(collection_type, collection_type)

                doc_text = doc["text"]

                # 컨텍스트 길이 제한 (vLLM 8192 토큰 제한 고려)
                if total_length + len(doc_text) > max_context_length:
                    remaining = max_context_length - total_length
                    if remaining > 100:  # 최소 100자는 포함
                        doc_text = doc_text[:remaining] + "... (생략)"
                        context_parts.append(f"[출처: {source}]\n{doc_text}")
                    break

                context_parts.append(f"[출처: {source}]\n{doc_text}")
                total_length += len(doc_text)

            result = "\n\n".join(context_parts)
            logger.info(f"📊 컨텍스트 길이: {len(result)} 문자 (~{len(result)//4} 토큰)")
            return result

        except Exception as e:
            logger.error(f"Error retrieving all documents: {e}")
            return ""

    async def retrieve_context(
        self, query: str, user_id: str, context_types: list[str] = None, n_results: int = 3
    ) -> str:
        """
        Retrieve relevant context from VectorDB (RAGChain MMR when available, else VectorDB query).

        Args:
            query: User's query
            user_id: User ID for filtering
            context_types: List of collection types to search
            n_results: Number of results per collection (fallback path only)

        Returns:
            Formatted context string
        """
        if context_types is None:
            context_types = ["resume", "job_posting"]

        if self._rag_chain and (self._rag_chain._vectorstores or self._rag_chain._vectorstore):
            return await self._rag_chain.retrieve_context(query, user_id, context_types)

        try:

            async def query_one(collection_type: str):
                where_filter = None
                if collection_type != "portfolio" and user_id:
                    where_filter = {"user_id": user_id}
                results = await self.vectordb.query(
                    query_text=query,
                    collection_type=collection_type,
                    n_results=n_results,
                    where=where_filter,
                )
                return [(collection_type, r) for r in results]

            results_per_type = await asyncio.gather(*[query_one(ct) for ct in context_types])
            all_results = []
            for pairs in results_per_type:
                all_results.extend(pairs)

            # Format context
            if not all_results:
                return ""

            context_parts = []
            for collection_type, result in all_results:
                source = {
                    "resume": "이력서",
                    "job_posting": "채용공고",
                    "portfolio": "포트폴리오",
                }.get(collection_type, collection_type)

                context_parts.append(f"[출처: {source}]\n{result['text']}")

            return "\n\n".join(context_parts)

        except Exception as e:
            logger.error(f"Error retrieving context: {e}")
            return ""

    async def chat_with_rag(
        self,
        user_message: str,
        user_id: str,
        history: list[dict[str, str]] | None = None,
        use_rag: bool = True,
        context_types: list[str] = None,
        model: str = "gemini",
        n_results: int = 1,  # 기본값을 1로 설정하여 속도 개선
    ) -> AsyncIterator[str]:
        """
        Chat with RAG context retrieval

        Args:
            user_message: User's message
            user_id: User ID
            history: Chat history
            use_rag: Whether to use RAG (retrieve context)
            context_types: Collection types to search
            model: Model to use ("gemini" or "vllm")

        Yields:
            Response chunks
        """
        if context_types is None:
            context_types = ["resume", "job_posting"]
        try:
            context = None

            # Retrieve context if RAG is enabled
            if use_rag:
                # 사용자 ID는 로그에 포함하지 않음 (보안)
                logger.info("Retrieving RAG context for user")
                context = await self.retrieve_context(
                    query=user_message,
                    user_id=user_id,
                    context_types=context_types,
                    n_results=n_results,
                )

                if context:
                    logger.info(f"Retrieved context length: {len(context)} characters")
                else:
                    logger.info("No context found, using general knowledge")

            # System prompt for job search assistant (from prompts module)
            system_prompt = SYSTEM_RAG_CHAT if context else SYSTEM_GENERAL_CHAT

            # 평시 질의응답: 사용자 질문과 유사한 few-shot 예제 추가 (SemanticSimilarityExampleSelector)
            if not context and system_prompt == SYSTEM_GENERAL_CHAT:
                try:
                    few_shot = await get_few_shot_for_general(self.vectordb, user_message, k=2)
                    if few_shot:
                        system_prompt = (
                            system_prompt.rstrip()
                            + "\n\n## 추가 유사 예시 (참고)\n\n"
                            + few_shot
                            + "\n\n위와 같은 톤으로 답변해주세요."
                        )
                except Exception as e:
                    logger.debug("Few-shot general selection skipped: %s", e)

            # Select model
            if model == "vllm" and self.vllm:
                logger.info("Using vLLM model")
                async for chunk in self.vllm.generate_response(
                    user_message=user_message,
                    context=context,
                    history=history,
                    system_prompt=system_prompt,
                ):
                    yield chunk
            elif self._langchain_gateway:
                # LCEL 체인: history 있으면 create_chat_chain(MessagesPlaceholder), 없으면 create_chain
                logger.info("Using Gemini model (LCEL chain)")
                settings = get_settings()
                if history:
                    final_message = user_message
                    if context:
                        final_message = f"""관련 정보:
{context or "없음"}

질문: {user_message}

위 관련 정보를 참고하여 질문에 답변해주세요. 관련 정보가 없으면 일반적인 지식으로 답변해주세요."""
                    lc_messages: list[HumanMessage | AIMessage] = []
                    for msg in history:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "assistant":
                            lc_messages.append(AIMessage(content=content))
                        else:
                            lc_messages.append(HumanMessage(content=content))
                    lc_messages.append(HumanMessage(content=final_message))
                    chain = self._langchain_gateway.create_chat_chain(
                        system_prompt=system_prompt,
                        temperature=settings.llm_temperature_chat,
                        max_tokens=settings.llm_max_tokens_chat,
                    )
                    async for chunk in chain.astream({"messages": lc_messages}):
                        if chunk:
                            yield chunk
                else:
                    chain = self._langchain_gateway.create_chain(
                        RAG_QNA_PROMPT,
                        system_prompt=system_prompt,
                        temperature=settings.llm_temperature_chat,
                        max_tokens=settings.llm_max_tokens_chat,
                    )
                    async for chunk in chain.astream(
                        {"context": context or "없음", "question": user_message}
                    ):
                        if chunk:
                            yield chunk
            else:
                logger.info("Using Gemini model")
                async for chunk in self.llm.generate_response(
                    user_message=user_message,
                    context=context,
                    history=history,
                    system_prompt=system_prompt,
                    user_id=user_id,
                ):
                    yield chunk

        except Exception as e:
            logger.error(f"Error in RAG chat: {e}")
            yield f"죄송합니다. 응답 생성 중 오류가 발생했습니다: {str(e)}"

    async def analyze_resume_and_posting(
        self, user_id: str, resume_id: str | None = None, posting_id: str | None = None
    ) -> dict[str, Any]:
        """
        Analyze resume and job posting match

        Args:
            user_id: User ID
            resume_id: Resume document ID (optional, will search if not provided)
            posting_id: Posting document ID (optional, will search if not provided)

        Returns:
            Analysis result
        """
        try:

            async def get_resume_text():
                if resume_id:
                    doc = await self.vectordb.get_document(resume_id, "resume")
                    return doc["text"] if doc else ""
                results = await self.vectordb.query(
                    query_text="이력서 전체 내용",
                    collection_type="resume",
                    n_results=1,
                    where={"user_id": user_id},
                )
                return results[0]["text"] if results else ""

            async def get_posting_text():
                if posting_id:
                    doc = await self.vectordb.get_document(posting_id, "job_posting")
                    return doc["text"] if doc else ""
                results = await self.vectordb.query(
                    query_text="채용공고 전체 내용",
                    collection_type="job_posting",
                    n_results=1,
                    where={"user_id": user_id},
                )
                return results[0]["text"] if results else ""

            resume_text, posting_text = await asyncio.gather(get_resume_text(), get_posting_text())

            if not resume_text or not posting_text:
                raise ValueError("이력서 또는 채용공고를 찾을 수 없습니다")

            # Generate analysis
            analysis = await self.llm.generate_analysis(resume_text, posting_text, user_id=user_id)
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing resume and posting: {e}")
            raise

    async def generate_interview_question(
        self, user_id: str, interview_type: str = "technical"
    ) -> dict[str, Any]:
        """
        Generate interview question based on user's resume and job posting

        Args:
            user_id: User ID
            interview_type: "technical" or "personality"

        Returns:
            Interview question
        """
        try:

            async def fetch_feedback_single():
                try:
                    feedback_results = await self.vectordb.query(
                        query_text=f"{interview_type} 면접 약점 피드백",
                        collection_type="interview_feedback",
                        n_results=2,
                        where={"user_id": user_id, "interview_type": interview_type},
                    )
                    if feedback_results:
                        return "\n".join(r["text"] for r in feedback_results[:2])
                except Exception:
                    pass
                return ""

            resume_results, posting_results, feedback_text = await asyncio.gather(
                self.vectordb.query(
                    query_text="이력서 전체 내용",
                    collection_type="resume",
                    n_results=1,
                    where={"user_id": user_id},
                ),
                self.vectordb.query(
                    query_text="채용공고 전체 내용",
                    collection_type="job_posting",
                    n_results=1,
                    where={"user_id": user_id},
                ),
                fetch_feedback_single(),
            )
            resume_text = resume_results[0]["text"] if resume_results else "정보 없음"
            posting_text = posting_results[0]["text"] if posting_results else "정보 없음"

            # LCEL 체인 사용 (gateway 있으면)
            if self._langchain_gateway:
                type_label = "기술" if interview_type == "technical" else "인성"
                system_prompt = (
                    "당신은 기술 면접 전문가입니다. 이력서와 채용공고를 바탕으로 기술 면접 질문을 JSON 형식으로만 생성합니다."
                    if interview_type == "technical"
                    else "당신은 인성 면접 전문가입니다. 이력서를 바탕으로 인성 면접 질문을 JSON 형식으로만 생성합니다."
                )
                feedback_block = (
                    f"\n참고 - 이전 면접 피드백:\n{feedback_text[:800]}" if feedback_text else ""
                )
                resume_trim = resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
                posting_trim = (
                    posting_text[:500] + "..." if len(posting_text) > 500 else posting_text
                )
                settings = get_settings()
                chain = self._langchain_gateway.create_chain(
                    INTERVIEW_SINGLE_PROMPT,
                    system_prompt=system_prompt,
                    temperature=settings.llm_temperature_interview_question,
                    max_tokens=settings.llm_max_tokens_interview,
                )
                result_str = await chain.ainvoke(
                    {
                        "resume": resume_trim,
                        "posting": posting_trim,
                        "feedback_block": feedback_block,
                        "interview_type": type_label,
                    }
                )
                parsed = _parse_interview_json(result_str)
                if parsed:
                    return parsed
                return {
                    "question": result_str,
                    "difficulty": "medium",
                    "category": interview_type,
                    "follow_up": False,
                }

            # Fallback: 기존 Gemini SDK 직접 호출
            question = await self.llm.generate_interview_question(
                resume_text,
                posting_text,
                interview_type,
                user_id=user_id,
                previous_feedback=feedback_text or None,
            )
            return question

        except Exception as e:
            logger.error(f"Error generating interview question: {e}")
            raise

    async def generate_interview_questions_batch(
        self,
        user_id: str,
        interview_type: str = "technical",
        count: int = 5,
    ) -> list[dict[str, Any]]:
        """
        면접 질문을 한번에 여러 개 생성 (배치 처리)

        Args:
            user_id: User ID
            interview_type: "technical" or "personality"
            count: Number of questions to generate (default: 5)

        Returns:
            List of interview questions
        """
        try:

            async def fetch_feedback():
                try:
                    feedback_results = await self.vectordb.query(
                        query_text=f"{interview_type} 면접 약점 피드백",
                        collection_type="interview_feedback",
                        n_results=2,
                        where={"user_id": user_id, "interview_type": interview_type},
                    )
                    if feedback_results:
                        return "\n".join(r["text"] for r in feedback_results[:2])
                except Exception:
                    pass
                return ""

            resume_results, posting_results, feedback_text = await asyncio.gather(
                self.vectordb.query(
                    query_text="이력서 전체 내용",
                    collection_type="resume",
                    n_results=1,
                    where={"user_id": user_id},
                ),
                self.vectordb.query(
                    query_text="채용공고 전체 내용",
                    collection_type="job_posting",
                    n_results=1,
                    where={"user_id": user_id},
                ),
                fetch_feedback(),
            )
            resume_text = resume_results[0]["text"] if resume_results else "정보 없음"
            posting_text = posting_results[0]["text"] if posting_results else "정보 없음"

            # 5개 질문 = 템플릿 3개 + LLM(프로젝트 기반) 2개
            template_count = min(3, count)
            llm_count = count - template_count  # 나머지는 LLM

            # 1단계: 템플릿 기반 질문 생성 (LLM 호출 없이 즉시)
            template_questions = self.interview_templates.generate_questions(
                resume_text=resume_text,
                posting_text=posting_text,
                interview_type=interview_type,
                count=template_count,
            )
            logger.info(f"✅ 템플릿 질문 {len(template_questions)}개 생성 완료")

            # 2단계: LLM 프로젝트 기반 질문 생성 (LCEL 체인 또는 기존 Gemini)
            llm_questions = []
            if llm_count > 0:
                try:
                    if self._langchain_gateway:
                        type_label = "기술" if interview_type == "technical" else "인성"
                        feedback_block = (
                            f"\n참고 - 이전 면접 피드백:\n{feedback_text[:800]}"
                            if feedback_text
                            else ""
                        )
                        resume_trim = (
                            resume_text[:800] + "..." if len(resume_text) > 800 else resume_text
                        )
                        posting_trim = (
                            posting_text[:800] + "..." if len(posting_text) > 800 else posting_text
                        )
                        settings = get_settings()
                        chain = self._langchain_gateway.create_chain_from_yaml(
                            "interview",
                            "interview_batch",
                            temperature=settings.llm_temperature_interview_question,
                            max_tokens=settings.llm_max_tokens_interview * 2,
                        )
                        result_str = await chain.ainvoke(
                            {
                                "resume": resume_trim,
                                "posting": posting_trim,
                                "feedback_block": feedback_block,
                                "interview_type": type_label,
                                "count": llm_count,
                            }
                        )
                        llm_questions = _parse_interview_json_list(result_str)[:llm_count]
                        logger.info(
                            f"✅ LCEL 체인으로 프로젝트 질문 {len(llm_questions)}개 생성 완료"
                        )
                    else:
                        llm_questions = await self.llm.generate_interview_questions_batch(
                            resume_text=resume_text,
                            posting_text=posting_text,
                            interview_type=interview_type,
                            count=llm_count,
                            user_id=user_id,
                            previous_feedback=feedback_text or None,
                        )
                        logger.info(f"✅ LLM 프로젝트 질문 {len(llm_questions)}개 생성 완료")
                except Exception as llm_err:
                    # LLM 실패 시 템플릿으로 폴백
                    logger.warning(f"LLM 질문 생성 실패 → 템플릿 폴백: {llm_err}")
                    fallback = self.interview_templates.generate_questions(
                        resume_text=resume_text,
                        posting_text=posting_text,
                        interview_type=interview_type,
                        count=llm_count,
                        asked_questions=[q["question"] for q in template_questions],
                    )
                    llm_questions = fallback

            return template_questions + llm_questions

        except Exception as e:
            logger.error(f"Error generating batch interview questions: {e}")
            raise

    async def evaluate_interview_answer(
        self, question: str, answer: str, history: list[dict[str, str]] | None = None
    ) -> AsyncIterator[str]:
        """
        Evaluate interview answer and provide feedback

        Args:
            question: Interview question
            answer: User's answer
            history: Previous Q&A history

        Yields:
            Evaluation and feedback chunks
        """
        try:
            evaluation_content = f"질문: {question}\n\n답변: {answer}"
            prompt = create_feedback_prompt(evaluation_content)
            system_prompt = (
                "당신은 면접 평가 전문가입니다. 답변을 분석하고 건설적인 피드백을 제공하세요."
            )

            async for chunk in self.llm.generate_response(
                user_message=prompt, context=None, history=history, system_prompt=system_prompt
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Error evaluating interview answer: {e}")
            yield f"죄송합니다. 평가 중 오류가 발생했습니다: {str(e)}"

    async def generate_followup_question(
        self,
        original_question: str,
        candidate_answer: str,
        star_analysis: dict[str, str] | None = None,
        model: str = "gemini",
        user_id: str | None = None,
    ) -> AsyncIterator[str]:
        """
        꼬리질문 생성 (STAR 분석 기반)

        Args:
            original_question: 원본 면접 질문
            candidate_answer: 지원자 답변
            star_analysis: STAR 분석 결과 (Optional)
            model: 사용할 모델 ("gemini" 또는 "vllm")
            user_id: 사용자 ID (Gemini 사용 시)

        Yields:
            꼬리질문 생성 스트리밍 청크
        """
        try:
            # STAR 분석이 없으면 기본값 사용
            if star_analysis is None:
                star_analysis = {
                    "situation": "unknown",
                    "task": "unknown",
                    "action": "unknown",
                    "result": "unknown",
                }

            # 꼬리질문 생성 프롬프트 (prompts 모듈 사용)
            followup_prompt = create_followup_prompt(
                original_question=original_question,
                candidate_answer=candidate_answer,
                star_analysis=star_analysis,
            )

            logger.info("🔍 [꼬리질문 생성] 시작")
            # 사용자 입력은 로그에 포함하지 않음 (보안)
            logger.info("   원본 질문: [REDACTED]")
            logger.info("   답변 길이: %d자", len(candidate_answer))
            # 모델명은 로그에 포함하지 않음 (보안)
            logger.info("   모델: [REDACTED]")

            # vLLM 또는 Gemini 선택
            if model == "vllm" and self.vllm:
                logger.info("💬 [vLLM] 꼬리질문 생성 시작")
                async for chunk in self.vllm.generate_response(
                    user_message=followup_prompt,
                    context=None,
                    history=[],
                    system_prompt=SYSTEM_FOLLOWUP,
                ):
                    yield chunk
            else:
                logger.info("💬 [Gemini] 꼬리질문 생성 시작")
                async for chunk in self.llm.generate_response(
                    user_message=followup_prompt,
                    context=None,
                    history=[],
                    system_prompt=SYSTEM_FOLLOWUP,
                    user_id=user_id,
                ):
                    yield chunk

        except Exception as e:
            logger.error(f"Error generating followup question: {e}")
            yield f"죄송합니다. 꼬리질문 생성 중 오류가 발생했습니다: {str(e)}"
