"""
RAG (Retrieval-Augmented Generation) Service

Implements the RAG pipeline following the architecture diagram:
1. User asks Question/Interview request
2. Query VectorDB for relevant context (Resume/Result)
3. Send Question + Context to LLM
4. Generate and stream Answer
"""

import logging
from typing import List, Dict, Any, Optional, AsyncIterator, Union
from .llm_service import LLMService
from .vllm_service import VLLMService
from .vectordb_service import VectorDBService
from app.prompts import (
    SYSTEM_GENERAL_CHAT,
    SYSTEM_RAG_CHAT,
    SYSTEM_FOLLOWUP,
    create_rag_prompt,
    create_followup_prompt,
)

logger = logging.getLogger(__name__)


class RAGService:
    """RAG Service for chatbot with VectorDB context retrieval"""

    def __init__(
        self, 
        llm_service: LLMService, 
        vectordb_service: VectorDBService,
        vllm_service: Optional[VLLMService] = None
    ):
        """
        Initialize RAG Service

        Args:
            llm_service: LLM service instance (Gemini)
            vectordb_service: VectorDB service instance
            vllm_service: vLLM service instance (optional)
        """
        self.llm = llm_service
        self.vllm = vllm_service
        self.vectordb = vectordb_service
        logger.info("RAG Service initialized")

    async def retrieve_all_documents(
        self,
        user_id: str,
        context_types: List[str] = ["resume", "job_posting"]
    ) -> str:
        """
        Retrieve ALL documents for a user (for analysis mode)

        Args:
            user_id: User ID
            context_types: List of collection types to retrieve

        Returns:
            Formatted context string with all documents (truncated if needed)
        """
        try:
            all_results = []

            # Get all documents from each collection type
            for collection_type in context_types:
                # Portfolio 컬렉션은 user_id 필터 없이 검색하지 않음 (분석 시에는 사용자 데이터만)
                if collection_type == "portfolio":
                    continue

                docs = await self.vectordb.get_all_documents_by_user(
                    user_id=user_id,
                    collection_type=collection_type
                )
                all_results.extend([(collection_type, doc) for doc in docs])

            # Format context
            if not all_results:
                return ""

            context_parts = []
            total_length = 0
            max_context_length = 4000  # ~1000 tokens (4 chars ≈ 1 token)
            
            for collection_type, doc in all_results:
                source = {
                    "resume": "이력서",
                    "job_posting": "채용공고",
                    "portfolio": "포트폴리오"
                }.get(collection_type, collection_type)

                doc_text = doc['text']
                
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
        self,
        query: str,
        user_id: str,
        context_types: List[str] = ["resume", "job_posting"],
        n_results: int = 3
    ) -> str:
        """
        Retrieve relevant context from VectorDB

        Args:
            query: User's query
            user_id: User ID for filtering
            context_types: List of collection types to search
            n_results: Number of results per collection

        Returns:
            Formatted context string
        """
        try:
            all_results = []

            # Query each collection type
            for collection_type in context_types:
                # Portfolio (면접 질문) 컬렉션은 user_id 필터 없이 검색 (공통 데이터)
                where_filter = None
                if collection_type != "portfolio" and user_id:
                    where_filter = {"user_id": user_id}
                
                results = await self.vectordb.query(
                    query_text=query,
                    collection_type=collection_type,
                    n_results=n_results,
                    where=where_filter
                )
                all_results.extend([(collection_type, r) for r in results])

            # Format context
            if not all_results:
                return ""

            context_parts = []
            for collection_type, result in all_results:
                source = {
                    "resume": "이력서",
                    "job_posting": "채용공고",
                    "portfolio": "포트폴리오"
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
        history: Optional[List[Dict[str, str]]] = None,
        use_rag: bool = True,
        context_types: List[str] = ["resume", "job_posting"],
        model: str = "gemini",
        n_results: int = 1  # 기본값을 1로 설정하여 속도 개선
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
        try:
            context = None

            # Retrieve context if RAG is enabled
            if use_rag:
                logger.info(f"Retrieving RAG context for user {user_id}")
                context = await self.retrieve_context(
                    query=user_message,
                    user_id=user_id,
                    context_types=context_types,
                    n_results=n_results
                )

                if context:
                    logger.info(f"Retrieved context length: {len(context)} characters")
                else:
                    logger.info("No context found, using general knowledge")

            # System prompt for job search assistant (from prompts module)
            system_prompt = SYSTEM_RAG_CHAT if context else SYSTEM_GENERAL_CHAT

            # Select model
            if model == "vllm" and self.vllm:
                logger.info("Using vLLM model")
                async for chunk in self.vllm.generate_response(
                    user_message=user_message,
                    context=context,
                    history=history,
                    system_prompt=system_prompt
                ):
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
        self,
        user_id: str,
        resume_id: Optional[str] = None,
        posting_id: Optional[str] = None
    ) -> Dict[str, Any]:
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
            # Get resume text
            if resume_id:
                resume_doc = await self.vectordb.get_document(resume_id, "resume")
                resume_text = resume_doc['text'] if resume_doc else ""
            else:
                # Search for user's resume
                resume_results = await self.vectordb.query(
                    query_text="이력서 전체 내용",
                    collection_type="resume",
                    n_results=1,
                    where={"user_id": user_id}
                )
                resume_text = resume_results[0]['text'] if resume_results else ""

            # Get posting text
            if posting_id:
                posting_doc = await self.vectordb.get_document(posting_id, "job_posting")
                posting_text = posting_doc['text'] if posting_doc else ""
            else:
                # Search for recent posting
                posting_results = await self.vectordb.query(
                    query_text="채용공고 전체 내용",
                    collection_type="job_posting",
                    n_results=1,
                    where={"user_id": user_id}
                )
                posting_text = posting_results[0]['text'] if posting_results else ""

            if not resume_text or not posting_text:
                raise ValueError("이력서 또는 채용공고를 찾을 수 없습니다")

            # Generate analysis
            analysis = await self.llm.generate_analysis(resume_text, posting_text, user_id=user_id)
            return analysis

        except Exception as e:
            logger.error(f"Error analyzing resume and posting: {e}")
            raise

    async def generate_interview_question(
        self,
        user_id: str,
        interview_type: str = "technical"
    ) -> Dict[str, Any]:
        """
        Generate interview question based on user's resume and job posting

        Args:
            user_id: User ID
            interview_type: "technical" or "personality"

        Returns:
            Interview question
        """
        try:
            # Get resume
            resume_results = await self.vectordb.query(
                query_text="이력서 전체 내용",
                collection_type="resume",
                n_results=1,
                where={"user_id": user_id}
            )
            resume_text = resume_results[0]['text'] if resume_results else ""

            # Get posting
            posting_results = await self.vectordb.query(
                query_text="채용공고 전체 내용",
                collection_type="job_posting",
                n_results=1,
                where={"user_id": user_id}
            )
            posting_text = posting_results[0]['text'] if posting_results else ""

            if not resume_text:
                resume_text = "정보 없음"

            if not posting_text:
                posting_text = "정보 없음"

            # Generate question
            question = await self.llm.generate_interview_question(
                resume_text, posting_text, interview_type, user_id=user_id
            )
            return question

        except Exception as e:
            logger.error(f"Error generating interview question: {e}")
            raise

    async def evaluate_interview_answer(
        self,
        question: str,
        answer: str,
        history: Optional[List[Dict[str, str]]] = None
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
            prompt = f"""면접 질문과 답변을 평가하고 피드백을 제공해주세요.

질문: {question}

답변: {answer}

다음 항목에 대해 피드백해주세요:
1. 좋은 점 (good_points)
2. 개선할 점 (improvements)
3. 모범 답안 예시 (example_answer)

친절하고 건설적으로 피드백해주세요."""

            system_prompt = "당신은 면접 평가 전문가입니다. 답변을 분석하고 건설적인 피드백을 제공하세요."

            async for chunk in self.llm.generate_response(
                user_message=prompt,
                context=None,
                history=history,
                system_prompt=system_prompt
            ):
                yield chunk

        except Exception as e:
            logger.error(f"Error evaluating interview answer: {e}")
            yield f"죄송합니다. 평가 중 오류가 발생했습니다: {str(e)}"

    async def generate_followup_question(
        self,
        original_question: str,
        candidate_answer: str,
        star_analysis: Optional[Dict[str, str]] = None,
        model: str = "gemini",
        user_id: Optional[str] = None
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
                    "result": "unknown"
                }
            
            # 꼬리질문 생성 프롬프트 (prompts 모듈 사용)
            followup_prompt = create_followup_prompt(
                original_question=original_question,
                candidate_answer=candidate_answer,
                star_analysis=star_analysis
            )
            
            logger.info(f"🔍 [꼬리질문 생성] 시작")
            logger.info(f"   원본 질문: {original_question[:50]}...")
            logger.info(f"   답변 길이: {len(candidate_answer)}자")
            logger.info(f"   모델: {model}")
            
            # vLLM 또는 Gemini 선택
            if model == "vllm" and self.vllm:
                logger.info(f"💬 [vLLM] 꼬리질문 생성 시작")
                async for chunk in self.vllm.generate_response(
                    user_message=followup_prompt,
                    context=None,
                    history=[],
                    system_prompt=SYSTEM_FOLLOWUP
                ):
                    yield chunk
            else:
                logger.info(f"💬 [Gemini] 꼬리질문 생성 시작")
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
