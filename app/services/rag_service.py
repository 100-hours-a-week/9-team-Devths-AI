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

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

try:
    from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
except ImportError:
    LLMChainExtractor = None  # type: ignore[misc, assignment]

from app.config.settings import get_settings
from app.domain.chat.chains import RAGChain
from app.prompts import (
    SYSTEM_FOLLOWUP,
    SYSTEM_GENERAL_CHAT,
    SYSTEM_RAG_CHAT,
    create_followup_prompt,
)
from app.prompts.interview import create_feedback_prompt
from app.utils.log_sanitizer import sanitize_log_input

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

# RRF constant for ensemble retriever (ADR-069)
_RRF_K = 60

# ADR-077: 다중 쿼리 생성 프롬프트 (NORMAL 모드 일반 채팅용)
_MULTI_QUERY_SYSTEM = """당신은 면접 준비 도우미입니다.
사용자의 질문을 다양한 관점에서 검색 쿼리로 재작성하세요.
각 쿼리는 한 줄로 작성하고, 번호를 붙이지 마세요.

관점:
1. 기술 역량 중심: 사용 기술, 프레임워크, 라이브러리 관점
2. 경험/프로젝트 중심: 프로젝트명, 성과, 역할 관점
3. 직무 적합성 중심: 채용공고 요구사항, 자격 요건 관점"""

_MULTI_QUERY_HUMAN = "원래 질문: {question}"

# ADR-080: 질문 유형별 시간 감쇠율 (interview_feedback 최신성 반영)
_DECAY_RATES = {
    "technical": 0.1,  # 기술 질문: 최신 정보 우선 (반감기 ~6.6시간)
    "personality": 0.01,  # 인성 질문: 시간 거의 무관 (반감기 ~69시간)
    "general": 0.05,  # 일반: 중간 (반감기 ~13.5시간)
}

# ADR-078: 요약 임베딩 생성 프롬프트
_SUMMARY_FOR_EMBEDDING_TEMPLATE = "다음 문서의 핵심 내용을 2-3문장으로 요약하세요:\n\n{doc}"

# ADR-078: 가설 쿼리 생성 프롬프트
_HYPOTHETICAL_QUERIES_TEMPLATE = (
    "다음 문서를 읽고, 이 문서로 답변할 수 있는 질문 {count}개를 생성하세요.\n"
    "각 질문은 한 줄로 작성하고, 번호를 붙이지 마세요.\n\n문서: {doc}"
)

# ADR-079: 채용 트렌드 메타데이터 스키마 (SelfQueryRetriever용)
TREND_METADATA_FIELDS: list[dict] = [
    {
        "name": "year",
        "description": "채용 트렌드 데이터의 연도 (예: 2024, 2025, 2026)",
        "type": "integer",
    },
    {
        "name": "company",
        "description": "채용 기업명 (예: 카카오, 네이버, 삼성, 라인)",
        "type": "string",
    },
    {
        "name": "position",
        "description": "직무/포지션 (예: 백엔드, 프론트엔드, AI/ML)",
        "type": "string",
    },
    {
        "name": "category",
        "description": "데이터 카테고리 (예: 기술질문, 인성질문, 채용공고, 면접후기)",
        "type": "string",
    },
    {
        "name": "recruitment_type",
        "description": "채용 유형 (예: 공채, 수시, 인턴)",
        "type": "string",
    },
    {"name": "source", "description": "데이터 출처 (예: webbaseloader)", "type": "string"},
]


def _rrf_merge(
    dense_pairs: list[tuple[str, Document]],
    sparse_pairs: list[tuple[str, Document]],
    dense_weight: float,
    sparse_weight: float,
    top_k: int = 0,
) -> list[tuple[str, Document]]:
    """Merge dense (MMR) and sparse (BM25) results with Reciprocal Rank Fusion.

    top_k: max documents to return (0 = no limit).
    """

    def _key(doc: Document) -> str:
        return (doc.page_content or "")[:500]

    score_map: dict[str, tuple[str, Document, float]] = {}

    for rank, (ct, doc) in enumerate(dense_pairs, start=1):
        key = _key(doc)
        rrf = dense_weight / (_RRF_K + rank)
        if key not in score_map:
            score_map[key] = (ct, doc, 0.0)
        ct_cur, doc_cur, s_cur = score_map[key]
        score_map[key] = (ct_cur, doc_cur, s_cur + rrf)

    for rank, (ct, doc) in enumerate(sparse_pairs, start=1):
        key = _key(doc)
        rrf = sparse_weight / (_RRF_K + rank)
        if key not in score_map:
            score_map[key] = (ct, doc, 0.0)
        ct_cur, doc_cur, s_cur = score_map[key]
        score_map[key] = (ct_cur, doc_cur, s_cur + rrf)

    merged = sorted(score_map.values(), key=lambda x: x[2], reverse=True)
    pairs = [(ct, doc) for ct, doc, _ in merged]
    if top_k > 0:
        pairs = pairs[:top_k]
    return pairs


# ADR-104: FlashRank 리랭커 싱글톤 (모델 로딩 1회만)
_flashrank_ranker: Any = None


def _get_flashrank_ranker(model_name: str = "ms-marco-MiniLM-L-12-v2") -> Any:
    """ADR-104: FlashRank Ranker 싱글톤 반환. 최초 호출 시 모델 다운로드 (~60MB)."""
    global _flashrank_ranker  # noqa: PLW0603
    if _flashrank_ranker is None:
        from flashrank import Ranker

        _flashrank_ranker = Ranker(model_name=model_name, cache_dir="./flashrank_cache")
        _flashrank_ranker._loaded_model_name = model_name  # 추적용
        logger.info("ADR-104: FlashRank Ranker initialized (model=%s)", model_name)
    elif getattr(_flashrank_ranker, "_loaded_model_name", None) != model_name:
        logger.warning(
            "ADR-104: FlashRank 모델 불일치 (loaded=%s, requested=%s). 서버 재시작 필요.",
            getattr(_flashrank_ranker, "_loaded_model_name", "unknown"),
            model_name,
        )
    return _flashrank_ranker


async def _rerank(
    pairs: list[tuple[str, Document]],
    top_k: int,
    query: str = "",
) -> list[tuple[str, Document]]:
    """ADR-104: FlashRank 리랭커로 의미적 재정렬 후 top_k 반환.

    settings.rag_use_flashrank=True이고 query가 있으면 FlashRank 적용.
    그 외에는 기존 동작(단순 top_k 자르기) 유지.
    실패 시 기존 방식으로 폴백.
    """
    if top_k <= 0 or not pairs:
        return pairs

    settings = get_settings()
    if not settings.rag_use_flashrank or not query:
        return pairs[:top_k]

    try:
        from flashrank import RerankRequest

        ranker = _get_flashrank_ranker(settings.rag_flashrank_model)
        passages = [
            {"id": idx, "text": (doc.page_content or "")[:1000]}
            for idx, (_, doc) in enumerate(pairs)
        ]
        rerank_req = RerankRequest(query=query, passages=passages)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(None, ranker.rerank, rerank_req)

        # FlashRank 점수순으로 pairs 재정렬
        id_to_score: dict[int, float] = {int(r["id"]): float(r["score"]) for r in results}
        scored = [(id_to_score.get(i, 0.0), pair) for i, pair in enumerate(pairs)]
        scored.sort(key=lambda x: x[0], reverse=True)
        reranked = [pair for _, pair in scored[:top_k]]

        logger.info(
            "ADR-104: FlashRank reranked %d → %d docs (top=%.3f, bottom=%.3f)",
            len(pairs),
            len(reranked),
            scored[0][0] if scored else 0,
            scored[min(top_k - 1, len(scored) - 1)][0] if scored else 0,
        )
        return reranked
    except Exception as e:
        logger.warning("ADR-104: FlashRank 실패, top_k 폴백: %s", e)
        return pairs[:top_k]


async def _compress_pairs_with_llm(
    query: str,
    pairs: list[tuple[str, Document]],
    top_k: int,
    llm: Any,
) -> list[tuple[str, Document]]:
    """ADR-069 Phase 3: 질문 관련 문장만 추출해 컨텍스트 압축. top_k개만 압축해 LLM 호출 수 제한."""
    if not pairs or LLMChainExtractor is None or llm is None:
        return pairs
    to_compress = pairs[:top_k]
    docs_only = [doc for _, doc in to_compress]
    try:
        compressor = LLMChainExtractor.from_llm(llm)
        compressed_docs = await asyncio.to_thread(compressor.compress_documents, docs_only, query)
        if len(compressed_docs) == len(to_compress):
            return list(zip([ct for ct, _ in to_compress], compressed_docs, strict=True))
    except Exception as e:
        logger.warning("Document compressor failed, using uncompressed: %s", e)
    return pairs


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


# ADR-076: ParentDocumentRetriever 대상 컬렉션 타입 (모듈 레벨 상수 — 매 호출마다 재생성 방지)
_PARENT_RETRIEVER_TYPES: frozenset[str] = frozenset(
    {"resumes", "resume", "portfolios", "portfolio"}
)


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
        # ADR-106 Phase 4: RAPTOR 서비스 (설정 활성화 시에만 초기화)
        self._raptor: Any = None
        if get_settings().raptor_enabled:
            from app.services.raptor_service import RaptorService as _RaptorSvc

            self._raptor = _RaptorSvc(
                vectordb_service=vectordb_service,
                langchain_gateway=langchain_gateway,
            )
        logger.info(
            "RAG Service initialized (OCR + InterviewTemplate; LCEL=%s, RAPTOR=%s)",
            "on" if langchain_gateway else "off",
            "on" if self._raptor else "off",
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
            all_results.sort(
                key=lambda x: (
                    x[1].get("metadata", {}).get("parent_document_id", ""),
                    x[1].get("metadata", {}).get("chunk_index", 0),
                )
            )

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

    async def _generate_multi_queries(self, query: str) -> list[str]:
        """ADR-077: LCEL 체인으로 다중 검색 쿼리 생성.

        사용자 질문을 기술역량/경험·프로젝트/직무적합성 3가지 관점으로 재작성.
        실패 시 원래 쿼리만 반환.
        """
        if not self._langchain_gateway:
            return [query]

        settings = get_settings()
        try:
            from langchain_core.output_parsers import StrOutputParser
            from langchain_core.prompts import ChatPromptTemplate

            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", _MULTI_QUERY_SYSTEM),
                    ("human", _MULTI_QUERY_HUMAN),
                ]
            )
            chain = prompt | self._langchain_gateway.llm | StrOutputParser()
            result = await chain.ainvoke({"question": query})

            lines = [line.strip() for line in result.strip().split("\n") if line.strip()]
            # 설정된 개수만큼만 사용
            queries = lines[: settings.rag_multi_query_count]
            if not queries:
                return [query]

            # 원래 쿼리도 포함하여 누락 방지
            if query not in queries:
                queries.insert(0, query)

            # SAST: 사용자 입력(query)은 로그에 넣지 않음 — 개수만 기록
            logger.info("ADR-077: MultiQuery generated %d queries", len(queries))
            for i, q in enumerate(queries):
                logger.debug("  MQ[%d]: %s", i, q[:80])
            return queries
        except Exception as e:
            logger.warning("ADR-077: MultiQuery generation failed, using original query: %s", e)
            return [query]

    @staticmethod
    def _apply_time_weighting(
        results: list[dict],
        interview_type: str = "general",
    ) -> list[dict]:
        """ADR-080: 시간 감쇠 점수 적용 후 재정렬.

        공식: score = semantic_similarity + (1.0 - decay_rate) ^ hours_passed
        created_at 메타데이터가 없는 기존 문서는 시간 점수 0으로 처리 (의미 유사도만 사용).
        """
        from datetime import datetime, timezone

        settings = get_settings()
        decay_rate = getattr(
            settings, f"rag_decay_rate_{interview_type}", settings.rag_decay_rate_general
        )

        now = datetime.now(timezone.utc)
        scored: list[tuple[float, dict]] = []

        for r in results:
            # ChromaDB distance는 L2 거리 (lower = more similar)
            # similarity = 1 / (1 + distance) 으로 변환
            distance = r.get("distance", 0.0)
            similarity = 1.0 / (1.0 + distance)

            # 시간 점수 계산
            created_at_str = (r.get("metadata") or {}).get("created_at")
            if created_at_str:
                try:
                    created_at = datetime.fromisoformat(created_at_str)
                    if created_at.tzinfo is None:
                        created_at = created_at.replace(tzinfo=timezone.utc)
                    hours_passed = (now - created_at).total_seconds() / 3600.0
                    time_score = (1.0 - decay_rate) ** max(hours_passed, 0)
                except (ValueError, TypeError):
                    time_score = 0.0
            else:
                time_score = 0.0  # created_at 없는 기존 문서

            total_score = similarity + time_score
            scored.append((total_score, r))

        # 점수 내림차순 정렬
        scored.sort(key=lambda x: x[0], reverse=True)
        logger.info(
            "ADR-080: TimeWeighted re-scored %d results (type=%s, decay=%.3f)",
            len(scored),
            interview_type,
            decay_rate,
        )
        return [r for _, r in scored]

    async def _retrieve_trend_data(self, query: str, n_results: int = 5) -> str:
        """ADR-078/079: trend_data 컬렉션 전용 검색.

        SelfQueryRetriever (ADR-079) 우선, 실패 시 MultiVectorRetriever (ADR-078),
        둘 다 비활성화면 기본 의미 검색.
        """
        settings = get_settings()

        # ADR-079: SelfQueryRetriever — 메타데이터 필터 자동 추출
        if settings.rag_use_self_query and self.vectordb:
            try:
                results = await self._search_with_self_query(query, n_results)
                if results:
                    return results
            except Exception as e:
                logger.warning("ADR-079: SelfQuery failed, falling back: %s", e)

        # ADR-078: MultiVectorRetriever — 요약/가설 쿼리 기반 검색
        if settings.rag_use_multi_vector and self.vectordb:
            try:
                results = await self._search_with_multi_vector(query, n_results)
                if results:
                    return results
            except Exception as e:
                logger.warning("ADR-078: MultiVector failed, falling back: %s", e)

        # Fallback: 기본 의미 검색
        try:
            trend_results = await self.vectordb.query(
                query_text=query,
                collection_type="trend_data",
                n_results=n_results,
            )
            if trend_results:
                return "\n\n".join(r["text"] for r in trend_results)
        except Exception as e:
            logger.warning("trend_data fallback query failed: %s", e)

        return ""

    async def _search_with_self_query(self, query: str, n_results: int = 5) -> str:
        """ADR-079: SelfQueryRetriever로 메타데이터 필터 자동 추출 검색."""
        from langchain.chains.query_constructor.schema import AttributeInfo
        from langchain.retrievers.self_query.base import SelfQueryRetriever

        lc_chroma = self.vectordb.get_langchain_chroma("trend_data")
        metadata_field_info = [
            AttributeInfo(name=f["name"], description=f["description"], type=f["type"])
            for f in TREND_METADATA_FIELDS
        ]

        retriever = SelfQueryRetriever.from_llm(
            llm=self._langchain_gateway.llm if self._langchain_gateway else None,
            vectorstore=lc_chroma,
            document_contents="채용 트렌드, 면접 질문, 기술 동향 관련 데이터",
            metadata_field_info=metadata_field_info,
            enable_limit=True,
        )

        docs = await asyncio.to_thread(retriever.invoke, query)
        docs = docs[:n_results]
        if docs:
            logger.info("ADR-079: SelfQuery returned %d docs", len(docs))
            return "\n\n".join(doc.page_content for doc in docs)
        return ""

    async def _search_with_multi_vector(self, query: str, n_results: int = 5) -> str:
        """ADR-078: MultiVectorRetriever로 요약/가설 쿼리 기반 검색.

        주의: MultiVectorRetriever는 문서 적재 시 요약/가설 쿼리를 사전 생성해야 함.
        이 메서드는 이미 적재된 데이터를 검색하는 용도.
        """
        from langchain.retrievers.multi_vector import MultiVectorRetriever
        from langchain.storage import InMemoryStore

        lc_chroma = self.vectordb.get_langchain_chroma("trend_data")
        docstore = InMemoryStore()

        retriever = MultiVectorRetriever(
            vectorstore=lc_chroma,
            docstore=docstore,
            id_key="doc_id",
        )

        docs = await asyncio.to_thread(retriever.invoke, query)
        docs = docs[:n_results]
        if docs:
            logger.info("ADR-078: MultiVector returned %d docs", len(docs))
            return "\n\n".join(doc.page_content for doc in docs)
        return ""

    async def _search_with_parent_retriever(
        self,
        query: str,
        user_id: str,
        collection_type: str,
        n_results: int | None = None,
    ) -> list[tuple[str, Document]]:
        """ADR-076: 자식 컬렉션(400자)으로 정밀 검색 → 부모 컬렉션(2000자) 반환.

        1. resumes_child / portfolios_child 에서 400자 단위 정밀 검색
        2. 결과 metadata 의 parent_chunk_id 추출
        3. 부모 컬렉션에서 2000자 원본 청크를 ID 로 직접 fetch
        4. (collection_type, Document) 쌍으로 반환

        Args:
            n_results: 자식 컬렉션 검색 수. None이면 settings.rag_retrieval_k * 4 사용. ADR-076.
        """
        _settings = get_settings()
        effective_n = n_results if n_results is not None else _settings.rag_retrieval_k * 4

        child_collection = (
            self.vectordb.RESUME_CHILD_COLLECTION
            if collection_type in ("resumes", "resume")
            else self.vectordb.PORTFOLIO_CHILD_COLLECTION
        )
        try:
            where = {"user_id": user_id} if user_id else None
            child_results = await self.vectordb.query(
                query_text=query,
                collection_type=child_collection,
                n_results=effective_n,
                where=where,
            )
            if not child_results:
                logger.info(
                    "ADR-076: No child results for %s — skip parent retrieval", collection_type
                )
                return []

            # 중복 제거하여 고유 parent_chunk_id 추출
            parent_ids = list(
                {
                    r["metadata"].get("parent_chunk_id")
                    for r in child_results
                    if r["metadata"].get("parent_chunk_id")
                }
            )
            if not parent_ids:
                logger.warning(
                    "ADR-076: child results found but no parent_chunk_id in metadata (%s)",
                    collection_type,
                )
                return []

            parent_docs = await self.vectordb.get_documents_by_ids(
                ids=parent_ids,
                collection_type=collection_type,
            )
            results = [
                (collection_type, Document(page_content=d["text"], metadata=d["metadata"]))
                for d in parent_docs
            ]
            logger.info(
                "ADR-076: ParentRetriever — %d child hits → %d parent docs (%s)",
                len(child_results),
                len(results),
                collection_type,
            )
            return results
        except Exception as e:
            logger.warning(
                "ADR-076: _search_with_parent_retriever failed (%s): %s", collection_type, e
            )
            return []

    async def ingest_trend_documents(
        self,
        documents: list[dict],
    ) -> int:
        """ADR-078: 트렌드 문서를 요약/가설 쿼리로 변환하여 MultiVector 구조로 적재.

        Args:
            documents: [{"id": str, "text": str, "metadata": dict}, ...]

        Returns:
            성공적으로 적재된 문서 수
        """
        import uuid

        from langchain_core.output_parsers import StrOutputParser
        from langchain_core.prompts import ChatPromptTemplate

        settings = get_settings()
        if not settings.rag_use_multi_vector or not self.vectordb:
            logger.warning("ADR-078: MultiVector 미활성화 또는 vectordb 미설정")
            return 0

        llm = self._langchain_gateway.llm if self._langchain_gateway else None
        if not llm:
            logger.warning("ADR-078: LLM 미설정, 적재 불가")
            return 0

        strategy = settings.rag_multi_vector_strategy
        count = 0

        for doc in documents:
            try:
                doc_id = doc.get("id") or str(uuid.uuid4())
                text = doc["text"]
                metadata = doc.get("metadata", {})

                if strategy == "summary":
                    # 요약 임베딩 생성
                    prompt = ChatPromptTemplate.from_template(_SUMMARY_FOR_EMBEDDING_TEMPLATE)
                    chain = prompt | llm | StrOutputParser()
                    summary = await chain.ainvoke({"doc": text[:5000]})

                    # 요약을 trend_data 컬렉션에 저장 (doc_id 매핑)
                    summary_meta = {**metadata, "doc_id": doc_id, "representation": "summary"}
                    await self.vectordb.add_document(
                        document_id=f"summary_{doc_id}",
                        text=summary,
                        collection_type="trend_data",
                        metadata=summary_meta,
                    )

                elif strategy == "hypothetical":
                    # 가설 쿼리 생성
                    query_count = settings.rag_hypothetical_query_count
                    prompt = ChatPromptTemplate.from_template(_HYPOTHETICAL_QUERIES_TEMPLATE)
                    chain = prompt | llm | StrOutputParser()
                    queries_text = await chain.ainvoke(
                        {
                            "doc": text[:5000],
                            "count": query_count,
                        }
                    )

                    # 각 가설 쿼리를 개별 문서로 저장
                    queries = [q.strip() for q in queries_text.strip().split("\n") if q.strip()]
                    for idx, q in enumerate(queries[:query_count]):
                        q_meta = {**metadata, "doc_id": doc_id, "representation": "hypothetical"}
                        await self.vectordb.add_document(
                            document_id=f"hyp_{doc_id}_{idx}",
                            text=q,
                            collection_type="trend_data",
                            metadata=q_meta,
                        )

                # 원본 문서도 저장 (검색 결과 반환용)
                orig_meta = {**metadata, "representation": "original"}
                await self.vectordb.add_document(
                    document_id=doc_id,
                    text=text,
                    collection_type="trend_data",
                    metadata=orig_meta,
                )
                count += 1
                logger.info("ADR-078: 문서 적재 완료 (strategy=%s, doc_id=%s)", strategy, doc_id)

            except Exception as e:
                logger.error("ADR-078: 문서 적재 실패: %s", e)
                continue

        logger.info(
            "ADR-078: 총 %d/%d 문서 적재 완료 (strategy=%s)", count, len(documents), strategy
        )
        return count

    async def retrieve_context(
        self,
        query: str,
        user_id: str,
        context_types: list[str] = None,
        n_results: int = 3,
        *,
        dense_weight: float | None = None,
        sparse_weight: float | None = None,
        chat_mode: str | None = None,
    ) -> str:
        """
        Retrieve relevant context: question → retriever (dense+sparse) → RRF merge → rerank → [compressor] → context.
        When rag_use_bm25 is True, uses dense (MMR) + sparse (BM25) ensemble with RRF (ADR-069).
        dense_weight/sparse_weight override settings at invoke time (ConfigurableField-style A/B test).

        ADR-077: When rag_use_multi_query is True and chat_mode is NORMAL, expands query to
        multiple perspective queries and merges results with deduplication.
        """
        if context_types is None:
            context_types = ["resume", "job_posting"]

        settings = get_settings()
        has_vectorstore = bool(
            self._rag_chain and (self._rag_chain._vectorstores or self._rag_chain._vectorstore)
        )

        # ADR-077: 다중 쿼리 확장 (NORMAL 모드에서만 활성화)
        use_multi_query = (
            settings.rag_use_multi_query and chat_mode and chat_mode.lower() == "normal"
        )
        if use_multi_query:
            queries = await self._generate_multi_queries(query)
        else:
            queries = [query]

        # ADR-078/079: trend_data 컬렉션 검색 경로 분기
        if "trend_data" in context_types and self.vectordb:
            trend_context = await self._retrieve_trend_data(query, n_results)
            if trend_context:
                # trend_data 외 다른 context_types도 있으면 함께 검색
                remaining_types = [t for t in context_types if t != "trend_data"]
                if not remaining_types:
                    return trend_context
                # 나머지 컨텍스트도 검색해서 병합
                context_types = remaining_types
                # trend_context는 나중에 병합
            else:
                trend_context = ""
                context_types = [t for t in context_types if t != "trend_data"]
                if not context_types:
                    return ""
        else:
            trend_context = ""

        # ADR-069: Dense (MMR) + Sparse (BM25) ensemble with RRF → rerank → optional compressor
        if settings.rag_use_bm25 and has_vectorstore and self.vectordb:
            try:
                types_to_fetch = [ct for ct in context_types if ct != "portfolio"]
                if not types_to_fetch:
                    return ""

                # ADR-077: 각 쿼리별 dense 검색 후 병합
                # ADR-076: resumes/portfolios는 ParentDocumentRetriever 경로로 분기
                all_dense_pairs: list[tuple[str, Document]] = []
                if settings.rag_use_parent_retriever and self.vectordb:
                    _parent_cts = [ct for ct in types_to_fetch if ct in _PARENT_RETRIEVER_TYPES]
                    _other_cts = [ct for ct in types_to_fetch if ct not in _PARENT_RETRIEVER_TYPES]
                    for mq in queries:
                        for ct in _parent_cts:
                            pairs = await self._search_with_parent_retriever(mq, user_id, ct)
                            if not pairs:
                                # 자식 컬렉션 미적재 또는 검색 실패 시 기존 MMR으로 fallback
                                logger.info(
                                    "ADR-076: parent retriever 결과 없음 (%s) — MMR fallback ('%s'...)",
                                    ct,
                                    sanitize_log_input(mq[:40]),
                                )
                                pairs = await self._rag_chain.retrieve_context_documents(
                                    mq, user_id, [ct]
                                )
                            all_dense_pairs.extend(pairs)
                        if _other_cts:
                            pairs = await self._rag_chain.retrieve_context_documents(
                                mq, user_id, _other_cts
                            )
                            all_dense_pairs.extend(pairs)
                else:
                    for mq in queries:
                        pairs = await self._rag_chain.retrieve_context_documents(
                            mq, user_id, types_to_fetch
                        )
                        all_dense_pairs.extend(pairs)

                # 다중 쿼리 결과 중복 제거 (page_content 해시 기준)
                if len(queries) > 1:
                    seen_keys: set[str] = set()
                    deduped: list[tuple[str, Document]] = []
                    for ct, doc in all_dense_pairs:
                        key = (doc.page_content or "")[:500]
                        if key not in seen_keys:
                            seen_keys.add(key)
                            deduped.append((ct, doc))
                    dense_pairs = deduped
                    logger.info(
                        "ADR-077: MultiQuery dense results: %d total → %d after dedup",
                        len(all_dense_pairs),
                        len(dense_pairs),
                    )
                else:
                    dense_pairs = all_dense_pairs

                logger.info("[Ensemble] Dense(MMR): %d건", len(dense_pairs))

                results_per_type = await asyncio.gather(
                    *[
                        self.vectordb.get_all_documents_by_user(user_id=user_id, collection_type=ct)
                        for ct in types_to_fetch
                    ]
                )
                bm25_docs: list[Document] = []
                for ct, doc_list in zip(types_to_fetch, results_per_type, strict=True):
                    for r in doc_list:
                        meta = dict(r.get("metadata") or {})
                        meta["collection_type"] = ct
                        bm25_docs.append(Document(page_content=r.get("text") or "", metadata=meta))

                # BM25 인덱스 상한 (ADR-069 후속: 메모리·지연 완화)
                if settings.rag_bm25_max_docs > 0 and len(bm25_docs) > settings.rag_bm25_max_docs:
                    bm25_docs = bm25_docs[: settings.rag_bm25_max_docs]

                logger.info("[Ensemble] BM25 인덱스: %d건", len(bm25_docs))

                sparse_pairs: list[tuple[str, Document]] = []
                if bm25_docs and query.strip():
                    k_bm25 = max(10, min(50, len(bm25_docs)))
                    loop = asyncio.get_running_loop()

                    def _bm25_invoke() -> list[Document]:
                        retriever = BM25Retriever.from_documents(bm25_docs, k=k_bm25)
                        return retriever.invoke(query)

                    sparse_docs = await loop.run_in_executor(None, _bm25_invoke)
                    sparse_pairs = [(d.metadata.get("collection_type", ""), d) for d in sparse_docs]

                logger.info("[Ensemble] Sparse(BM25): %d건", len(sparse_pairs))

                dw = (
                    dense_weight if dense_weight is not None else settings.rag_ensemble_dense_weight
                )
                sw = (
                    sparse_weight
                    if sparse_weight is not None
                    else settings.rag_ensemble_sparse_weight
                )
                merged = _rrf_merge(dense_pairs, sparse_pairs, dw, sw, top_k=0)
                logger.info(
                    "[Ensemble] RRF 병합: %d건 (dense=%.1f, sparse=%.1f)",
                    len(merged),
                    dw,
                    sw,
                )

                # ADR-106 Phase 4: RAPTOR 다단계 검색 결과 병합
                if self._raptor:
                    for ct in context_types:
                        coll = ct + "s" if not ct.endswith("s") else ct
                        raptor_pairs = await self._raptor.retrieve_multi_level(
                            query=query,
                            collection_name=coll,
                            user_id=user_id,
                            top_k=3,
                        )
                        if raptor_pairs:
                            merged.extend(raptor_pairs)
                            logger.info(
                                "ADR-106: RAPTOR %d개 결과 병합 (collection=%s)",
                                len(raptor_pairs),
                                coll,
                            )

                # 재정렬: top_k 적용 (ADR-069 question → retriever → rerank → context)
                rerank_k = settings.rag_rerank_top_k if settings.rag_rerank_top_k > 0 else 0
                merged = await _rerank(merged, rerank_k, query=query)

                if settings.rag_use_compressor and merged:
                    llm = getattr(getattr(self._rag_chain, "_llm_gateway", None), "llm", None)
                    merged = await _compress_pairs_with_llm(
                        query,
                        merged,
                        settings.rag_compressor_top_k,
                        llm,
                    )

                if not merged:
                    return ""
                return self._rag_chain._format_pairs_to_context(merged)
            except Exception as e:
                logger.warning("RAG ensemble failed, fallback to dense-only: %s", e)
                if has_vectorstore:
                    return await self._rag_chain.retrieve_context(query, user_id, context_types)
                return ""

        if has_vectorstore:
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
        chat_mode: str | None = None,  # ADR-077: 채팅 모드 전달 (NORMAL이면 MultiQuery 활성화)
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
            chat_mode: Chat mode ("normal", "interview", etc.) for ADR-077 multi-query

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
                    chat_mode=chat_mode,
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
                # ADR-106 Phase 2: LCEL 체인 활용
                logger.info("Using Gemini model (LCEL chain)")
                settings = get_settings()
                if history:
                    # history 있는 경우: MessagesPlaceholder 기반 chat chain 유지
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
                    # ADR-106: history 없는 경우 → LCEL 체인
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
                        n_results=5
                        if settings.rag_use_time_weighted
                        else 2,  # ADR-080: 시간 가중 시 더 많이 가져와서 re-score
                        where={"user_id": user_id, "interview_type": interview_type},
                    )
                    # ADR-080: 시간 가중 적용 후 상위 2건만 사용
                    if settings.rag_use_time_weighted and feedback_results:
                        feedback_results = self._apply_time_weighting(
                            feedback_results, interview_type
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
                        n_results=5 if settings.rag_use_time_weighted else 2,  # ADR-080
                        where={"user_id": user_id, "interview_type": interview_type},
                    )
                    # ADR-080: 시간 가중 적용 후 상위 2건만 사용
                    if settings.rag_use_time_weighted and feedback_results:
                        feedback_results = self._apply_time_weighting(
                            feedback_results, interview_type
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
