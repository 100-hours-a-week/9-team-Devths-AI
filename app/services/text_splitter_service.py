"""
텍스트 분할 서비스 (Text Splitting / Chunking)

RecursiveCharacterTextSplitter 기반으로 문서를 의미 단위 청크로 분할.
Gemini-embedding-001 토큰 제한(2,048)을 준수하면서
API 명세(500 tokens, 50 overlap)에 맞춤.

ADR-060 참조
ADR-106 Phase 3: SemanticChunker 통합 (임베딩 유사도 기반 의미 분할)
"""

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.config.settings import get_settings

logger = logging.getLogger(__name__)

# Gemini-embedding-001 안전 한계 (2,048 토큰 ≈ 8,000자)
MAX_EMBEDDING_CHARS = 8000


@dataclass
class TextChunk:
    """분할된 텍스트 청크"""

    id: str
    text: str
    metadata: dict = field(default_factory=dict)


# ADR-106: SemanticChunker 싱글톤 (임베딩 모델 로딩 1회만)
_semantic_chunker: Any = None


def _get_semantic_chunker() -> Any:
    """ADR-106: SemanticChunker 싱글톤 반환. 최초 호출 시 임베딩 모델 로딩."""
    global _semantic_chunker  # noqa: PLW0603
    if _semantic_chunker is None:
        from langchain_experimental.text_splitter import SemanticChunker
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        settings = get_settings()
        embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.gemini_embedding_model,
            google_api_key=settings.google_api_key,
        )
        _semantic_chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type=settings.semantic_chunker_threshold_type,
            breakpoint_threshold_amount=settings.semantic_chunker_threshold_amount,
        )
        logger.info(
            "ADR-106: SemanticChunker initialized (type=%s, amount=%.1f)",
            settings.semantic_chunker_threshold_type,
            settings.semantic_chunker_threshold_amount,
        )
    return _semantic_chunker


class TextSplitterService:
    """문서 텍스트를 VectorDB 저장을 위한 청크로 분할

    - 짧은 텍스트(chunk_size 이하)는 분할하지 않음
    - 각 청크에 parent_document_id, chunk_index, total_chunks 메타데이터 부여
    - add_documents_batch()와 호환되는 dict 리스트 반환 메서드 제공

    ADR-106: rag_use_semantic_chunker=True 시 SemanticChunker 사용.
    SemanticChunker는 임베딩 유사도 기반으로 의미가 바뀌는 지점에서 분할.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        use_semantic: bool | None = None,
    ):
        settings = get_settings()
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap
        self._use_semantic = (
            use_semantic if use_semantic is not None else settings.rag_use_semantic_chunker
        )

        # 고정 크기 분할기 (기본 / SemanticChunker 후처리용)
        self._fixed_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        if self._use_semantic:
            logger.info("ADR-106: TextSplitterService using SemanticChunker mode")

    def _split_with_semantic(self, text: str) -> list[str]:
        """ADR-106: SemanticChunker로 의미 단위 분할 + 최대 크기 후처리.

        SemanticChunker가 생성한 청크가 MAX_EMBEDDING_CHARS를 초과하면
        고정 크기 분할기로 재분할하여 임베딩 제한을 준수.
        실패 시 고정 크기 분할기로 폴백.
        """
        try:
            chunker = _get_semantic_chunker()
            raw_chunks = chunker.split_text(text)

            # 후처리: 너무 큰 청크는 고정 크기로 재분할
            final_chunks: list[str] = []
            for chunk in raw_chunks:
                if len(chunk) > MAX_EMBEDDING_CHARS:
                    sub_chunks = self._fixed_splitter.split_text(chunk)
                    final_chunks.extend(sub_chunks)
                    logger.debug(
                        "ADR-106: 과대 청크 재분할 (%d자 → %d개)", len(chunk), len(sub_chunks)
                    )
                else:
                    final_chunks.append(chunk)

            logger.info(
                "ADR-106: SemanticChunker 분할 완료 — %d자 → %d개 청크",
                len(text),
                len(final_chunks),
            )
            return final_chunks
        except Exception as e:
            logger.warning("ADR-106: SemanticChunker 실패, 고정 크기 폴백: %s", e)
            return self._fixed_splitter.split_text(text)

    def split_text(
        self,
        text: str,
        document_id: str,
        metadata: dict | None = None,
    ) -> list[TextChunk]:
        """텍스트를 청크로 분할

        Args:
            text: 전체 텍스트
            document_id: 원본 문서 ID (부모 ID)
            metadata: 추가 메타데이터

        Returns:
            TextChunk 리스트
        """
        if not text or not text.strip():
            return []

        base_metadata = metadata or {}

        # 텍스트가 단일 청크 이내면 분할 불필요
        if len(text) <= self._chunk_size:
            return [
                TextChunk(
                    id=document_id,
                    text=text,
                    metadata={
                        **base_metadata,
                        "parent_document_id": document_id,
                        "chunk_index": 0,
                        "total_chunks": 1,
                    },
                )
            ]

        # ADR-106: SemanticChunker 또는 고정 크기 분할
        if self._use_semantic:
            chunks = self._split_with_semantic(text)
        else:
            chunks = self._fixed_splitter.split_text(text)

        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i:03d}"
            chunk_metadata = {
                **base_metadata,
                "parent_document_id": document_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "split_method": "semantic" if self._use_semantic else "fixed",
            }
            result.append(
                TextChunk(
                    id=chunk_id,
                    text=chunk_text,
                    metadata=chunk_metadata,
                )
            )

        avg_len = len(text) // len(result) if result else 0
        logger.info(
            "📄 텍스트 분할 완료: %d자 → %d개 청크 (평균 %d자/청크, method=%s)",
            len(text),
            len(result),
            avg_len,
            "semantic" if self._use_semantic else "fixed",
        )
        return result

    def split_parent_chunk_to_children(self, parent_chunk: TextChunk) -> list[TextChunk]:
        """ADR-076: 부모 청크 → 자식 청크들 (정밀 검색용).

        부모 청크를 self._chunk_size(기본 400자) 단위로 재분할한다.
        자식 청크 메타데이터에 parent_chunk_id를 포함하여 역방향 조회 가능.

        Note: 자식 청크 분할은 항상 고정 크기 분할기 사용 (정밀 검색 일관성).

        Args:
            parent_chunk: 원본 2000자 부모 청크

        Returns:
            400자 자식 TextChunk 리스트
        """
        if not parent_chunk.text or not parent_chunk.text.strip():
            return []

        raw_splits = self._fixed_splitter.split_text(parent_chunk.text)
        children: list[TextChunk] = []
        for i, child_text in enumerate(raw_splits):
            child_id = f"{parent_chunk.id}_child_{i:03d}"
            children.append(
                TextChunk(
                    id=child_id,
                    text=child_text,
                    metadata={
                        **parent_chunk.metadata,
                        "parent_chunk_id": parent_chunk.id,
                        "child_index": i,
                        "is_child": True,
                    },
                )
            )

        logger.debug(
            "ADR-076: 부모 청크 '%s' → %d개 자식 청크 (chunk_size=%d)",
            parent_chunk.id,
            len(children),
            self._chunk_size,
        )
        return children

    def split_to_batch_docs(
        self,
        text: str,
        document_id: str,
        metadata: dict | None = None,
    ) -> list[dict]:
        """텍스트를 분할하고 add_documents_batch() 호환 dict 리스트로 반환

        Args:
            text: 전체 텍스트
            document_id: 원본 문서 ID
            metadata: 추가 메타데이터

        Returns:
            [{"id": str, "text": str, "metadata": dict}, ...]
        """
        chunks = self.split_text(text, document_id, metadata)
        return [
            {"id": chunk.id, "text": chunk.text, "metadata": chunk.metadata} for chunk in chunks
        ]
