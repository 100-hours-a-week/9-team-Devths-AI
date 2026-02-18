"""
텍스트 분할 서비스 (Text Splitting / Chunking)

RecursiveCharacterTextSplitter 기반으로 문서를 의미 단위 청크로 분할.
Gemini-embedding-001 토큰 제한(2,048)을 준수하면서
API 명세(500 tokens, 50 overlap)에 맞춤.

ADR-060 참조
"""

import logging
from dataclasses import dataclass, field

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


class TextSplitterService:
    """문서 텍스트를 VectorDB 저장을 위한 청크로 분할

    - 짧은 텍스트(chunk_size 이하)는 분할하지 않음
    - 각 청크에 parent_document_id, chunk_index, total_chunks 메타데이터 부여
    - add_documents_batch()와 호환되는 dict 리스트 반환 메서드 제공
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        settings = get_settings()
        self._chunk_size = chunk_size or settings.chunk_size
        self._chunk_overlap = chunk_overlap or settings.chunk_overlap

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

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

        chunks = self._splitter.split_text(text)

        result = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i:03d}"
            chunk_metadata = {
                **base_metadata,
                "parent_document_id": document_id,
                "chunk_index": i,
                "total_chunks": len(chunks),
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
            "📄 텍스트 분할 완료: %d자 → %d개 청크 (평균 %d자/청크)",
            len(text),
            len(result),
            avg_len,
        )
        return result

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
