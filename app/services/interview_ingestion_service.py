"""
면접 결과 VectorDB Ingestion Service.

모의면접 Q&A + 평가 결과를 구조화된 텍스트로 변환하고
VectorDB interview_feedback 컬렉션에 batch 저장합니다.
"""

import logging
from typing import Any

from app.schemas.ingestion import InterviewIngestionInput, InterviewResultDocument
from app.services.vectordb_service import VectorDBService

logger = logging.getLogger(__name__)

COLLECTION_NAME = "interview_feedback"


class InterviewIngestionService:
    """면접 결과를 VectorDB에 적재하는 서비스."""

    def __init__(self, vectordb_service: VectorDBService):
        self.vectordb = vectordb_service

    @staticmethod
    def _format_document(result: InterviewResultDocument) -> str:
        """Q&A + 평가를 단일 텍스트 문서로 변환.

        Args:
            result: 개별 Q&A 평가 결과.

        Returns:
            VectorDB에 저장될 텍스트 문서.
        """
        lines = [
            f"[면접 질문] {result.question}",
            f"[지원자 답변] {result.answer}",
        ]

        if result.score is not None:
            lines.append(f"[점수] {result.score}/5")
        if result.verdict:
            lines.append(f"[판정] {result.verdict}")
        if result.reasoning:
            lines.append(f"[평가 근거] {result.reasoning}")
        if result.recommended_answer:
            lines.append(f"[추천 답변] {result.recommended_answer}")

        return "\n".join(lines)

    @staticmethod
    def _make_id(user_id: str | int, session_id: str | int, question_index: int) -> str:
        """중복 방지를 위한 문서 ID 생성.

        동일 세션의 동일 질문을 다시 적재하면 upsert(덮어쓰기) 됩니다.
        """
        return f"user_{user_id}_session_{session_id}_q{question_index}"

    async def ingest_session(self, input_data: InterviewIngestionInput) -> dict[str, Any]:
        """면접 세션 전체 결과를 VectorDB에 batch 저장.

        Args:
            input_data: 면접 세션 결과 입력.

        Returns:
            저장 결과 (저장된 ID 목록, 개수 등).
        """
        texts: list[str] = []
        metadatas: list[dict[str, Any]] = []
        ids: list[str] = []

        for result in input_data.qa_results:
            text = self._format_document(result)
            doc_id = self._make_id(input_data.user_id, input_data.session_id, result.question_index)

            metadata: dict[str, Any] = {
                "user_id": str(input_data.user_id),
                "session_id": str(input_data.session_id),
                "room_id": str(input_data.room_id),
                "interview_type": input_data.interview_type,
                "question_index": result.question_index,
                "question_only": result.question,
                "answer_only": result.answer,
                "source": "user_interview",  # 외부 데이터셋과 구분
            }

            if result.score is not None:
                metadata["score"] = result.score
            if result.verdict:
                metadata["verdict"] = result.verdict

            texts.append(text)
            metadatas.append(metadata)
            ids.append(doc_id)

        if not texts:
            logger.warning("Ingestion 대상 Q&A가 없습니다 (session=%s)", input_data.session_id)
            return {"ingested_count": 0, "ids": []}

        added_ids = await self.vectordb.add_texts(
            texts=texts,
            metadatas=metadatas,
            ids=ids,
            user_id=input_data.user_id,
            collection_name=COLLECTION_NAME,
        )

        logger.info(
            "✅ 면접 결과 %d건 VectorDB 적재 완료 (session=%s, user=%s)",
            len(added_ids),
            input_data.session_id,
            input_data.user_id,
        )

        return {
            "ingested_count": len(added_ids),
            "ids": added_ids,
            "session_id": str(input_data.session_id),
            "collection": COLLECTION_NAME,
        }
