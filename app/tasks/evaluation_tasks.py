"""
Evaluation Tasks — ADR-102.

Celery 태스크: 면접 Q&A VectorDB 적재.
asyncio.create_task() 대신 Celery Worker에서 실행하여 504 타임아웃 해결.
"""

import asyncio
import logging

from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.tasks.evaluation_tasks.ingest_interview_qa_task")
def ingest_interview_qa_task(
    session_id: str,
    user_id: str,
    room_id: str,
    context: list[dict],
):
    """면접 Q&A VectorDB 적재 Celery 태스크 (ADR-102).

    Args:
        session_id: 면접 세션 ID
        user_id: 사용자 ID
        room_id: 채팅방 ID
        context: Q&A 목록 [{"question": ..., "answer": ..., "category": ...}, ...]
    """
    logger.info("[EvaluationTask] 면접 Q&A 적재 시작: session=%s", session_id)

    try:
        result = asyncio.run(
            _ingest_interview_qa_async(
                session_id=session_id,
                user_id=user_id,
                room_id=room_id,
                context=context,
            )
        )
        logger.info("[EvaluationTask] 면접 Q&A 적재 완료: session=%s", session_id)
        return result

    except Exception as e:
        logger.warning("[EvaluationTask] 면접 Q&A 적재 실패 (무시): %s", e)
        return {"status": "failed", "error": str(e)}


async def _ingest_interview_qa_async(
    session_id: str,
    user_id: str,
    room_id: str,
    context: list[dict],
) -> dict:
    """비동기 면접 Q&A 적재 로직 (기존 _ingest_interview_qa_best_effort 로직 이관)."""
    from app.api.routes.v2._helpers import get_services
    from app.schemas.ingestion import InterviewIngestionInput, InterviewResultDocument
    from app.services.interview_ingestion_service import InterviewIngestionService

    try:
        rag = get_services()
        vectordb = rag.vectordb
        ingestion_service = InterviewIngestionService(vectordb)

        qa_results = []
        for i, qa in enumerate(context or []):
            qa_results.append(
                InterviewResultDocument(
                    question=qa.get("question", ""),
                    answer=qa.get("answer", ""),
                    question_index=i,
                    interview_type=qa.get("category", "technical"),
                )
            )

        if not qa_results:
            return {"status": "skipped", "reason": "no_qa_data"}

        input_data = InterviewIngestionInput(
            session_id=session_id,
            user_id=user_id,
            room_id=room_id,
            interview_type="technical",
            qa_results=qa_results,
        )

        result = await ingestion_service.ingest_session(input_data)
        logger.info("📦 면접 Q&A VectorDB 자동 저장 완료: %d건", result["ingested_count"])

        return {
            "status": "completed",
            "session_id": session_id,
            "ingested_count": result["ingested_count"],
        }

    except Exception as e:
        logger.warning("면접 Q&A VectorDB 자동 저장 실패 (무시): %s", str(e), exc_info=True)
        raise
