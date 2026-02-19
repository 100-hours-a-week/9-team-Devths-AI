"""
Interview Question Deduplication Service (ADR-066)

같은 채팅 세션 내에서 완벽히 답변한 질문과 유사한 새 질문을 필터링한다.
임베딩 cosine similarity 기반으로 의미적 유사도를 비교한다.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 유사도 임계값: 이 값 이상이면 "유사 질문"으로 판단하여 스킵
SIMILARITY_THRESHOLD = 0.85

# answer_quality 등급 중 "마스터"로 간주하는 등급
MASTERED_QUALITIES = {"excellent", "good"}


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """두 벡터의 코사인 유사도 계산"""
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


def is_mastered_quality(answer_quality: str) -> bool:
    """answer_quality가 마스터(스킵 대상) 등급인지 판단"""
    return answer_quality.lower() in MASTERED_QUALITIES


async def filter_similar_questions(
    new_questions: list[dict],
    mastered_questions: list[dict],
    vectordb_service,
    threshold: float = SIMILARITY_THRESHOLD,
) -> list[dict]:
    """마스터한 질문과 유사한 새 질문을 필터링

    Args:
        new_questions: 출제 예정인 질문 리스트 [{"question": str, ...}]
        mastered_questions: 마스터한 질문 [{"question": str, "embedding": list}]
        vectordb_service: VectorDBService 인스턴스 (임베딩 생성용)
        threshold: 유사도 임계값 (기본 0.85)

    Returns:
        필터링된 질문 리스트 (유사 질문 제거됨)
    """
    if not mastered_questions:
        return new_questions

    mastered_embeddings = [q["embedding"] for q in mastered_questions]
    filtered = []

    for q in new_questions:
        try:
            q_embedding = await vectordb_service.create_embedding(q["question"])
            max_sim = max(cosine_similarity(q_embedding, m_emb) for m_emb in mastered_embeddings)
            if max_sim < threshold:
                filtered.append(q)
            else:
                logger.info(
                    "ADR-066: 질문 스킵 (유사도 %.2f): %s",
                    max_sim,
                    q["question"][:50],
                )
        except Exception as e:
            logger.warning("ADR-066: 유사도 비교 실패, 질문 유지: %s", e)
            filtered.append(q)

    skipped = len(new_questions) - len(filtered)
    if skipped > 0:
        logger.info(
            "ADR-066: %d개 유사 질문 스킵됨 (총 %d → %d)",
            skipped,
            len(new_questions),
            len(filtered),
        )

    return filtered
