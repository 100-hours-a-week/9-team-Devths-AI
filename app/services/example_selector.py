"""
SemanticSimilarity 기반 예제 선택 (평시 질의 응답, 인성 면접용).

- 평시: 정적 예제 풀에서 사용자 질문과 유사한 예제 k개 선택해 프롬프트에 주입.
- 인성: interview_feedback 컬렉션에서 유사 Q&A k개 조회해 프롬프트에 주입.
"""

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# 평시 질의 응답용 정적 예제 (유사도 선택 대상)
GENERAL_CHAT_EXAMPLES = [
    {
        "input": "자기소개서 첨삭 부탁해도 될까요?",
        "output": "네, 물론이에요. 먼저 작성하신 자기소개서를 붙여 주시거나 업로드해 주시면, 지원 직무와 맞춰서 강점을 살리는 방향으로 첨삭해 드릴게요. 어떤 회사 어떤 직무로 지원 예정이신가요?",
    },
    {
        "input": "면접에서 떨리는데 어떻게 하면 좋을까요?",
        "output": "면접 전에는 회사와 직무를 미리 파악하고, 자주 나올 질문에 대한 답을 한두 문장으로 정리해 두는 게 도움이 됩니다. 당일에는 심호흡을 하고, 질문을 끝까지 듣고 1~2초 생각한 뒤 답하시면 됩니다. 연습이 필요하시면 여기서 모의 면접도 진행해 드릴 수 있어요.",
    },
    {
        "input": "이력서 피드백 받을 수 있을까요?",
        "output": "네, 가능해요. 이력서 내용을 붙여 주시거나 업로드해 주시면, 지원하시는 직무에 맞춰 경력 정리나 표현을 다듬어 드릴게요. 지원 예정인 직무가 있으면 알려 주세요.",
    },
    {
        "input": "취업 준비 어떻게 하면 좋을까요?",
        "output": "이력서와 자기소개서를 먼저 정리한 뒤, 지원 직무별로 예상 질문을 뽑아 답을 준비하는 걸 추천해요. 여기서는 이력서·자소서 첨삭, 면접 예상 질문 생성, 모의 면접까지 도와드릴 수 있어요. 어떤 것부터 도와드릴까요?",
    },
    {
        "input": "채용공고 보면서 준비하고 싶어요.",
        "output": "채용공고를 등록하시거나 붙여 주시면, 그 공고에 맞춰 이력서 포인트나 예상 면접 질문을 뽑아 드릴 수 있어요. 공고와 이력서를 함께 주시면 매칭 관점으로도 조언해 드려요.",
    },
]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


async def get_few_shot_for_general(
    vectordb: Any,
    user_message: str,
    k: int = 2,
) -> str:
    """평시 질의 응답용: 사용자 질문과 유사한 예제 k개를 선택해 문자열로 반환."""
    if k <= 0 or not GENERAL_CHAT_EXAMPLES:
        return ""
    try:
        inputs = [ex["input"] for ex in GENERAL_CHAT_EXAMPLES]
        embeddings = await vectordb.create_embeddings([user_message] + inputs)
        if not embeddings or len(embeddings) < 2:
            return ""
        query_emb = embeddings[0]
        sims = [_cosine_similarity(query_emb, embeddings[i + 1]) for i in range(len(inputs))]
        indexed = list(zip(sims, range(len(sims)), strict=False))
        indexed.sort(key=lambda x: -x[0])
        top_indices = [idx for _, idx in indexed[:k]]
        lines = []
        for idx in top_indices:
            ex = GENERAL_CHAT_EXAMPLES[idx]
            lines.append(f"사용자: {ex['input']}")
            lines.append(f"어시스턴트: {ex['output']}")
        return "\n\n".join(lines) if lines else ""
    except Exception as e:
        logger.warning("Few-shot general selection failed: %s", e)
        return ""


async def get_few_shot_for_personality(
    vectordb: Any,
    query_text: str,
    k: int = 2,
    interview_type_filter: str | None = "personality",
) -> str:
    """인성 면접용: interview_feedback에서 유사 Q&A k개 조회해 문자열로 반환."""
    if k <= 0:
        return ""
    try:
        where = None
        if interview_type_filter:
            where = {"interview_type": interview_type_filter}
        results = await vectordb.query(
            query_text=query_text[:500],
            collection_type="interview_feedback",
            n_results=k,
            where=where,
        )
        if not results:
            results = await vectordb.query(
                query_text=query_text[:500],
                collection_type="interview_feedback",
                n_results=k,
            )
        lines = []
        for r in results:
            text = r.get("text") or ""
            meta = r.get("metadata") or {}
            q = (
                meta.get("question_only") or text.split("질문:")[-1].split("답변:")[0].strip()
                if "질문:" in text
                else ""
            )
            a = (
                meta.get("answer_only") or text.split("답변:")[-1].strip()
                if "답변:" in text
                else ""
            )
            if q or a:
                lines.append(f"질문: {q}")
                lines.append(f"답변: {a}")
        return "\n\n".join(lines) if lines else ""
    except Exception as e:
        logger.warning("Few-shot personality selection failed: %s", e)
        return ""
