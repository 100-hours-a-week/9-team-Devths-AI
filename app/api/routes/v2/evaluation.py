"""
면접 답변 평가 API 엔드포인트 (통합).

POST /ai/evaluation/analyze - 면접 평가 리포트 생성 (SSE 스트리밍)
  - retry=false: Gemini 단독 분석
  - retry=true:  Gemini×GPT-4o 토론 후 최종 리포트
"""

import json
import logging
import re

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from app.api.routes.v2._helpers import get_services
from app.api.routes.v2._sse_errors import sse_error_event
from app.config.dependencies import (
    get_debate_service,
    get_evaluation_analyzer,
)
from app.domain.evaluation.analyzer import InterviewAnalyzer
from app.domain.evaluation.debate_graph import DebateService
from app.schemas.evaluation import (
    AnalyzeInterviewRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation")


def _sanitize_log_value(value: str, max_length: int = 50) -> str:
    """Log Injection 방지를 위해 로그 값을 정제합니다."""
    sanitized = re.sub(r"[\r\n\t]", "", str(value))
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."
    return sanitized


# ============================================
# Gemini 단독 분석 스트리밍 (retry=false)
# ============================================


async def _generate_analyze_stream(request: AnalyzeInterviewRequest):
    """면접 평가 리포트 SSE 스트리밍 생성 (Gemini 단독)."""
    sse_end = "\n\n"
    qa_list = request.context or []

    if not qa_list:
        yield sse_error_event(
            code="EMPTY_CONTEXT",
            status=400,
            message="평가할 Q&A 데이터가 없습니다.",
            fallback="면접 문답 데이터가 비어 있습니다. context에 Q&A 목록을 전달해주세요.",
        )
        yield f"data: [DONE]{sse_end}"
        return

    rag = get_services()

    content = "면접 평가 리포트를 생성 중입니다...\n"
    yield f"data: {json.dumps({'chunk': content}, ensure_ascii=False)}{sse_end}"

    # Q&A 텍스트 구성
    qa_text = ""
    for i, qa in enumerate(qa_list, 1):
        q = qa.get("question", "")
        a = qa.get("answer", "")
        qa_text += f"질문 {i}: {q}\n답변 {i}: {a}\n\n"

    report_prompt = f"""
다음은 면접 Q&A 기록입니다:

{qa_text}

위 면접 내용을 바탕으로 상세한 평가 리포트를 작성해주세요.
다음 항목을 포함해주세요:
1. 각 답변에 대한 개별 평가 (잘한 점, 개선점)
2. 전체적인 강점 패턴
3. 전체적인 약점 패턴
4. 향후 학습 가이드
"""

    full_report = ""
    model_choice = (
        request.model.value if hasattr(request.model, "value") else str(request.model)
    )

    try:
        if model_choice == "vllm" and rag.vllm:
            logger.info("📊 [vLLM] 면접 평가 리포트 생성 시작")
            async for chunk in rag.vllm.generate_response(
                user_message=report_prompt,
                context=None,
                history=[],
                system_prompt="당신은 면접 평가 전문가입니다. 상세하고 건설적인 피드백을 제공합니다.",
            ):
                full_report += chunk
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"
        else:
            logger.info("📊 [Gemini] 면접 평가 리포트 생성 시작")
            async for chunk in rag.llm.generate_response(
                user_message=report_prompt,
                context=None,
                history=[],
                system_prompt="당신은 면접 평가 전문가입니다. 상세하고 건설적인 피드백을 제공합니다.",
                user_id=request.user_id,
            ):
                full_report += chunk
                yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

        logger.info("✅ 면접 평가 리포트 생성 완료 (응답 길이: %d자)", len(full_report))

    except Exception as e:
        logger.error("면접 평가 리포트 생성 오류: %s", str(e), exc_info=True)
        yield sse_error_event(
            code="LLM_ERROR",
            status=500,
            message=str(e),
            fallback="면접 리포트 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )

    yield f"data: [DONE]{sse_end}"


# ============================================
# Gemini×GPT-4o 토론 스트리밍 (retry=true)
# ============================================


async def _generate_debate_stream(
    request: AnalyzeInterviewRequest,
    analyzer: InterviewAnalyzer,
    debate_service: DebateService,
):
    """Gemini×GPT-4o 토론 후 최종 리포트 SSE 스트리밍 생성."""
    sse_end = "\n\n"
    qa_list = request.context or []

    if not qa_list:
        yield sse_error_event(
            code="EMPTY_CONTEXT",
            status=400,
            message="평가할 Q&A 데이터가 없습니다.",
            fallback="면접 문답 데이터가 비어 있습니다. context에 Q&A 목록을 전달해주세요.",
        )
        yield f"data: [DONE]{sse_end}"
        return

    # Q&A 변환
    qa_pairs = [
        {
            "question": qa.get("question", ""),
            "answer": qa.get("answer", ""),
            "category": qa.get("category", ""),
        }
        for qa in qa_list
    ]

    try:
        # 1단계: Gemini 구조화 분석
        progress = "🔍 Gemini 분석 중...\n"
        yield f"data: {json.dumps({'chunk': progress}, ensure_ascii=False)}{sse_end}"

        logger.info("📊 [Debate] 1단계: Gemini 구조화 분석 시작")
        gemini_result = await analyzer.analyze(qa_pairs=qa_pairs)
        gemini_dict = gemini_result.to_dict()
        logger.info("✅ [Debate] 1단계 완료: overall_score=%d", gemini_result.overall_score)

        # 2단계: Gemini×GPT-4o 토론
        progress = "🤖 GPT-4o 심층 분석 및 토론 중...\n"
        yield f"data: {json.dumps({'chunk': progress}, ensure_ascii=False)}{sse_end}"

        logger.info("📊 [Debate] 2단계: Gemini×GPT-4o 토론 시작")
        debate_result = await debate_service.run_debate(
            qa_pairs=qa_pairs,
            gemini_analysis=gemini_dict,
        )
        logger.info(
            "✅ [Debate] 2단계 완료: consensus=%s", debate_result.consensus_method
        )

        # 3단계: 최종 결과를 텍스트 리포트로 SSE 스트리밍
        progress = f"\n📋 토론 완료 (합의 방법: {debate_result.consensus_method})\n\n"
        yield f"data: {json.dumps({'chunk': progress}, ensure_ascii=False)}{sse_end}"

        final = debate_result.final_analysis

        # 각 질문별 평가 출력
        for i, q in enumerate(final.questions, 1):
            report_text = f"━━━ 질문 {i} ━━━\n"
            report_text += f"질문: {q.question}\n"
            report_text += f"답변: {q.user_answer}\n"
            report_text += f"판정: {q.verdict} (점수: {q.score}/5)\n"
            report_text += f"평가: {q.reasoning}\n"
            if q.recommended_answer:
                report_text += f"추천 답변: {q.recommended_answer}\n"
            report_text += "\n"
            yield f"data: {json.dumps({'chunk': report_text}, ensure_ascii=False)}{sse_end}"

        # 종합 피드백
        summary_text = "━━━ 종합 평가 ━━━\n"
        summary_text += f"종합 점수: {final.overall_score}/5\n"
        summary_text += f"종합 피드백: {final.overall_feedback}\n\n"

        if final.strengths:
            summary_text += "강점:\n"
            for s in final.strengths:
                summary_text += f"  - {s}\n"
            summary_text += "\n"

        if final.improvements:
            summary_text += "개선점:\n"
            for imp in final.improvements:
                summary_text += f"  - {imp}\n"
            summary_text += "\n"

        yield f"data: {json.dumps({'chunk': summary_text}, ensure_ascii=False)}{sse_end}"

        logger.info("✅ [Debate] 최종 리포트 스트리밍 완료")

    except Exception as e:
        logger.error("Debate 리포트 생성 오류: %s", str(e), exc_info=True)
        yield sse_error_event(
            code="LLM_ERROR",
            status=500,
            message=str(e),
            fallback="심층 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )

    yield f"data: [DONE]{sse_end}"


# ============================================
# 통합 엔드포인트
# ============================================


@router.post(
    "/analyze",
    summary="면접 평가 리포트 생성",
    description="""
    면접 종료 시 Q&A 데이터를 받아 SSE 스트리밍으로 평가 리포트를 생성합니다.

    **retry=false (기본):** Gemini 단독 분석 → SSE 리포트
    **retry=true (답변 다시 받기):** Gemini×GPT-4o 토론 → SSE 리포트
    """,
)
async def analyze_interview(
    request: AnalyzeInterviewRequest,
    analyzer: InterviewAnalyzer = Depends(get_evaluation_analyzer),
    debate_service: DebateService | None = Depends(get_debate_service),
):
    """면접 Q&A 기반 평가 리포트를 SSE 스트리밍으로 생성합니다."""
    logger.info(
        "Analyze interview: session=%s, user=%s, questions=%d, retry=%s",
        _sanitize_log_value(str(request.session_id)),
        _sanitize_log_value(str(request.user_id)),
        len(request.context),
        request.retry,
    )

    # retry=true → Gemini×GPT-4o 토론
    if request.retry:
        if not debate_service:
            # debate 서비스 미사용 시 Gemini 단독으로 폴백
            logger.warning("Debate service unavailable, falling back to Gemini-only")
            return StreamingResponse(
                _generate_analyze_stream(request),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        return StreamingResponse(
            _generate_debate_stream(request, analyzer, debate_service),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # retry=false → Gemini 단독 분석
    return StreamingResponse(
        _generate_analyze_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
