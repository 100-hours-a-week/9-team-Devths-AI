"""
면접 답변 평가 API 엔드포인트 (통합).

POST /ai/evaluation/analyze - 면접 평가 리포트 생성 (SSE 스트리밍)
  - retry=false: Gemini 단독 분석
  - retry=true:  Gemini×GPT-4o 토론 후 최종 리포트
"""

import json
import logging
import re

from fastapi import APIRouter, Depends
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


def _validate_qa_list(qa_list: list[dict]) -> str | None:
    """Q&A 리스트의 형식을 검증합니다.

    Returns:
        에러 메시지 (정상이면 None)
    """
    for i, qa in enumerate(qa_list):
        if "question" not in qa or "answer" not in qa:
            return f"context[{i}]에 question, answer 필드가 필요합니다."
    return None


# ============================================
# Gemini 단독 분석 스트리밍 (retry=false)
# ============================================


async def _generate_analyze_stream(request: AnalyzeInterviewRequest):
    """면접 평가 리포트 SSE 스트리밍 생성 (Gemini 단독)."""
    sse_end = "\n\n"
    qa_list = request.context or []

    # context 빈 배열 검사
    if not qa_list:
        yield sse_error_event(
            code="EMPTY_CONTEXT",
            status=400,
            message="평가할 Q&A 데이터가 없습니다.",
            fallback="면접 문답 데이터가 비어 있습니다. context에 Q&A 목록을 전달해주세요.",
        )
        yield f"data: [DONE]{sse_end}"
        return

    # context Q&A 형식 검증
    validation_error = _validate_qa_list(qa_list)
    if validation_error:
        yield sse_error_event(
            code="INVALID_CONTEXT",
            status=400,
            message=validation_error,
            fallback="Q&A 데이터 형식이 올바르지 않습니다. 각 항목에 question과 answer를 포함해주세요.",
        )
        yield f"data: [DONE]{sse_end}"
        return

    # RAG 서비스 초기화
    try:
        rag = get_services()
    except Exception as e:
        logger.error("RAG 서비스 초기화 실패: %s", str(e), exc_info=True)
        yield sse_error_event(
            code="LLM_UNAVAILABLE",
            status=503,
            message=str(e),
            fallback="AI 서비스에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.",
        )
        yield f"data: [DONE]{sse_end}"
        return

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

    except ConnectionError as e:
        logger.error("LLM 서비스 연결 실패: %s", str(e), exc_info=True)
        yield sse_error_event(
            code="LLM_UNAVAILABLE",
            status=503,
            message=str(e),
            fallback="AI 서비스에 연결할 수 없습니다. 잠시 후 다시 시도해주세요.",
        )
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


def _format_analysis_as_text(analysis) -> str:
    """InterviewAnalysis 객체를 텍스트 리포트로 변환합니다."""
    lines = []

    for i, q in enumerate(analysis.questions, 1):
        lines.append(f"━━━ 질문 {i} ━━━")
        lines.append(f"질문: {q.question}")
        lines.append(f"답변: {q.user_answer}")
        lines.append(f"판정: {q.verdict} (점수: {q.score}/5)")
        lines.append(f"평가: {q.reasoning}")
        if q.recommended_answer:
            lines.append(f"추천 답변: {q.recommended_answer}")
        lines.append("")

    lines.append("━━━ 종합 평가 ━━━")
    lines.append(f"종합 점수: {analysis.overall_score}/5")
    lines.append(f"종합 피드백: {analysis.overall_feedback}")
    lines.append("")

    if analysis.strengths:
        lines.append("강점:")
        for s in analysis.strengths:
            lines.append(f"  - {s}")
        lines.append("")

    if analysis.improvements:
        lines.append("개선점:")
        for imp in analysis.improvements:
            lines.append(f"  - {imp}")
        lines.append("")

    return "\n".join(lines)


async def _generate_debate_stream(
    request: AnalyzeInterviewRequest,
    analyzer: InterviewAnalyzer,
    debate_service: DebateService,
):
    """Gemini×GPT-4o 토론 후 최종 리포트 SSE 스트리밍 생성."""
    sse_end = "\n\n"
    qa_list = request.context or []

    # context 빈 배열 검사
    if not qa_list:
        yield sse_error_event(
            code="EMPTY_CONTEXT",
            status=400,
            message="평가할 Q&A 데이터가 없습니다.",
            fallback="면접 문답 데이터가 비어 있습니다. context에 Q&A 목록을 전달해주세요.",
        )
        yield f"data: [DONE]{sse_end}"
        return

    # context Q&A 형식 검증
    validation_error = _validate_qa_list(qa_list)
    if validation_error:
        yield sse_error_event(
            code="INVALID_CONTEXT",
            status=400,
            message=validation_error,
            fallback="Q&A 데이터 형식이 올바르지 않습니다. 각 항목에 question과 answer를 포함해주세요.",
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

    gemini_result = None

    # ── 1단계: Gemini 구조화 분석 ──
    try:
        progress = "🔍 Gemini 분석 중...\n"
        yield f"data: {json.dumps({'chunk': progress}, ensure_ascii=False)}{sse_end}"

        logger.info("📊 [Debate] 1단계: Gemini 구조화 분석 시작")
        gemini_result = await analyzer.analyze(qa_pairs=qa_pairs)
        gemini_dict = gemini_result.to_dict()
        logger.info("✅ [Debate] 1단계 완료: overall_score=%d", gemini_result.overall_score)

    except Exception as e:
        logger.error("Gemini 분석 실패: %s", str(e), exc_info=True)
        yield sse_error_event(
            code="LLM_ERROR",
            status=500,
            message=str(e),
            fallback="Gemini 분석 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
        )
        yield f"data: [DONE]{sse_end}"
        return

    # ── 2단계: Gemini×GPT-4o 토론 ──
    try:
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
        report_text = _format_analysis_as_text(final)
        yield f"data: {json.dumps({'chunk': report_text}, ensure_ascii=False)}{sse_end}"

        logger.info("✅ [Debate] 최종 리포트 스트리밍 완료")

    except Exception as e:
        logger.error("Debate 토론 실패: %s", str(e), exc_info=True)

        # 토론 실패 시 Gemini 1단계 결과라도 반환
        yield sse_error_event(
            code="DEBATE_ERROR",
            status=500,
            message=str(e),
            fallback="심층 분석 중 오류가 발생했습니다. Gemini 단독 분석 결과를 제공합니다.",
        )

        if gemini_result and gemini_result.questions:
            fallback_text = "\n[Gemini 단독 분석 결과]\n\n"
            fallback_text += _format_analysis_as_text(gemini_result)
            yield f"data: {json.dumps({'chunk': fallback_text}, ensure_ascii=False)}{sse_end}"

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
    responses={
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "examples": {
                        "empty_context": {
                            "value": {
                                "detail": {
                                    "code": "EMPTY_CONTEXT",
                                    "message": "평가할 Q&A 데이터가 없습니다.",
                                }
                            }
                        },
                        "invalid_context": {
                            "value": {
                                "detail": {
                                    "code": "INVALID_CONTEXT",
                                    "message": "context의 각 항목에 question, answer 필드가 필요합니다.",
                                }
                            }
                        },
                    }
                }
            },
        },
        422: {
            "description": "Validation Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "code": "VALIDATION_ERROR",
                            "message": "필수 필드 누락 (room_id, user_id, session_id, context)",
                        }
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "examples": {
                        "llm_error": {
                            "value": {
                                "detail": {
                                    "code": "LLM_ERROR",
                                    "message": "LLM 서비스 호출 실패하였습니다.",
                                }
                            }
                        },
                        "debate_error": {
                            "value": {
                                "detail": {
                                    "code": "DEBATE_ERROR",
                                    "message": "심층 분석 중 오류가 발생했습니다.",
                                }
                            }
                        },
                        "stream_error": {
                            "value": {
                                "detail": {
                                    "code": "STREAM_ERROR",
                                    "message": "스트리밍 연결이 중단되었습니다.",
                                }
                            }
                        },
                    }
                }
            },
        },
        503: {
            "description": "Service Unavailable",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "code": "LLM_UNAVAILABLE",
                            "message": "AI 서비스에 연결할 수 없습니다.",
                        }
                    }
                }
            },
        },
    },
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

    sse_headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    # retry=true → Gemini×GPT-4o 토론
    if request.retry:
        if not debate_service:
            logger.warning("Debate service unavailable, falling back to Gemini-only")
            return StreamingResponse(
                _generate_analyze_stream(request),
                media_type="text/event-stream",
                headers=sse_headers,
            )

        return StreamingResponse(
            _generate_debate_stream(request, analyzer, debate_service),
            media_type="text/event-stream",
            headers=sse_headers,
        )

    # retry=false → Gemini 단독 분석
    return StreamingResponse(
        _generate_analyze_stream(request),
        media_type="text/event-stream",
        headers=sse_headers,
    )
