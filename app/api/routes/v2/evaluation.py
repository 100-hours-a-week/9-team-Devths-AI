"""
면접 답변 평가 API 엔드포인트.

1단계: POST /ai/evaluation/analyze - Gemini 3 Pro 면접 분석 (면접 종료 시)
2단계: POST /ai/evaluation/debate  - Gemini×OpenAI 토론 (사용자 수동 트리거)
"""

import logging

from fastapi import APIRouter, Depends, HTTPException

from app.config.dependencies import (
    get_debate_service,
    get_evaluation_analyzer,
)
from app.domain.evaluation.analyzer import InterviewAnalyzer
from app.domain.evaluation.debate_graph import DebateService
from app.schemas.evaluation import (
    AnalyzeInterviewRequest,
    AnalyzeInterviewResponse,
    DebateRequest,
    DebateResponse,
    DisagreementDetail,
    QuestionAnalysisResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluation")


# ============================================
# 1단계: Gemini 면접 분석
# ============================================


@router.post(
    "/analyze",
    response_model=AnalyzeInterviewResponse,
    summary="면접 답변 분석 (1단계)",
    description="면접 종료 시 Gemini 3 Pro가 전체 Q&A를 분석합니다.",
)
async def analyze_interview(
    request: AnalyzeInterviewRequest,
    analyzer: InterviewAnalyzer = Depends(get_evaluation_analyzer),
):
    """면접 종료 시 Gemini 3 Pro로 답변을 분석합니다.

    각 질문별로:
    - 답변 적절성 평가 (적절/부적절/보완필요)
    - 점수 (1-5)
    - 부적절/보완필요 시 추천 답변 제공
    """
    logger.info(
        "Analyze interview: session=%s, questions=%d",
        request.session_id,
        len(request.qa_pairs),
    )

    # Q&A 데이터 변환
    qa_pairs = [
        {
            "question": qa.question,
            "answer": qa.answer,
            "category": qa.category,
        }
        for qa in request.qa_pairs
    ]

    # Gemini 분석 실행
    analysis = await analyzer.analyze(
        qa_pairs=qa_pairs,
        resume_text=request.resume_text,
        job_posting_text=request.job_posting_text,
    )

    # 응답 구성
    return AnalyzeInterviewResponse(
        success=True,
        session_id=request.session_id,
        questions=[
            QuestionAnalysisResponse(
                question=q.question,
                user_answer=q.user_answer,
                verdict=q.verdict,
                score=q.score,
                reasoning=q.reasoning,
                recommended_answer=q.recommended_answer,
                category=q.category,
            )
            for q in analysis.questions
        ],
        overall_score=analysis.overall_score,
        overall_feedback=analysis.overall_feedback,
        strengths=analysis.strengths,
        improvements=analysis.improvements,
        model_used=analysis.model_used,
        debate_available=True,  # 토론 가능 여부는 True로 설정 (OpenAI 키 유무는 debate 호출 시 체크)
    )


# ============================================
# 2단계: Gemini×OpenAI 토론 (심층 분석)
# ============================================


@router.post(
    "/debate",
    response_model=DebateResponse,
    summary="심층 분석 - 토론 (2단계)",
    description="사용자가 심층 분석을 요청하면 Gemini와 GPT-4o가 토론하여 더 나은 분석을 제공합니다.",
)
async def debate_analysis(
    request: DebateRequest,
    debate_service: DebateService | None = Depends(get_debate_service),
):
    """Gemini×OpenAI 토론으로 심층 분석을 수행합니다.

    사용자가 1단계 분석 결과를 보고 부족하다고 판단할 때 호출합니다.
    - GPT-4o가 동일 Q&A를 독립 분석
    - 두 분석을 비교하여 불일치 항목 추출
    - 불일치 항목에 대해 토론 (1라운드)
    - 최종 합의 도출
    """
    if not debate_service:
        raise HTTPException(
            status_code=503,
            detail="토론 기능이 비활성화되어 있습니다. OpenAI API 키를 설정해주세요.",
        )

    logger.info(
        "Debate analysis: session=%s, questions=%d",
        request.session_id,
        len(request.qa_pairs),
    )

    # Q&A 데이터 변환
    qa_pairs = [
        {
            "question": qa.question,
            "answer": qa.answer,
            "category": qa.category,
        }
        for qa in request.qa_pairs
    ]

    # 토론 실행
    result = await debate_service.run_debate(
        qa_pairs=qa_pairs,
        gemini_analysis=request.gemini_analysis,
        resume_text=request.resume_text,
        job_posting_text=request.job_posting_text,
        interview_type=request.interview_type,
    )

    # 응답 구성
    def _to_analysis_response(analysis, session_id: str) -> AnalyzeInterviewResponse:
        return AnalyzeInterviewResponse(
            success=True,
            session_id=session_id,
            questions=[
                QuestionAnalysisResponse(
                    question=q.question,
                    user_answer=q.user_answer,
                    verdict=q.verdict,
                    score=q.score,
                    reasoning=q.reasoning,
                    recommended_answer=q.recommended_answer,
                    category=q.category,
                )
                for q in analysis.questions
            ],
            overall_score=analysis.overall_score,
            overall_feedback=analysis.overall_feedback,
            strengths=analysis.strengths,
            improvements=analysis.improvements,
            model_used=analysis.model_used,
            debate_available=False,  # 토론 후에는 더 이상 토론 불가
        )

    return DebateResponse(
        success=True,
        session_id=request.session_id,
        final_analysis=_to_analysis_response(result.final_analysis, request.session_id),
        gemini_analysis=_to_analysis_response(result.gemini_analysis, request.session_id),
        gpt4o_analysis=(
            _to_analysis_response(result.gpt4o_analysis, request.session_id)
            if result.gpt4o_analysis
            else None
        ),
        disagreements=[
            DisagreementDetail(
                question_index=d.get("question_index", 0),
                question=d.get("question", ""),
                gemini_score=d.get("gemini_score", 0),
                gpt4o_score=d.get("gpt4o_score", 0),
                score_diff=d.get("score_diff", 0),
            )
            for d in result.disagreements
        ],
        consensus_method=result.consensus_method,
    )
