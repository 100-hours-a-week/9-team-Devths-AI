"""
면접 답변 평가 API 스키마.

1단계: Gemini 분석 요청/응답
2단계: 토론(심층 분석) 요청/응답
"""

from pydantic import BaseModel, Field

from app.schemas.chat import LLMModel

# ============================================
# 공통 타입
# ============================================


class QAPair(BaseModel):
    """질의응답 쌍."""

    question: str = Field(..., description="면접 질문")
    answer: str = Field(..., description="지원자 답변")
    category: str = Field("", description="질문 카테고리 (optional)")


class QuestionAnalysisResponse(BaseModel):
    """개별 질문 분석 결과."""

    question: str = Field(..., description="면접 질문")
    user_answer: str = Field(..., description="지원자 답변")
    verdict: str = Field(..., description="판정 (적절/부적절/보완필요)")
    score: int = Field(..., ge=1, le=5, description="점수 (1-5)")
    reasoning: str = Field(..., description="평가 근거")
    recommended_answer: str | None = Field(None, description="추천 답변 (보완/부적절 시)")
    category: str = Field("", description="질문 카테고리")


# ============================================
# 1단계: Gemini 분석
# ============================================


class AnalyzeInterviewRequest(BaseModel):
    """면접 분석 요청 (1단계 - 면접 종료 시).

    프론트엔드에서 면접 종료 버튼을 누르면
    채팅에서 수집된 Q&A 데이터를 context로 전달하여 평가 리포트를 생성합니다.
    """

    model: LLMModel = Field(
        default=LLMModel.GEMINI, description="사용할 LLM 모델 (gemini 또는 vllm)"
    )
    room_id: int = Field(..., description="채팅방 ID")
    user_id: int = Field(..., description="사용자 ID")
    message: str | None = Field(None, description="사용자 메시지")
    session_id: int | str = Field(..., description="면접 세션 ID")
    context: list[dict] = Field(..., description="Q&A 목록 [{question: str, answer: str}, ...]")
    retry: bool = Field(
        default=False,
        description="true면 Gemini×GPT-4o 토론(답변 다시 받기), false면 Gemini 단독 분석",
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "면접 리포트 생성 (면접 종료)",
                    "value": {
                        "model": "gemini",
                        "room_id": 1,
                        "user_id": 12,
                        "message": None,
                        "session_id": 23,
                        "context": [
                            {
                                "question": "자기소개 해주세요",
                                "answer": "안녕하세요...",
                            },
                            {
                                "question": "프로젝트 경험을 말씀해주세요",
                                "answer": "저는...",
                            },
                        ],
                        "retry": False,
                    },
                },
                {
                    "name": "답변 다시 받기 (Gemini×GPT-4o 토론)",
                    "value": {
                        "model": "gemini",
                        "room_id": 1,
                        "user_id": 12,
                        "message": None,
                        "session_id": 23,
                        "context": [
                            {
                                "question": "자기소개 해주세요",
                                "answer": "안녕하세요...",
                            },
                            {
                                "question": "프로젝트 경험을 말씀해주세요",
                                "answer": "저는...",
                            },
                        ],
                        "retry": True,
                    },
                },
            ]
        }


class AnalyzeInterviewResponse(BaseModel):
    """면접 분석 응답 (1단계)."""

    success: bool = Field(True, description="성공 여부")
    session_id: str = Field(..., description="면접 세션 ID")
    questions: list[QuestionAnalysisResponse] = Field(
        default_factory=list, description="각 질문별 분석 결과"
    )
    overall_score: int = Field(0, ge=0, le=5, description="종합 점수")
    overall_feedback: str = Field("", description="종합 피드백")
    strengths: list[str] = Field(default_factory=list, description="강점")
    improvements: list[str] = Field(default_factory=list, description="개선점")
    model_used: str = Field("", description="사용된 모델")
    debate_available: bool = Field(False, description="심층 분석(토론) 가능 여부")


# ============================================
# 2단계: 토론 (심층 분석)
# ============================================


class DebateRequest(BaseModel):
    """심층 분석(토론) 요청 (2단계 - 사용자 수동 트리거)."""

    session_id: str = Field(..., description="면접 세션 ID")
    qa_pairs: list[QAPair] = Field(..., description="질의응답 목록")
    gemini_analysis: dict = Field(..., description="1단계 Gemini 분석 결과")
    resume_text: str = Field("", description="이력서 텍스트")
    job_posting_text: str = Field("", description="채용공고 텍스트")
    interview_type: str = Field("tech", description="면접 유형")


class DisagreementDetail(BaseModel):
    """불일치 항목."""

    question_index: int = Field(..., description="질문 인덱스")
    question: str = Field("", description="질문 내용")
    gemini_score: int = Field(0, description="Gemini 점수")
    gpt4o_score: int = Field(0, description="GPT-4o 점수")
    score_diff: int = Field(0, description="점수 차이")


class DebateResponse(BaseModel):
    """심층 분석(토론) 응답 (2단계)."""

    success: bool = Field(True, description="성공 여부")
    session_id: str = Field(..., description="면접 세션 ID")

    # 최종 합의 결과
    final_analysis: AnalyzeInterviewResponse = Field(..., description="최종 분석 결과 (토론 합의)")

    # 개별 분석 비교
    gemini_analysis: AnalyzeInterviewResponse = Field(..., description="Gemini 분석 결과")
    gpt4o_analysis: AnalyzeInterviewResponse | None = Field(None, description="GPT-4o 분석 결과")

    # 토론 메타데이터
    disagreements: list[DisagreementDetail] = Field(
        default_factory=list, description="불일치 항목들"
    )
    consensus_method: str = Field("single", description="합의 방법 (single/merged/debated)")
