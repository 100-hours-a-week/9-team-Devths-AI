"""
LLM 구조화 출력 (Structured Output) Pydantic 모델.

LangChain의 with_structured_output()과 함께 사용하여
LLM 응답 포맷을 강제합니다.
"""

from pydantic import BaseModel, Field


# ============================================
# 면접 질문 생성
# ============================================


class InterviewQuestion(BaseModel):
    """면접 질문 구조화 출력."""

    question: str = Field(..., description="면접 질문 내용")
    difficulty: str = Field(
        default="medium",
        description="난이도 (easy, medium, hard)",
    )
    category: str = Field(
        default="기술",
        description="질문 카테고리 (기술, 인성, 상황)",
    )
    follow_up: bool = Field(
        default=False,
        description="꼬리질문 여부",
    )


class InterviewQuestionBatch(BaseModel):
    """면접 질문 배치 생성 구조화 출력."""

    questions: list[InterviewQuestion] = Field(
        ..., description="생성된 면접 질문 목록"
    )


# ============================================
# 면접 분석/평가
# ============================================


class AnalysisResult(BaseModel):
    """채용공고/이력서 분석 구조화 출력."""

    company_name: str = Field(default="", description="회사명")
    job_title: str = Field(default="", description="채용 직무")
    key_requirements: list[str] = Field(
        default_factory=list, description="핵심 요구사항"
    )
    candidate_strengths: list[str] = Field(
        default_factory=list, description="지원자 강점"
    )
    candidate_weaknesses: list[str] = Field(
        default_factory=list, description="지원자 약점/보완점"
    )
    match_score: int = Field(
        default=0, ge=0, le=100, description="적합도 점수 (0-100)"
    )
    summary: str = Field(default="", description="종합 분석 요약")


# ============================================
# 면접 답변 평가
# ============================================


class QuestionEvaluation(BaseModel):
    """개별 질문 평가 구조화 출력."""

    question: str = Field(..., description="면접 질문")
    user_answer: str = Field(..., description="지원자 답변")
    verdict: str = Field(
        ..., description="판정 (적절, 부적절, 보완필요)"
    )
    score: int = Field(..., ge=1, le=5, description="점수 (1-5)")
    reasoning: str = Field(..., description="평가 근거")
    recommended_answer: str | None = Field(
        None, description="추천 답변 (보완/부적절 시)"
    )


class InterviewEvaluation(BaseModel):
    """면접 종합 평가 구조화 출력."""

    questions: list[QuestionEvaluation] = Field(
        default_factory=list, description="각 질문별 평가"
    )
    overall_score: int = Field(
        default=0, ge=0, le=5, description="종합 점수 (1-5)"
    )
    overall_feedback: str = Field(default="", description="종합 피드백")
    strengths: list[str] = Field(
        default_factory=list, description="강점"
    )
    improvements: list[str] = Field(
        default_factory=list, description="개선점"
    )


# ============================================
# 채팅방 제목 추출
# ============================================


class ChatRoomTitle(BaseModel):
    """채팅방 제목 추출 구조화 출력."""

    company_name: str = Field(default="", description="회사명")
    job_title: str = Field(default="", description="채용 직무")
    title: str = Field(default="", description="생성된 채팅방 제목")


# ============================================
# 캘린더 파싱
# ============================================


class CalendarEvent(BaseModel):
    """채용 일정 파싱 구조화 출력."""

    title: str = Field(default="", description="일정 제목")
    start_date: str = Field(default="", description="시작 날짜 (YYYY-MM-DD)")
    end_date: str = Field(default="", description="종료 날짜 (YYYY-MM-DD)")
    description: str = Field(default="", description="일정 설명")
    company_name: str = Field(default="", description="회사명")
    event_type: str = Field(
        default="기타", description="일정 유형 (서류마감, 면접, 코딩테스트 등)"
    )
