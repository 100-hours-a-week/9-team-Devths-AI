"""
면접 결과 VectorDB 적재용 스키마.

모의면접 결과(Q&A + 평가)를 구조화하여 VectorDB에 batch 저장할 때 사용합니다.
"""

from pydantic import BaseModel, Field


class InterviewResultDocument(BaseModel):
    """VectorDB에 저장할 개별 면접 Q&A 문서."""

    question: str = Field(..., description="면접 질문")
    answer: str = Field(..., description="지원자 답변")
    question_index: int = Field(..., ge=0, description="질문 번호 (0-based)")
    interview_type: str = Field("technical", description="면접 유형 (technical/personality)")

    # 평가 결과 (있을 경우)
    score: int | None = Field(None, ge=1, le=5, description="점수 (1-5)")
    verdict: str | None = Field(None, description="판정 (적절/부적절/보완필요)")
    reasoning: str | None = Field(None, description="평가 근거")
    recommended_answer: str | None = Field(None, description="추천 답변")


class InterviewIngestionInput(BaseModel):
    """면접 세션 결과 적재 입력."""

    session_id: str | int = Field(..., description="면접 세션 ID")
    user_id: str | int = Field(..., description="사용자 ID")
    room_id: str | int = Field(..., description="채팅방 ID")
    interview_type: str = Field("technical", description="면접 유형")
    qa_results: list[InterviewResultDocument] = Field(..., description="Q&A + 평가 결과 리스트")
