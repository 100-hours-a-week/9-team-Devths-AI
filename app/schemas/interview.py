from pydantic import BaseModel, Field
from typing import List
from enum import Enum


class InterviewType(str, Enum):
    """면접 타입"""
    PERSONALITY = "personality"
    TECHNICAL = "technical"


class EndedBy(str, Enum):
    """면접 종료 방식"""
    AUTO = "auto"  # 5개 완료
    MANUAL = "manual"  # 직접 종료


class InterviewQuestionRequest(BaseModel):
    """면접 질문 생성 요청 (API 4)"""
    room_id: str = Field(..., description="채팅방 ID")
    session_id: str = Field(..., description="면접 세션 ID")
    interview_type: InterviewType = Field(..., description="면접 타입 (personality/technical)")
    resume_text: str = Field(..., description="이력서 텍스트")
    posting_text: str = Field(..., description="채용공고 텍스트")

    class Config:
        json_schema_extra = {
            "example": {
                "room_id": "room_001",
                "session_id": "session_abc123",
                "interview_type": "technical",
                "resume_text": "이력서 텍스트...",
                "posting_text": "채용공고 텍스트..."
            }
        }


class InterviewQuestionResponse(BaseModel):
    """면접 질문 생성 응답 (API 4)"""
    success: bool = Field(True, description="성공 여부")
    question_id: str = Field(..., description="생성된 질문 ID")
    question: str = Field(..., description="생성된 질문")
    is_followup: bool = Field(..., description="꼬리질문 여부")
    question_number: int = Field(..., ge=1, le=5, description="질문 번호 (1-5)")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "question_id": "q_001",
                "question": "React의 Virtual DOM이 무엇인가요?",
                "is_followup": False,
                "question_number": 1
            }
        }


class InterviewSaveRequest(BaseModel):
    """면접 Q&A 저장 요청 (API 5)"""
    room_id: str = Field(..., description="채팅방 ID")
    session_id: str = Field(..., description="면접 세션 ID")
    question_id: str = Field(..., description="질문 ID")
    question: str = Field(..., description="질문")
    answer: str = Field(..., description="답변")
    is_followup: bool = Field(..., description="꼬리질문 여부")
    question_number: int = Field(..., ge=1, le=5, description="질문 번호")

    class Config:
        json_schema_extra = {
            "example": {
                "room_id": "room_001",
                "session_id": "session_abc123",
                "question_id": "q_001",
                "question": "React의 Virtual DOM이 무엇인가요?",
                "answer": "실제 DOM과 비교해서 변경된 부분만 업데이트하는 거예요",
                "is_followup": False,
                "question_number": 1
            }
        }


class InterviewSaveResponse(BaseModel):
    """면접 Q&A 저장 응답 (API 5)"""
    success: bool = Field(True, description="성공 여부")
    qa_id: str = Field(..., description="저장된 Q&A ID")
    session_id: str = Field(..., description="면접 세션 ID")
    saved_count: int = Field(..., ge=1, le=5, description="현재까지 저장된 Q&A 수")
    max_questions: int = Field(5, description="최대 질문 수")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "qa_id": "qa_001",
                "session_id": "session_abc123",
                "saved_count": 1,
                "max_questions": 5
            }
        }


class InterviewReportRequest(BaseModel):
    """면접 평가 요청 (API 6)"""
    room_id: str = Field(..., description="채팅방 ID")
    session_id: str = Field(..., description="면접 세션 ID")
    interview_type: InterviewType = Field(..., description="면접 타입")
    resume_text: str = Field(..., description="이력서 텍스트")
    posting_text: str = Field(..., description="채용공고 텍스트")
    ended_by: EndedBy = Field(..., description="종료 방식 (auto/manual)")

    class Config:
        json_schema_extra = {
            "example": {
                "room_id": "room_001",
                "session_id": "session_abc123",
                "interview_type": "technical",
                "resume_text": "이력서 텍스트...",
                "posting_text": "채용공고 텍스트...",
                "ended_by": "auto"
            }
        }


class QAEvaluation(BaseModel):
    """Q&A 평가"""
    qa_id: str = Field(..., description="Q&A ID")
    question: str = Field(..., description="질문")
    answer: str = Field(..., description="답변")
    score: int = Field(..., ge=0, le=100, description="점수 (0-100)")
    good_points: List[str] = Field(..., description="잘한 점")
    improvements: List[str] = Field(..., description="개선할 점")


class Report(BaseModel):
    """면접 리포트"""
    total_score: int = Field(..., ge=0, le=100, description="총점")
    grade: str = Field(..., description="등급 (A+, A, B+, B, C+, C, D, F)")
    strength_patterns: List[str] = Field(..., description="강점 패턴")
    weakness_patterns: List[str] = Field(..., description="약점 패턴")
    learning_guide: List[str] = Field(..., description="학습 가이드")


class InterviewReportResponse(BaseModel):
    """면접 평가 응답 (API 6)"""
    success: bool = Field(True, description="성공 여부")
    room_id: str = Field(..., description="채팅방 ID")
    session_id: str = Field(..., description="면접 세션 ID")
    evaluations: List[QAEvaluation] = Field(..., description="Q&A별 평가")
    report: Report = Field(..., description="종합 리포트")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "room_id": "room_001",
                "session_id": "session_abc123",
                "evaluations": [
                    {
                        "qa_id": "qa_001",
                        "question": "React의 Virtual DOM이 무엇인가요?",
                        "answer": "실제 DOM과 비교해서...",
                        "score": 80,
                        "good_points": ["Virtual DOM의 기본 개념을 잘 이해하고 있음"],
                        "improvements": ["Reconciliation 알고리즘 설명 추가하면 좋음"]
                    }
                ],
                "report": {
                    "total_score": 78,
                    "grade": "B+",
                    "strength_patterns": ["기술 개념에 대한 이해도가 높음"],
                    "weakness_patterns": ["심화 개념 설명이 부족함"],
                    "learning_guide": ["React 심화 개념 학습 (Fiber, Concurrent Mode)"]
                }
            }
        }
