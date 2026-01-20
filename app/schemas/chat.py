from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class MessageRole(str, Enum):
    """메시지 역할"""
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """채팅 메시지"""
    role: MessageRole = Field(..., description="역할 (user/assistant)")
    content: str = Field(..., description="메시지 내용")


class ToolName(str, Enum):
    """Tool 이름"""
    GET_SCHEDULE = "get_schedule"
    ADD_SCHEDULE = "add_schedule"
    UPDATE_SCHEDULE = "update_schedule"
    DELETE_SCHEDULE = "delete_schedule"


class ChatMode(str, Enum):
    """채팅 모드"""
    GENERAL = "general"              # 일반 대화
    ANALYSIS = "analysis"            # 이력서/채용공고 분석
    INTERVIEW_QUESTION = "interview_question"  # 면접 질문 생성
    INTERVIEW_REPORT = "interview_report"      # 면접 리포트


class LLMModel(str, Enum):
    """LLM 모델 선택"""
    GEMINI = "gemini"                # Google Gemini
    VLLM = "vllm"                    # vLLM (Llama-3-Korean-Bllossom-8B)


class ChatContext(BaseModel):
    """채팅 컨텍스트 (모드별 추가 정보)"""
    mode: ChatMode = Field(default=ChatMode.GENERAL, description="채팅 모드")

    # 분석 모드
    resume_id: Optional[str] = Field(None, description="이력서 ID (분석 모드)")
    posting_id: Optional[str] = Field(None, description="채용공고 ID (분석 모드)")

    # 면접 모드
    session_id: Optional[str] = Field(None, description="면접 세션 ID (면접 모드)")
    interview_type: Optional[str] = Field(None, description="면접 유형 (personality/technical)")
    question_count: Optional[int] = Field(0, description="현재까지 생성된 질문 수")

    class Config:
        extra = 'allow'  # 추가 필드 허용


class ChatRequest(BaseModel):
    """채팅 요청 (통합)"""
    room_id: str = Field(..., description="채팅방 ID")
    user_id: str = Field(..., description="사용자 ID")
    message: Optional[str] = Field(None, description="사용자 메시지")
    model: LLMModel = Field(
        default=LLMModel.GEMINI,
        description="사용할 LLM 모델 (gemini 또는 vllm)"
    )
    context: ChatContext = Field(
        default_factory=lambda: ChatContext(mode=ChatMode.GENERAL),
        description="채팅 컨텍스트"
    )
    history: List[ChatMessage] = Field(
        default=[], 
        max_length=20,
        description="대화 히스토리 (최근 20개)"
    )

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "일반 대화",
                    "value": {
                        "room_id": "room_001",
                        "user_id": "user_456",
                        "message": "이력서 작성 팁 알려줘",
                        "context": {"mode": "general"}
                    }
                },
                {
                    "name": "분석 요청",
                    "value": {
                        "room_id": "room_001",
                        "user_id": "user_456",
                        "message": "이력서와 채용공고를 분석해주세요",
                        "context": {
                            "mode": "analysis",
                            "resume_id": "resume_123",
                            "posting_id": "posting_456"
                        }
                    }
                }
            ]
        }


# ============================================================================
# Pydantic AI 모델들 (LLM 출력 구조화)
# ============================================================================

class MatchGrade(str, Enum):
    """매칭 등급"""
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class ResumeAnalysis(BaseModel):
    """이력서 분석 결과 (Pydantic AI 모델)"""
    strengths: List[str] = Field(..., description="강점 목록")
    weaknesses: List[str] = Field(..., description="약점 목록")
    suggestions: List[str] = Field(..., description="개선 제안")


class PostingAnalysis(BaseModel):
    """채용공고 분석 결과 (Pydantic AI 모델)"""
    company: str = Field(..., description="회사명")
    position: str = Field(..., description="포지션")
    required_skills: List[str] = Field(..., description="필수 스킬")
    preferred_skills: List[str] = Field(default=[], description="우대 스킬")


class MatchingResult(BaseModel):
    """매칭도 분석 결과 (Pydantic AI 모델)"""
    score: int = Field(..., ge=0, le=100, description="매칭 점수")
    grade: MatchGrade = Field(..., description="등급")
    matched_skills: List[str] = Field(..., description="매칭된 스킬")
    missing_skills: List[str] = Field(..., description="부족한 스킬")


class AnalysisResult(BaseModel):
    """분석 결과 (Pydantic AI result_type)"""
    resume_analysis: ResumeAnalysis
    posting_analysis: PostingAnalysis
    matching: MatchingResult


class InterviewQuestion(BaseModel):
    """면접 질문 (Pydantic AI result_type)"""
    question: str = Field(..., description="생성된 질문")
    difficulty: str = Field(..., description="난이도 (easy/medium/hard)")
    category: str = Field(..., description="카테고리 (기술/인성/경험)")
    follow_up: bool = Field(default=False, description="꼬리질문 여부")


class QAEvaluation(BaseModel):
    """개별 Q&A 평가 (Pydantic AI 모델)"""
    question: str = Field(..., description="질문")
    answer: str = Field(..., description="답변")
    good_points: List[str] = Field(..., description="잘한 점")
    improvements: List[str] = Field(..., description="개선점")


class InterviewReport(BaseModel):
    """면접 종합 리포트 (Pydantic AI result_type)"""
    evaluations: List[QAEvaluation] = Field(..., description="개별 Q&A 평가")
    strength_patterns: List[str] = Field(..., description="강점 패턴")
    weakness_patterns: List[str] = Field(..., description="약점 패턴")
    learning_guide: List[str] = Field(..., description="학습 가이드")


# ============================================================================
# 응답 스키마
# ============================================================================

class ToolUsed(BaseModel):
    """사용된 Tool 정보"""
    tool: ToolName = Field(..., description="Tool 이름")
    params: Dict[str, Any] = Field(..., description="Tool 파라미터")


class ChatResponse(BaseModel):
    """채팅 응답 (통합)"""
    success: bool = Field(True, description="성공 여부")
    mode: ChatMode = Field(..., description="응답 모드")
    
    # 일반 대화
    response: Optional[str] = Field(None, description="텍스트 응답")
    
    # 분석 결과 (Pydantic AI)
    analysis: Optional[AnalysisResult] = Field(None, description="분석 결과")
    
    # 면접 질문 (Pydantic AI)
    question: Optional[InterviewQuestion] = Field(None, description="면접 질문")
    
    # 면접 리포트 (Pydantic AI)
    report: Optional[InterviewReport] = Field(None, description="면접 리포트")
    
    # Tool 호출
    tool_used: Optional[ToolUsed] = Field(None, description="실행할 Tool 정보")


class ToolResult(BaseModel):
    """Tool 실행 결과"""
    tool: ToolName = Field(..., description="Tool 이름")
    success: bool = Field(..., description="실행 성공 여부")
    data: Any = Field(..., description="Tool 실행 결과 데이터")


class ChatToolResultRequest(BaseModel):
    """Tool 결과 전달 요청"""
    room_id: str = Field(..., description="채팅방 ID")
    user_id: str = Field(..., description="사용자 ID")
    tool_result: ToolResult = Field(..., description="Tool 실행 결과")

    class Config:
        json_schema_extra = {
            "example": {
                "room_id": "room_001",
                "user_id": "user_456",
                "tool_result": {
                    "tool": "get_schedule",
                    "success": True,
                    "data": [
                        {"title": "카카오 1차 면접", "date": "2026-01-08", "time": "14:00"}
                    ]
                }
            }
        }
