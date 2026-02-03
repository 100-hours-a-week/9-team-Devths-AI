from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class MessageRole(str, Enum):
    """메시지 역할"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


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

    NORMAL = "normal"  # 일반 대화
    INTERVIEW = "interview"  # 면접 모드 (질문 생성)
    REPORT = "report"  # 면접 평가 리포트 생성


class LLMModel(str, Enum):
    """LLM 모델 선택"""

    GEMINI = "gemini"  # Google Gemini
    VLLM = "vllm"  # vLLM (Llama-3-Korean-Bllossom-8B)


class InterviewType(str, Enum):
    """면접 유형"""

    BEHAVIOR = "behavior"  # 인성 면접
    TECH = "tech"  # 기술 면접


class ChatContext(BaseModel):
    """채팅 컨텍스트 (모드별 추가 정보)"""

    mode: ChatMode = Field(
        default=ChatMode.NORMAL, description="채팅 모드 (normal/interview/report)"
    )

    # 문서 정보 (OCR 텍스트)
    resume_ocr: str | None = Field(None, description="이력서 OCR 텍스트 (면접 모드 시)")
    job_posting_ocr: str | None = Field(None, description="채용공고 OCR 텍스트 (면접 모드 시)")
    portfolio_text: str | None = Field(None, description="포트폴리오 텍스트 (면접 모드 시)")

    # 면접 모드
    interview_type: str | None = Field(None, description="면접 유형 (behavior/tech)")
    question_count: int | None = Field(None, description="현재까지 생성된 질문 수")
    asked_questions: list[str] | None = Field(
        None, description="이미 했던 질문 목록 (반복 방지, 새 질문 생성 시 전달)"
    )

    # 기술 면접 세션 상태 (5개 질문 × 최대 3 depths 꼬리질문)
    interview_session: "InterviewSession | None" = Field(
        None, description="기술 면접 세션 상태 (질문 세트, 현재 진행 상황)"
    )

    # 리포트 모드
    qa_list: list[dict] | None = Field(None, description="Q&A 목록 (리포트 생성 시)")

    class Config:
        extra = "allow"  # 추가 필드 허용


class InterviewQuestionState(BaseModel):
    """개별 면접 질문 상태"""

    id: int = Field(..., description="질문 ID (1-5)")
    category: str = Field(..., description="카테고리 코드 (cs_network_os, cs_db_algo, ...)")
    category_name: str = Field(..., description="카테고리 이름 (기본 CS, 프로젝트 기술 등)")
    question: str = Field(..., description="질문 내용")
    intent: str = Field(default="", description="질문 의도")
    keywords: list[str] = Field(default=[], description="관련 키워드")

    # 진행 상태
    is_completed: bool = Field(default=False, description="질문 완료 여부")
    current_depth: int = Field(default=0, description="현재 꼬리질문 깊이 (0-3)")
    max_depth: int = Field(default=3, description="최대 꼬리질문 깊이")

    # 대화 이력 (해당 질문에 대한)
    conversation: list[dict] = Field(
        default=[],
        description="대화 이력 [{'role': 'interviewer|candidate', 'content': '...'}]",
    )


class InterviewSession(BaseModel):
    """기술 면접 세션 전체 상태"""

    session_id: str = Field(..., description="면접 세션 고유 ID")
    interview_type: str = Field(default="tech", description="면접 유형 (tech/behavior)")

    # 5개 질문 세트
    questions: list[InterviewQuestionState] = Field(
        default=[], description="5개 질문 세트"
    )

    # 현재 진행 상태
    current_question_id: int = Field(default=1, description="현재 진행 중인 질문 ID (1-5)")
    total_questions: int = Field(default=5, description="총 질문 수")

    # 면접 진행 단계
    phase: str = Field(
        default="init",
        description="진행 단계 (init: 초기화, questioning: 질문 중, followup: 꼬리질문 중, completed: 완료)",
    )

    # 전체 대화 이력 (모든 질문 통합)
    full_conversation: list[dict] = Field(
        default=[],
        description="전체 대화 이력",
    )


class QAItem(BaseModel):
    """Q&A 항목 (면접 리포트용)"""

    question: str = Field(..., description="질문")
    answer: str = Field(..., description="답변")


class ChatRequest(BaseModel):
    """채팅 요청 (통합)"""

    room_id: int = Field(..., description="채팅방 ID")
    user_id: int = Field(..., description="사용자 ID")
    message: str | None = Field(None, description="사용자 메시지")
    interview_id: int | None = Field(
        None, description="면접 ID (면접 모드일 때만 값 있음, 일반 채팅이면 null)"
    )
    model: LLMModel = Field(
        default=LLMModel.GEMINI, description="사용할 LLM 모델 (gemini 또는 vllm)"
    )
    context: ChatContext = Field(
        default_factory=lambda: ChatContext(mode=ChatMode.NORMAL),
        description="채팅 컨텍스트",
    )

    @validator("context", pre=True)
    def parse_context(cls, v):
        """context를 적절한 타입으로 파싱"""
        if v is None:
            return ChatContext(mode=ChatMode.NORMAL)

        # 이미 ChatContext 인스턴스면 그대로 반환
        if isinstance(v, ChatContext):
            return v

        # 리스트인 경우 → Q&A 배열로 간주 (리포트 모드)
        # 백엔드가 context: [{"question": "...", "answer": "..."}] 형태로 보낼 때 대응
        if isinstance(v, list):
            return ChatContext(mode=ChatMode.REPORT, qa_list=v)

        # 딕셔너리인 경우 ChatContext로 변환
        if isinstance(v, dict):
            # mode 값을 소문자로 정규화 (REPORT → report)
            if "mode" in v and isinstance(v["mode"], str):
                v["mode"] = v["mode"].lower()
            return ChatContext(**v)

        return v

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "일반 모드",
                    "value": {
                        "model": "gemini",
                        "room_id": 1,
                        "user_id": 10,
                        "message": "질문 내용",
                        "interview_id": None,
                        "context": {
                            "mode": "normal",
                            "resume_ocr": None,
                            "job_posting_ocr": None,
                            "interview_type": None,
                            "question_count": None,
                        },
                    },
                },
                {
                    "name": "면접 모드",
                    "value": {
                        "model": "gemini",
                        "room_id": 1,
                        "user_id": 10,
                        "message": "질문 내용",
                        "interview_id": 100,
                        "context": {
                            "mode": "interview",
                            "resume_ocr": "이력서 OCR 텍스트",
                            "job_posting_ocr": "채용공고 OCR 텍스트",
                            "interview_type": "behavior",
                            "question_count": 3,
                            "asked_questions": [
                                "자기소개 해주세요",
                                "프로젝트 경험을 말씀해주세요",
                            ],
                        },
                    },
                },
                {
                    "name": "리포트 모드",
                    "value": {
                        "model": "gemini",
                        "room_id": 1,
                        "user_id": 10,
                        "message": None,
                        "interview_id": 100,
                        "context": {
                            "mode": "report",
                            "interview_type": "tech",
                            "qa_list": [
                                {"question": "자기소개 해주세요", "answer": "안녕하세요..."},
                                {"question": "프로젝트 경험을 말씀해주세요", "answer": "저는..."},
                            ],
                        },
                    },
                },
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

    strengths: list[str] = Field(..., description="강점 목록")
    weaknesses: list[str] = Field(..., description="약점 목록")
    suggestions: list[str] = Field(..., description="개선 제안")


class PostingAnalysis(BaseModel):
    """채용공고 분석 결과 (Pydantic AI 모델)"""

    company: str = Field(..., description="회사명")
    position: str = Field(..., description="포지션")
    required_skills: list[str] = Field(..., description="필수 스킬")
    preferred_skills: list[str] = Field(default=[], description="우대 스킬")


class MatchingResult(BaseModel):
    """매칭도 분석 결과 (Pydantic AI 모델)"""

    score: int = Field(..., ge=0, le=100, description="매칭 점수")
    grade: MatchGrade = Field(..., description="등급")
    matched_skills: list[str] = Field(..., description="매칭된 스킬")
    missing_skills: list[str] = Field(..., description="부족한 스킬")


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
    good_points: list[str] = Field(..., description="잘한 점")
    improvements: list[str] = Field(..., description="개선점")


class InterviewReport(BaseModel):
    """면접 종합 리포트 (Pydantic AI result_type)"""

    evaluations: list[QAEvaluation] = Field(..., description="개별 Q&A 평가")
    strength_patterns: list[str] = Field(..., description="강점 패턴")
    weakness_patterns: list[str] = Field(..., description="약점 패턴")
    learning_guide: list[str] = Field(..., description="학습 가이드")


# ============================================================================
# 응답 스키마
# ============================================================================


class ToolUsed(BaseModel):
    """사용된 Tool 정보"""

    tool: ToolName = Field(..., description="Tool 이름")
    params: dict[str, Any] = Field(..., description="Tool 파라미터")


class ChatResponse(BaseModel):
    """채팅 응답 (통합)"""

    success: bool = Field(True, description="성공 여부")
    mode: ChatMode = Field(..., description="응답 모드")

    # 채팅방 제목 (회사명/채용직무)
    summary: str | None = Field(None, description="채팅방 제목 (회사명/채용직무)")

    # 일반 대화
    response: str | None = Field(None, description="텍스트 응답")

    # 분석 결과 (Pydantic AI)
    analysis: AnalysisResult | None = Field(None, description="분석 결과")

    # 면접 질문 (Pydantic AI)
    question: InterviewQuestion | None = Field(None, description="면접 질문")

    # 면접 리포트 (Pydantic AI)
    report: InterviewReport | None = Field(None, description="면접 리포트")

    # Tool 호출
    tool_used: ToolUsed | None = Field(None, description="실행할 Tool 정보")


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
                    "data": [{"title": "카카오 1차 면접", "date": "2026-01-08", "time": "14:00"}],
                },
            }
        }
