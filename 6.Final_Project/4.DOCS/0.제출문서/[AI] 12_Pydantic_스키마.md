# Pydantic 스키마 명세 (v3.0)

> AI Server의 모든 Request/Response Pydantic 스키마 정의

---

## 📚 목차

- [1. 개요](#1-개요)
- [2. 공통 스키마](#2-공통-스키마)
- [3. 텍스트 추출 + 임베딩 (통합)](#3-텍스트-추출--임베딩-통합)
- [4. LLM 응답 (통합)](#4-llm-응답-통합)
- [5. 캘린더 파싱](#5-캘린더-파싱)
- [6. 마스킹](#6-마스킹)
- [7. 파일 구조](#7-파일-구조)

---

## 1. 개요

### 1.1. Pydantic 사용 목적

| 목적 | 설명 |
|------|------|
| **입력 검증** | Request 데이터의 타입, 필수 여부, 제약 조건 검증 |
| **출력 직렬화** | Response 데이터의 JSON 변환 및 스키마 보장 |
| **API 문서화** | FastAPI 자동 OpenAPI 문서 생성 |
| **타입 안정성** | IDE 자동완성 및 타입 힌트 지원 |

### 1.2. 스키마 구조 (v3.0)

```
app/schemas/
├── common.py          # 공통 스키마
├── text_extract.py    # 텍스트 추출 (OCR + 임베딩 통합)
├── chat.py            # LLM 응답 (대화/분석/면접 통합)
├── calendar.py        # 캘린더 관련
├── masking.py         # 마스킹 관련
└── __init__.py        # 전체 export
```

**v3.0 변경사항:**
- ❌ `ocr.py` + `embed.py` 제거 → ✅ `text_extract.py`로 통합
- ❌ `analyze.py` + `interview.py` 제거 → ✅ `chat.py`로 통합

---

## 2. LLM 출력 구조화 방식

### 2.1. LangChain Structured Output

**프로젝트에서 사용하는 방식:**

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel

# 1. Pydantic 모델 정의
class AnalysisResult(BaseModel):
    resume_summary: str
    strengths: list[str]
    match_score: int

# 2. Output Parser 생성
parser = PydanticOutputParser(pydantic_object=AnalysisResult)

# 3. LLM 호출
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
prompt = f"{parser.get_format_instructions()}\\n\\n이력서를 분석해주세요"
result = llm.invoke(prompt)

# 4. 파싱
parsed = parser.parse(result.content)
print(parsed.resume_summary)  # ✅ 타입 안전!
```

### 2.2. 프로젝트 스택

```
┌─────────────────────────────────────────────────────────────┐
│  AI Server 기술 스택                                         │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ✅ Pydantic (데이터 검증 및 직렬화)                         │
│  ✅ LangChain (LLM 오케스트레이션)                           │
│  ✅ LangGraph (상태 관리, 면접 모드)                         │
│  ✅ LangChain Output Parsers (LLM 출력 구조화)               │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

**참고:** Pydantic AI는 LangChain의 대체재이므로 사용하지 않습니다.

---

## 3. 공통 스키마

### 3.1. 기본 응답 스키마

```python
# app/schemas/common.py
from pydantic import BaseModel, Field
from typing import Optional, Any, List
from datetime import datetime
from enum import Enum


class TaskStatus(str, Enum):
    """비동기 작업 상태"""
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ErrorDetail(BaseModel):
    """에러 상세 정보"""
    code: str = Field(..., description="에러 코드", example="INVALID_REQUEST")
    message: str = Field(..., description="에러 메시지", example="필수 필드가 누락되었습니다")
    details: Optional[dict] = Field(None, description="추가 상세 정보")


class BaseResponse(BaseModel):
    """기본 응답 스키마"""
    success: bool = Field(..., description="요청 성공 여부")


class ErrorResponse(BaseResponse):
    """에러 응답 스키마"""
    success: bool = False
    error: ErrorDetail


class AsyncTaskResponse(BaseModel):
    """비동기 작업 초기 응답"""
    task_id: str = Field(..., description="비동기 작업 ID", example="task_abc123")
    status: TaskStatus = Field(default=TaskStatus.PROCESSING, description="작업 상태")


class AsyncTaskStatusResponse(BaseModel):
    """비동기 작업 상태 조회 응답"""
    task_id: str
    status: TaskStatus
    progress: Optional[int] = Field(None, ge=0, le=100, description="진행률 (%)")
    message: Optional[str] = Field(None, description="상태 메시지")
    result: Optional[Any] = Field(None, description="완료 시 결과 데이터")
    error: Optional[ErrorDetail] = Field(None, description="실패 시 에러 정보")
```

### 3.2. 스트리밍 응답 스키마

```python
# app/schemas/common.py (계속)
from typing import Literal


class StreamingChunk(BaseModel):
    """SSE 스트리밍 청크"""
    type: Literal["chunk", "complete", "error"] = Field(..., description="청크 타입")
    content: Optional[str] = Field(None, description="스트리밍 텍스트 (chunk 타입)")
    data: Optional[dict] = Field(None, description="완료 시 전체 데이터 (complete 타입)")
    error: Optional[ErrorDetail] = Field(None, description="에러 정보 (error 타입)")


# SSE 이벤트 포맷 예시
"""
data: {"type": "chunk", "content": "분석 결과..."}

data: {"type": "chunk", "content": "를 확인했습니다."}

data: {"type": "complete", "data": {"score": 85, "grade": "A"}}
"""
```

---

## 4. 텍스트 추출 + 임베딩 (통합)

### 4.1. Request

```python
# app/schemas/text_extract.py
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Literal, Optional


class TextExtractRequest(BaseModel):
    """텍스트 추출 + 임베딩 요청 (파일/텍스트 통합)"""
    
    # 문서 정보
    type: Literal["resume", "portfolio", "job_posting"] = Field(
        ..., 
        description="문서 타입"
    )
    user_id: str = Field(
        ..., 
        min_length=1,
        description="사용자 ID",
        example="user_456"
    )
    document_id: str = Field(
        ..., 
        min_length=1,
        description="문서 ID",
        example="resume_123"
    )
    
    # 입력 (file_url 또는 text 중 하나 필수)
    file_url: Optional[HttpUrl] = Field(
        None,
        description="S3 파일 URL (text와 둘 중 하나 필수)",
        example="https://s3.amazonaws.com/bucket/resume.pdf"
    )
    file_type: Optional[Literal["pdf", "image"]] = Field(
        None,
        description="파일 타입 (file_url 사용 시 필수)"
    )
    text: Optional[str] = Field(
        None,
        min_length=10,
        description="직접 입력한 텍스트 (file_url과 둘 중 하나 필수)"
    )

    @validator('text', always=True)
    def validate_input(cls, v, values):
        """file_url 또는 text 중 하나는 필수"""
        if not v and not values.get('file_url'):
            raise ValueError('file_url 또는 text 중 하나는 필수입니다')
        if v and values.get('file_url'):
            raise ValueError('file_url과 text를 동시에 사용할 수 없습니다')
        return v

    @validator('file_type', always=True)
    def validate_file_type(cls, v, values):
        """file_url 사용 시 file_type 필수"""
        if values.get('file_url') and not v:
            raise ValueError('file_url 사용 시 file_type은 필수입니다')
        return v

    @validator('file_url')
    def validate_file_url(cls, v):
        if v:
            url_str = str(v)
            if not any(ext in url_str.lower() for ext in ['.pdf', '.png', '.jpg', '.jpeg', '.webp']):
                raise ValueError('지원하지 않는 파일 형식입니다')
        return v

    @validator('text')
    def validate_text_length(cls, v):
        if v and len(v) > 100000:  # 약 25,000 토큰
            raise ValueError('텍스트가 너무 깁니다 (최대 100,000자)')
        return v

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "파일 입력",
                    "value": {
                        "type": "resume",
                        "user_id": "user_456",
                        "document_id": "resume_123",
                        "file_url": "https://s3.amazonaws.com/bucket/resume.pdf",
                        "file_type": "pdf"
                    }
                },
                {
                    "name": "텍스트 입력",
                    "value": {
                        "type": "resume",
                        "user_id": "user_456",
                        "document_id": "resume_123",
                        "text": "이름: 홍길동\n경력: 3년\n기술스택: Python, FastAPI..."
                    }
                }
            ]
        }
```

### 4.2. Response

```python
# app/schemas/text_extract.py (계속)
from typing import List, Optional


class PageText(BaseModel):
    """페이지별 텍스트"""
    page: int = Field(..., ge=1, description="페이지 번호")
    text: str = Field(..., description="페이지 텍스트")


class TextExtractResult(BaseModel):
    """텍스트 추출 결과"""
    success: bool
    extracted_text: str = Field(..., description="추출된 전체 텍스트")
    pages: Optional[List[PageText]] = Field(
        None, 
        description="페이지별 텍스트 (파일 입력 시만 제공)"
    )
    
    # ❌ VectorDB 정보 제거 (AI 서버 내부에서만 사용)
    # vector_id: str  
    # collection: str


class TextExtractResponse(AsyncTaskResponse):
    """텍스트 추출 초기 응답 (비동기)"""
    pass  # task_id, status 반환


class TextExtractStatusResponse(AsyncTaskStatusResponse):
    """텍스트 추출 상태 조회 응답"""
    result: Optional[TextExtractResult] = None
```

**💡 핵심 변경사항:**
- ✅ `file_url` 또는 `text` 중 하나 선택 가능
- ✅ VectorDB 정보(`vector_id`, `collection`) 제거
- ✅ 백엔드는 `extracted_text`만 받아서 MongoDB에 저장

---

## 5. LLM 응답 (통합)

### 5.1. 공통 Context 스키마

```python
# app/schemas/chat.py
from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Any
from enum import Enum


class ChatMode(str, Enum):
    """채팅 모드"""
    GENERAL = "general"              # 일반 대화
    ANALYSIS = "analysis"            # 이력서/채용공고 분석
    INTERVIEW_QUESTION = "interview_question"  # 면접 질문 생성
    INTERVIEW_REPORT = "interview_report"      # 면접 리포트


class ChatContext(BaseModel):
    """채팅 컨텍스트 (모드별 추가 정보)"""
    mode: ChatMode = Field(default=ChatMode.GENERAL, description="채팅 모드")
    
    # 분석 모드
    resume_id: Optional[str] = Field(None, description="이력서 ID (분석 모드)")
    posting_id: Optional[str] = Field(None, description="채용공고 ID (분석 모드)")
    
    # 면접 모드
    session_id: Optional[str] = Field(None, description="면접 세션 ID (면접 모드)")
    interview_type: Optional[Literal["technical", "personality"]] = Field(
        None, 
        description="면접 유형 (면접 모드)"
    )
    
    class Config:
        extra = 'allow'  # 추가 필드 허용
```

### 5.2. Request

```python
# app/schemas/chat.py (계속)


class ChatMessage(BaseModel):
    """채팅 메시지"""
    role: Literal["user", "assistant"] = Field(..., description="메시지 발신자")
    content: str = Field(..., description="메시지 내용")


class ToolResult(BaseModel):
    """Tool 실행 결과"""
    tool: str = Field(..., description="실행된 Tool 이름")
    success: bool = Field(..., description="실행 성공 여부")
    data: Any = Field(..., description="Tool 실행 결과 데이터")


class ChatRequest(BaseModel):
    """채팅 요청 (모든 LLM 응답 통합)"""
    room_id: str = Field(..., description="채팅방 ID")
    user_id: str = Field(..., description="사용자 ID")
    message: Optional[str] = Field(None, description="사용자 메시지")
    context: ChatContext = Field(
        default_factory=lambda: ChatContext(mode=ChatMode.GENERAL),
        description="채팅 컨텍스트"
    )
    history: List[ChatMessage] = Field(
        default=[], 
        max_length=20,
        description="대화 히스토리 (최근 20개)"
    )
    tool_result: Optional[ToolResult] = Field(
        None, 
        description="Tool 실행 결과 (Backend가 전달)"
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
                },
                {
                    "name": "면접 질문 생성",
                    "value": {
                        "room_id": "room_001",
                        "user_id": "user_456",
                        "message": "면접 질문을 생성해주세요",
                        "context": {
                            "mode": "interview_question",
                            "session_id": "session_abc123",
                            "interview_type": "technical"
                        }
                    }
                }
            ]
        }
```

### 5.3. Response (Pydantic 모델)

#### 5.3.1. 분석 응답

```python
# app/schemas/chat.py (계속)
from typing import List
from enum import Enum


class MatchGrade(str, Enum):
    """매칭 등급"""
    S = "S"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


class ResumeAnalysis(BaseModel):
    """이력서 분석 결과 (Pydantic 모델)"""
    strengths: List[str] = Field(..., description="강점 목록")
    weaknesses: List[str] = Field(..., description="약점 목록")
    suggestions: List[str] = Field(..., description="개선 제안")


class PostingAnalysis(BaseModel):
    """채용공고 분석 결과 (Pydantic 모델)"""
    company: str = Field(..., description="회사명")
    position: str = Field(..., description="포지션")
    required_skills: List[str] = Field(..., description="필수 스킬")
    preferred_skills: List[str] = Field(default=[], description="우대 스킬")


class MatchingResult(BaseModel):
    """매칭도 분석 결과 (Pydantic 모델)"""
    score: int = Field(..., ge=0, le=100, description="매칭 점수")
    grade: MatchGrade = Field(..., description="등급")
    matched_skills: List[str] = Field(..., description="매칭된 스킬")
    missing_skills: List[str] = Field(..., description="부족한 스킬")


class AnalysisResult(BaseModel):
    """분석 결과 (Pydantic 모델)"""
    resume_analysis: ResumeAnalysis
    posting_analysis: PostingAnalysis
    matching: MatchingResult
```

**LangChain 사용 예시:**
```python
from pydantic_ai import Agent

# Agent 생성 (result_type 지정)
analysis_agent = Agent('gemini-2.0-flash', result_type=AnalysisResult)

# 실행 - LLM이 자동으로 AnalysisResult 형식으로 응답
result = analysis_agent.run_sync(
    f"이력서: {resume_text}\n채용공고: {posting_text}\n분석해주세요"
)

# ✅ 타입 안전한 결과 사용
print(result.data.matching.score)  # int 보장
print(result.data.resume_analysis.strengths)  # List[str] 보장
```

#### 5.3.2. 면접 질문 응답

```python
# app/schemas/chat.py (계속)


class InterviewQuestion(BaseModel):
    """면접 질문 (Pydantic 모델)"""
    question: str = Field(..., description="생성된 질문")
    difficulty: Literal["easy", "medium", "hard"] = Field(..., description="난이도")
    category: Literal["기술", "인성", "경험"] = Field(..., description="카테고리")
    follow_up: bool = Field(default=False, description="꼬리질문 여부")
```

**LangChain 사용 예시:**
```python
from pydantic_ai import Agent

# Agent 생성
question_agent = Agent('gemini-2.0-flash', result_type=InterviewQuestion)

# 실행
result = question_agent.run_sync(
    f"이력서: {resume_text}\n기술 면접 질문을 생성해주세요"
)

# ✅ 타입 안전
print(result.data.question)      # str
print(result.data.difficulty)    # "easy" | "medium" | "hard"
print(result.data.follow_up)     # bool
```

#### 5.3.3. 면접 리포트 응답

```python
# app/schemas/chat.py (계속)


class QAEvaluation(BaseModel):
    """개별 Q&A 평가 (Pydantic 모델)"""
    question: str = Field(..., description="질문")
    answer: str = Field(..., description="답변")
    good_points: List[str] = Field(..., description="잘한 점")
    improvements: List[str] = Field(..., description="개선점")


class InterviewReport(BaseModel):
    """면접 종합 리포트 (Pydantic 모델)"""
    evaluations: List[QAEvaluation] = Field(..., description="개별 Q&A 평가")
    strength_patterns: List[str] = Field(..., description="강점 패턴")
    weakness_patterns: List[str] = Field(..., description="약점 패턴")
    learning_guide: List[str] = Field(..., description="학습 가이드")
```

**LangChain 사용 예시:**
```python
from pydantic_ai import Agent

# Agent 생성
report_agent = Agent('gemini-2.0-flash', result_type=InterviewReport)

# 실행
result = report_agent.run_sync(
    f"면접 Q&A: {qa_list}\n평가 리포트를 생성해주세요"
)

# ✅ 타입 안전
for eval in result.data.evaluations:
    print(eval.good_points)  # List[str] 보장
```

#### 5.3.4. 통합 응답 스키마

```python
# app/schemas/chat.py (계속)


class ToolCall(BaseModel):
    """Tool 호출 정보"""
    tool: Literal[
        "get_schedule", 
        "add_schedule", 
        "update_schedule", 
        "delete_schedule"
    ] = Field(..., description="호출할 Tool 이름")
    params: dict = Field(..., description="Tool 파라미터")


class ChatResponse(BaseModel):
    """채팅 응답 (통합)"""
    success: bool = True
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
    tool_used: Optional[ToolCall] = Field(None, description="실행할 Tool 정보")
```

---

## 6. 캘린더 파싱

### 6.1. Request

```python
# app/schemas/calendar.py
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Optional


class CalendarParseRequest(BaseModel):
    """캘린더 일정 파싱 요청"""
    file_url: Optional[HttpUrl] = Field(
        None, 
        description="파일 URL (text와 둘 중 하나 필수)"
    )
    text: Optional[str] = Field(
        None, 
        description="텍스트 (file_url과 둘 중 하나 필수)"
    )

    @validator('text', always=True)
    def validate_input(cls, v, values):
        if not v and not values.get('file_url'):
            raise ValueError('file_url 또는 text 중 하나는 필수입니다')
        return v
```

### 6.2. Response (Pydantic 모델)

```python
# app/schemas/calendar.py (계속)
from typing import List, Optional


class ScheduleItem(BaseModel):
    """일정 항목 (Pydantic 모델)"""
    stage: str = Field(..., description="단계 (서류마감, 코테, 면접 등)")
    date: str = Field(..., description="날짜 (YYYY-MM-DD)")
    time: Optional[str] = Field(None, description="시간 (HH:MM)")


class CalendarParseResult(BaseModel):
    """캘린더 파싱 결과 (Pydantic 모델)"""
    company: str = Field(..., description="회사명")
    position: str = Field(..., description="포지션")
    schedules: List[ScheduleItem] = Field(..., description="일정 목록")
    hashtags: List[str] = Field(..., description="자동 생성된 해시태그")


class CalendarParseResponse(BaseModel):
    """캘린더 파싱 응답"""
    success: bool = True
    result: CalendarParseResult
```

**LangChain 사용 예시:**
```python
from pydantic_ai import Agent

# Agent 생성
calendar_agent = Agent('gemini-2.0-flash', result_type=CalendarParseResult)

# 실행
result = calendar_agent.run_sync(
    f"채용공고: {text}\n일정 정보를 추출해주세요"
)

# ✅ 타입 안전
print(result.data.company)  # str
for schedule in result.data.schedules:
    print(schedule.date)  # str (YYYY-MM-DD)
```

---

## 7. 마스킹

### 7.1. Request

```python
# app/schemas/masking.py
from pydantic import BaseModel, Field, HttpUrl
from typing import Literal


class MaskingRequest(BaseModel):
    """마스킹 요청"""
    file_url: HttpUrl = Field(..., description="S3 파일 URL")
    file_type: Literal["image", "pdf"] = Field(..., description="파일 타입")
```

### 7.2. Response (Pydantic 모델)

```python
# app/schemas/masking.py (계속)
from typing import List


class PIIDetection(BaseModel):
    """개인정보 감지 결과 (Pydantic 모델)"""
    type: Literal["name", "phone", "email", "face"] = Field(..., description="개인정보 타입")
    coordinates: List[int] = Field(..., description="좌표 [x1, y1, x2, y2]")
    confidence: float = Field(..., ge=0, le=1, description="신뢰도")


class MaskingResult(BaseModel):
    """마스킹 결과 (Pydantic 모델)"""
    detected_pii: List[PIIDetection] = Field(..., description="감지된 개인정보 목록")


class MaskingResponse(AsyncTaskResponse):
    """마스킹 초기 응답"""
    pass


class MaskingStatusResponse(AsyncTaskStatusResponse):
    """마스킹 상태 조회 응답"""
    result: Optional[dict] = None  # 완료 시 마스킹된 파일 URL + 좌표
```

**LangChain 사용 예시:**
```python
from pydantic_ai import Agent

# Agent 생성
masking_agent = Agent('gemini-2.0-flash', result_type=MaskingResult)

# 실행
result = masking_agent.run_sync(
    f"이미지에서 개인정보를 감지하고 좌표를 반환해주세요"
)

# ✅ 타입 안전
for pii in result.data.detected_pii:
    print(pii.type)         # "name" | "phone" | "email" | "face"
    print(pii.coordinates)  # List[int]
    print(pii.confidence)   # float
```

---

## 8. 파일 구조

```python
# app/schemas/__init__.py
"""
Pydantic 스키마 전체 export
"""

from .common import (
    TaskStatus,
    ErrorDetail,
    BaseResponse,
    ErrorResponse,
    AsyncTaskResponse,
    AsyncTaskStatusResponse,
    StreamingChunk,
)

from .text_extract import (
    TextExtractRequest,
    TextExtractResult,
    TextExtractResponse,
    TextExtractStatusResponse,
)

from .chat import (
    ChatMode,
    ChatContext,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    # Pydantic 모델
    AnalysisResult,
    ResumeAnalysis,
    PostingAnalysis,
    MatchingResult,
    InterviewQuestion,
    InterviewReport,
    QAEvaluation,
)

from .calendar import (
    CalendarParseRequest,
    CalendarParseResult,
    CalendarParseResponse,
    ScheduleItem,
)

from .masking import (
    MaskingRequest,
    MaskingResult,
    MaskingResponse,
    MaskingStatusResponse,
    PIIDetection,
)

__all__ = [
    # Common
    "TaskStatus",
    "ErrorDetail",
    "BaseResponse",
    "ErrorResponse",
    "AsyncTaskResponse",
    "AsyncTaskStatusResponse",
    "StreamingChunk",
    
    # Text Extract
    "TextExtractRequest",
    "TextExtractResult",
    "TextExtractResponse",
    "TextExtractStatusResponse",
    
    # Chat (LLM 통합)
    "ChatMode",
    "ChatContext",
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "AnalysisResult",
    "ResumeAnalysis",
    "PostingAnalysis",
    "MatchingResult",
    "InterviewQuestion",
    "InterviewReport",
    "QAEvaluation",
    
    # Calendar
    "CalendarParseRequest",
    "CalendarParseResult",
    "CalendarParseResponse",
    "ScheduleItem",
    
    # Masking
    "MaskingRequest",
    "MaskingResult",
    "MaskingResponse",
    "MaskingStatusResponse",
    "PIIDetection",
]
```

---

## 9. LangChain 실전 예제

### 9.1. API 라우터에서 사용

```python
# app/routers/chat.py
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic_ai import Agent
from app.schemas.chat import (
    ChatRequest, 
    ChatResponse, 
    ChatMode,
    AnalysisResult,
    InterviewQuestion,
    InterviewReport,
)

router = APIRouter()

# LangChain 초기화
analysis_agent = Agent('gemini-2.0-flash', result_type=AnalysisResult)
question_agent = Agent('gemini-2.0-flash', result_type=InterviewQuestion)
report_agent = Agent('gemini-2.0-flash', result_type=InterviewReport)


@router.post("/ai/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """통합 채팅 API (모든 LLM 응답)"""
    
    mode = request.context.mode
    
    # 1. 일반 대화
    if mode == ChatMode.GENERAL:
        # LangChain 불필요 (일반 텍스트)
        response_text = await generate_chat_response(request)
        return ChatResponse(
            mode=ChatMode.GENERAL,
            response=response_text
        )
    
    # 2. 분석 모드 (LangChain 사용)
    elif mode == ChatMode.ANALYSIS:
        # VectorDB에서 이력서/채용공고 조회
        resume_text = await get_resume(request.context.resume_id)
        posting_text = await get_posting(request.context.posting_id)
        
        # LangChain으로 구조화된 분석 결과 생성
        result = analysis_agent.run_sync(
            f"이력서: {resume_text}\n채용공고: {posting_text}\n분석해주세요"
        )
        
        return ChatResponse(
            mode=ChatMode.ANALYSIS,
            analysis=result.data  # ✅ AnalysisResult 타입 보장
        )
    
    # 3. 면접 질문 생성 (LangChain 사용)
    elif mode == ChatMode.INTERVIEW_QUESTION:
        resume_text = await get_resume_by_session(request.context.session_id)
        
        # LangChain으로 구조화된 질문 생성
        result = question_agent.run_sync(
            f"이력서: {resume_text}\n{request.context.interview_type} 면접 질문 생성"
        )
        
        return ChatResponse(
            mode=ChatMode.INTERVIEW_QUESTION,
            question=result.data  # ✅ InterviewQuestion 타입 보장
        )
    
    # 4. 면접 리포트 (LangChain 사용)
    elif mode == ChatMode.INTERVIEW_REPORT:
        qa_list = await get_interview_qa(request.context.session_id)
        
        # LangChain으로 구조화된 리포트 생성
        result = report_agent.run_sync(
            f"면접 Q&A: {qa_list}\n평가 리포트를 생성해주세요"
        )
        
        return ChatResponse(
            mode=ChatMode.INTERVIEW_REPORT,
            report=result.data  # ✅ InterviewReport 타입 보장
        )
```

### 9.2. 스트리밍 + LangChain

```python
# app/routers/chat.py (계속)
from pydantic_ai import Agent
import json


@router.post("/ai/chat/stream")
async def chat_stream(request: ChatRequest):
    """스트리밍 채팅 (분석 결과는 마지막에 JSON으로)"""
    
    async def generate():
        if request.context.mode == ChatMode.ANALYSIS:
            # 1. 스트리밍으로 텍스트 설명 전송
            async for chunk in stream_analysis_explanation(request):
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk})}\n\n"
            
            # 2. LangChain으로 구조화된 결과 생성
            result = analysis_agent.run_sync(...)
            
            # 3. 마지막에 complete 이벤트로 JSON 전송
            yield f"data: {json.dumps({'type': 'complete', 'data': result.data.dict()})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")
```

### 9.3. 에러 처리

```python
# app/routers/chat.py (계속)
from pydantic import ValidationError


@router.post("/ai/chat")
async def chat(request: ChatRequest):
    try:
        # LangChain 실행
        result = analysis_agent.run_sync(prompt)
        
        return ChatResponse(
            mode=ChatMode.ANALYSIS,
            analysis=result.data
        )
        
    except ValidationError as e:
        # Pydantic 검증 에러
        raise HTTPException(
            status_code=500,
            detail={
                "code": "LLM_OUTPUT_VALIDATION_ERROR",
                "message": "LLM 응답이 예상 형식과 다릅니다",
                "details": e.errors()
            }
        )
    except Exception as e:
        # 기타 에러
        raise HTTPException(
            status_code=500,
            detail={
                "code": "LLM_ERROR",
                "message": str(e)
            }
        )
```

---

## 10. 요약

### 10.1. LangChain 사용 가이드

| API | Pydantic AI 사용 | result_type |
|-----|-----------------|-------------|
| `/ai/text/extract` | ❌ | - |
| `/ai/chat` (일반 대화) | ❌ | - |
| `/ai/chat` (분석) | ✅ | `AnalysisResult` |
| `/ai/chat` (면접 질문) | ✅ | `InterviewQuestion` |
| `/ai/chat` (면접 리포트) | ✅ | `InterviewReport` |
| `/ai/calendar/parse` | ✅ | `CalendarParseResult` |
| `/ai/masking/draft` | ✅ | `MaskingResult` |

### 10.2. 핵심 원칙

1. **LLM이 JSON을 반환해야 하는 경우** → Pydantic AI 사용
2. **LLM이 일반 텍스트를 반환하는 경우** → LangChain 불필요
3. **모든 Pydantic 모델은 BaseModel 상속**
4. **Agent 생성 시 result_type 지정**
5. **result.data로 타입 안전한 결과 접근**

---

**문의사항이 있으시면 AI 팀에 연락 주세요!** 🚀
