from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ErrorCode(str, Enum):
    """에러 코드 열거형"""

    INVALID_REQUEST = "INVALID_REQUEST"
    INVALID_TASK_ID = "INVALID_TASK_ID"
    UNAUTHORIZED = "UNAUTHORIZED"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    TASK_EXPIRED = "TASK_EXPIRED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    OCR_ERROR = "OCR_ERROR"
    OCR_FAILED = "OCR_FAILED"
    LLM_ERROR = "LLM_ERROR"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    VECTORDB_ERROR = "VECTORDB_ERROR"
    VECTORDB_UNAVAILABLE = "VECTORDB_UNAVAILABLE"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    MASKING_ERROR = "MASKING_ERROR"
    MASKING_FAILED = "MASKING_FAILED"
    INVALID_FILE_TYPE = "INVALID_FILE_TYPE"
    INVALID_DOCUMENT = "INVALID_DOCUMENT"
    INVALID_URL = "INVALID_URL"
    INVALID_MODE = "INVALID_MODE"
    INVALID_INTERVIEW_TYPE = "INVALID_INTERVIEW_TYPE"
    MISSING_CONTEXT = "MISSING_CONTEXT"
    EMPTY_MESSAGE = "EMPTY_MESSAGE"
    HISTORY_TOO_LONG = "HISTORY_TOO_LONG"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    STREAM_ERROR = "STREAM_ERROR"
    PARSE_FAILED = "PARSE_FAILED"
    NO_SCHEDULE_FOUND = "NO_SCHEDULE_FOUND"
    S3_UNAVAILABLE = "S3_UNAVAILABLE"
    PROCESSING_ERROR = "PROCESSING_ERROR"


class ErrorDetail(BaseModel):
    """에러 상세 정보"""

    code: ErrorCode = Field(..., description="에러 코드")
    message: str = Field(..., description="에러 메시지")
    details: dict[str, Any] | None = Field(None, description="추가 정보")


class ErrorResponse(BaseModel):
    """에러 응답"""

    success: bool = Field(False, description="성공 여부")
    error: ErrorDetail = Field(..., description="에러 정보")


class TaskStatus(str, Enum):
    """비동기 작업 상태"""

    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AsyncTaskResponse(BaseModel):
    """비동기 작업 초기 응답"""

    task_id: str | int = Field(..., description="작업 ID (text_extract: int, masking: str)")
    status: TaskStatus = Field(TaskStatus.PROCESSING, description="작업 상태")
    message: str | None = Field(None, description="상태 메시지")


class TaskStatusResponse(BaseModel):
    """비동기 작업 상태 조회 응답 (통합)"""

    task_id: str | int = Field(..., description="작업 ID (text_extract: int, masking: str)")
    status: TaskStatus = Field(..., description="작업 상태")
    progress: int | None = Field(None, ge=0, le=100, description="진행률 (0-100)")
    message: str | None = Field(None, description="상태 메시지")
    result: dict[str, Any] | None = Field(None, description="완료 시 결과")
    error: dict[str, Any] | None = Field(None, description="실패 시 에러 정보")


class StreamChunk(BaseModel):
    """스트리밍 청크"""

    type: str = Field("chunk", description="이벤트 타입")
    content: str = Field(..., description="텍스트 청크")


class StreamComplete(BaseModel):
    """스트리밍 완료"""

    type: str = Field("complete", description="이벤트 타입")
    data: dict[str, Any] = Field(..., description="전체 JSON 응답")


class StreamError(BaseModel):
    """스트리밍 에러"""

    type: str = Field("error", description="이벤트 타입")
    error: ErrorDetail = Field(..., description="에러 정보")
