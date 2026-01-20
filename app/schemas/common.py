from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from enum import Enum


class ErrorCode(str, Enum):
    """에러 코드 열거형"""
    INVALID_REQUEST = "INVALID_REQUEST"
    UNAUTHORIZED = "UNAUTHORIZED"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    OCR_ERROR = "OCR_ERROR"
    LLM_ERROR = "LLM_ERROR"
    VECTORDB_ERROR = "VECTORDB_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"
    MASKING_ERROR = "MASKING_ERROR"


class ErrorDetail(BaseModel):
    """에러 상세 정보"""
    code: ErrorCode = Field(..., description="에러 코드")
    message: str = Field(..., description="에러 메시지")
    details: Optional[Dict[str, Any]] = Field(None, description="추가 정보")


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
    task_id: str = Field(..., description="작업 ID")
    status: TaskStatus = Field(TaskStatus.PROCESSING, description="작업 상태")
    message: Optional[str] = Field(None, description="상태 메시지")


class TaskStatusResponse(BaseModel):
    """비동기 작업 상태 조회 응답 (처리 중)"""
    task_id: str = Field(..., description="작업 ID")
    status: TaskStatus = Field(..., description="작업 상태")
    progress: Optional[int] = Field(None, ge=0, le=100, description="진행률 (0-100)")
    message: Optional[str] = Field(None, description="상태 메시지")
    result: Optional[Dict[str, Any]] = Field(None, description="완료 시 결과")
    error: Optional[ErrorDetail] = Field(None, description="실패 시 에러 정보")


class StreamChunk(BaseModel):
    """스트리밍 청크"""
    type: str = Field("chunk", description="이벤트 타입")
    content: str = Field(..., description="텍스트 청크")


class StreamComplete(BaseModel):
    """스트리밍 완료"""
    type: str = Field("complete", description="이벤트 타입")
    data: Dict[str, Any] = Field(..., description="전체 JSON 응답")


class StreamError(BaseModel):
    """스트리밍 에러"""
    type: str = Field("error", description="이벤트 타입")
    error: ErrorDetail = Field(..., description="에러 정보")
