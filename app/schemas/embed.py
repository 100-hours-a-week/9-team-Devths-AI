from pydantic import BaseModel, Field
from enum import Enum


class EmbedType(str, Enum):
    """임베딩 타입"""
    RESUME = "resume"
    PORTFOLIO = "portfolio"
    JOB_POSTING = "job_posting"


class FileEmbedRequest(BaseModel):
    """텍스트 임베딩 저장 요청 (API 2)"""
    type: EmbedType = Field(..., description="문서 타입 (resume/portfolio/job_posting)")
    id: str = Field(..., description="문서 ID (resume_id, portfolio_id, posting_id)")
    user_id: str = Field(..., description="사용자 ID")
    text: str = Field(..., description="마스킹된 텍스트")

    class Config:
        json_schema_extra = {
            "example": {
                "type": "resume",
                "id": "resume_123",
                "user_id": "user_456",
                "text": "마스킹된 이력서 텍스트..."
            }
        }


class FileEmbedResponse(BaseModel):
    """텍스트 임베딩 저장 응답 (API 2)"""
    success: bool = Field(True, description="성공 여부")
    type: str = Field(..., description="문서 타입")
    id: str = Field(..., description="문서 ID")
    vector_id: str = Field(..., description="VectorDB에 저장된 ID")
    collection: str = Field(..., description="VectorDB 컬렉션 이름")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "type": "resume",
                "id": "resume_123",
                "vector_id": "vec_abc123",
                "collection": "resumes"
            }
        }
