from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional
from enum import Enum


class FileType(str, Enum):
    """파일 타입"""
    PDF = "pdf"
    IMAGE = "image"


class DocumentType(str, Enum):
    """문서 타입"""
    RESUME = "resume"
    PORTFOLIO = "portfolio"
    JOB_POSTING = "job_posting"


class OCRExtractRequest(BaseModel):
    """OCR 텍스트 추출 요청 (API 1)"""
    file_url: HttpUrl = Field(..., description="파일 URL (S3 등)")
    file_type: FileType = Field(..., description="파일 형식 (pdf/image)")
    type: DocumentType = Field(..., description="문서 타입 (resume/portfolio/job_posting)")
    user_id: str = Field(..., description="사용자 ID")
    document_id: str = Field(..., description="문서 ID (resume_id, portfolio_id, posting_id)")

    class Config:
        json_schema_extra = {
            "example": {
                "file_url": "https://s3.amazonaws.com/bucket/resume.pdf",
                "file_type": "pdf",
                "type": "resume",
                "user_id": "user_456",
                "document_id": "resume_123"
            }
        }


class PageText(BaseModel):
    """페이지별 텍스트"""
    page: int = Field(..., ge=1, description="페이지 번호")
    text: str = Field(..., description="페이지 텍스트")


class OCRExtractResult(BaseModel):
    """OCR 추출 결과 (완료 시)"""
    success: bool = Field(True, description="성공 여부")
    extracted_text: str = Field(..., description="추출된 전체 텍스트")
    pages: List[PageText] = Field(..., description="페이지별 텍스트")
    vector_id: str = Field(..., description="VectorDB에 저장된 ID")
    collection: str = Field(..., description="VectorDB 컬렉션 이름")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "extracted_text": "추출된 텍스트 전체...",
                "pages": [
                    {"page": 1, "text": "1페이지 텍스트..."},
                    {"page": 2, "text": "2페이지 텍스트..."}
                ],
                "vector_id": "vec_abc123",
                "collection": "resumes"
            }
        }
