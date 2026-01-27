"""
Frontend 직접 호출용 스키마 (데모/테스트용)
"""

from pydantic import BaseModel, Field


class FrontendDirectRequest(BaseModel):
    """Frontend에서 직접 호출하는 요청 형식 (데모용)"""

    file_content: str = Field(..., description="Base64 인코딩된 파일 내용")
    type: str = Field(..., description="문서 타입 (resume 또는 job_posting)")
    user_id: str = Field(..., description="사용자 ID (문자열)")
    document_id: str = Field(..., description="문서 ID")
    model: str = Field("gemini", description="사용할 모델")

    class Config:
        json_schema_extra = {
            "example": {
                "file_content": "JVBERi0xLjQK...",
                "type": "resume",
                "user_id": "user_1769324207140",
                "document_id": "doc_1769324213520",
                "model": "gemini",
            }
        }
