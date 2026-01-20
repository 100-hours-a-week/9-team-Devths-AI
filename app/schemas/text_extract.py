"""
텍스트 추출 + 임베딩 스키마 (API 1 + API 2 통합)
"""
from pydantic import BaseModel, Field, HttpUrl, validator
from typing import Optional, List, Literal


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
    file_url: Optional[str] = Field(
        None,
        description="파일 URL 또는 Base64 데이터 URL (text와 둘 중 하나 필수)",
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
    
    # 모델 선택 (기본값: gemini)
    model: Optional[Literal["gemini", "vllm"]] = Field(
        "gemini",
        description="사용할 모델 (gemini: Gemini Vision API OCR, vllm: pytesseract OCR + Llama)"
    )

    @validator('text', always=True)
    def validate_input(cls, v, values):
        """file_url 또는 text 중 하나는 필수"""
        if not v and not values.get('file_url'):
            raise ValueError('file_url 또는 text 중 하나는 필수입니다')
        if v and values.get('file_url'):
            raise ValueError('file_url과 text를 동시에 사용할 수 없습니다')
        return v

    @validator('file_url')
    def validate_file_url(cls, v):
        if v:
            url_str = str(v)
            # Base64 데이터 URL 허용
            if url_str.startswith('data:'):
                if not any(mime in url_str.lower() for mime in ['image/', 'application/pdf']):
                    raise ValueError('지원하지 않는 파일 형식입니다 (이미지 또는 PDF만 가능)')
            # HTTP(S) URL 검증
            elif url_str.startswith('http'):
                if not any(ext in url_str.lower() for ext in ['.pdf', '.png', '.jpg', '.jpeg', '.webp']):
                    raise ValueError('지원하지 않는 파일 형식입니다')
            else:
                raise ValueError('file_url은 HTTP URL 또는 Base64 데이터 URL이어야 합니다')
        return v

    @validator('file_type', always=True)
    def validate_and_infer_file_type(cls, v, values):
        """file_type 자동 추론 및 검증"""
        file_url = values.get('file_url')

        if not file_url:
            return v

        # file_type이 없으면 자동 추론
        if not v:
            url_str = str(file_url)
            # Base64 데이터 URL에서 타입 추론
            if url_str.startswith('data:image/'):
                return 'image'
            elif url_str.startswith('data:application/pdf'):
                return 'pdf'
            # HTTP URL에서 타입 추론
            elif any(ext in url_str.lower() for ext in ['.pdf']):
                return 'pdf'
            elif any(ext in url_str.lower() for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                return 'image'
            else:
                raise ValueError('file_url 사용 시 file_type을 명시하거나 파일 확장자가 URL에 포함되어야 합니다')

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
                        "text": "이름: 홍길동\\n경력: 3년\\n기술스택: Python, FastAPI..."
                    }
                }
            ]
        }


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
