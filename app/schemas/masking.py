from enum import Enum

from pydantic import BaseModel, Field


class MaskingFileType(str, Enum):
    """마스킹 파일 타입"""

    IMAGE = "image"
    PDF = "pdf"


class MaskingModelType(str, Enum):
    """마스킹 모델 타입"""

    GEMINI = "gemini"
    CHANDRA = "chandra"


class PIIType(str, Enum):
    """개인정보 타입"""

    NAME = "name"
    PHONE = "phone"
    PHONE_NUMBER = "phone_number"  # Chandra용
    EMAIL = "email"
    EMAIL_ADDRESS = "email_address"  # Chandra용
    ADDRESS = "address"
    FACE = "face"
    SSN = "ssn"
    CARD = "card"
    URL = "url"  # Chandra용
    UNIVERSITY = "university"  # Chandra용
    MAJOR = "major"  # Chandra용
    UNKNOWN = "unknown"


class MaskingDraftRequest(BaseModel):
    """게시판 첨부파일 마스킹 요청 (API 9)"""

    s3_key: str = Field(..., description="S3 파일 URL 또는 키")
    file_type: MaskingFileType = Field(..., description="파일 타입 (image/pdf)")
    model: MaskingModelType = Field(
        default=MaskingModelType.GEMINI,
        description="사용할 AI 모델 (gemini: Google Gemini 1.5 Flash, chandra: datalab-to/chandra)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "s3_key": "https://s3.../document.png",
                "file_type": "image",
                "model": "gemini",
            }
        }


class DetectedPII(BaseModel):
    """감지된 개인정보"""

    type: PIIType = Field(..., description="개인정보 타입")
    coordinates: list[int] = Field(
        ..., description="좌표 [x1, y1, x2, y2]", min_length=4, max_length=4
    )
    confidence: float = Field(..., ge=0, le=1, description="신뢰도 (0-1)")


class MaskingDraftResult(BaseModel):
    """마스킹 결과 (완료 시)"""

    success: bool = Field(True, description="성공 여부")
    original_url: str = Field(..., description="원본 파일 URL")
    masked_url: str = Field(..., description="마스킹된 파일 URL (data: URL 또는 http(s) URL)")
    thumbnail_url: str = Field(..., description="썸네일 URL (data: URL 또는 http(s) URL)")
    detected_pii: list[DetectedPII] = Field(..., description="감지된 개인정보 목록")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "original_url": "https://s3.amazonaws.com/bucket/document.png",
                "masked_url": "https://s3.amazonaws.com/bucket/document_masked.png",
                "thumbnail_url": "https://s3.amazonaws.com/bucket/document_masked_thumb.png",
                "detected_pii": [
                    {"type": "name", "coordinates": [100, 50, 200, 80], "confidence": 0.95},
                    {"type": "phone", "coordinates": [100, 100, 250, 130], "confidence": 0.92},
                    {"type": "email", "coordinates": [100, 150, 300, 180], "confidence": 0.98},
                    {"type": "face", "coordinates": [400, 50, 500, 180], "confidence": 0.89},
                ],
            }
        }
