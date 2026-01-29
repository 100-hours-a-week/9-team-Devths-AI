"""
텍스트 추출 + 임베딩 스키마 (API 1 + API 2 통합)
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, validator


class DocumentInput(BaseModel):
    """문서 입력 (이력서 또는 채용공고)

    입력 방식:
    1. 파일 업로드: file_id + s3_key + file_type
    2. 텍스트 입력: text
    3. 혼합 불가: s3_key와 text를 동시에 사용할 수 없음
    """

    file_id: int | None = Field(None, description="파일 ID (파일 업로드 시)", example=23)
    s3_key: str | None = Field(
        None,
        description="S3 파일 키 또는 URL (예: 'uploads/2026/01/xxx.png' 또는 'https://...')",
        example="uploads/2026/01/9eb3907b-xxx.pdf",
    )
    file_type: str | None = Field(
        None,
        description="파일 타입 MIME 형식 (s3_key 사용 시 필수) - application/pdf, image/png, image/jpeg 등",
        example="application/pdf",
    )
    text: str | None = Field(
        None,
        description="직접 입력한 텍스트 (텍스트 입력 시)",
        example="이름: 홍길동\n경력: 3년\n기술스택: React, TypeScript...",
    )

    @validator("text", always=True)
    def validate_input_source(cls, v, values):
        """s3_key 또는 text 중 하나는 필수 (둘 다 있으면 s3_key 우선)"""
        s3_key = values.get("s3_key")

        # 둘 다 없으면 에러
        if not v and not s3_key:
            raise ValueError("s3_key 또는 text 중 하나는 필수입니다")

        # 둘 다 있으면 s3_key 우선, text는 무시
        if v and s3_key:
            return None  # s3_key가 있으면 text는 무시

        return v

    @validator("file_type")
    def validate_file_type(cls, v, values):
        """s3_key 사용 시 file_type 필수, MIME 타입 허용"""
        s3_key = values.get("s3_key")
        if s3_key and not v:
            raise ValueError("s3_key 사용 시 file_type은 필수입니다")
        if v:
            # MIME 타입 검증: application/pdf, image/png, image/jpeg 등
            valid_mimes = [
                "application/pdf",
                "image/png",
                "image/jpeg",
                "image/jpg",
                "image/gif",
                "image/bmp",
                "image/webp",
            ]
            # 또는 단순 타입 (pdf, image)도 허용 (하위 호환성)
            simple_types = ["pdf", "image"]

            if v not in valid_mimes and v not in simple_types:
                raise ValueError("file_type은 pdf 또는 image만 가능합니다")
        return v

    def get_file_type_simple(self) -> str | None:
        """MIME 타입을 단순 타입(pdf/image)으로 변환"""
        if not self.file_type:
            return None

        # 이미 단순 타입이면 그대로 반환
        if self.file_type in ["pdf", "image"]:
            return self.file_type

        # MIME 타입을 단순 타입으로 변환
        if "pdf" in self.file_type.lower():
            return "pdf"
        elif "image" in self.file_type.lower():
            return "image"

        return None

    @validator("s3_key")
    def validate_s3_key(cls, v):
        """s3_key 형식 검증"""
        if v:
            url_str = str(v)
            # S3 URL 또는 키 형식 검증
            if not (url_str.startswith("http") or url_str.startswith("s3://") or "/" in url_str):
                raise ValueError("s3_key는 S3 URL 또는 키 형식이어야 합니다")
        return v

    @validator("text")
    def validate_text_length(cls, v):
        """텍스트 길이 검증"""
        if v and len(v) > 100000:  # 약 25,000 토큰
            raise ValueError("텍스트가 너무 깁니다 (최대 100,000자)")
        return v


class TextExtractRequest(BaseModel):
    """텍스트 추출 + 임베딩 요청 (이력서 + 채용공고)

    resume과 job_posting은 필수 입력해야 합니다.
    각 문서는 파일 업로드 또는 텍스트 입력 중 하나의 방식으로 제공됩니다.
    """

    task_id: int = Field(..., description="작업 ID (백엔드에서 생성)", example=1)
    room_id: int = Field(..., description="채팅방 ID", example=23)
    user_id: int = Field(..., description="사용자 ID", example=12)
    resume: DocumentInput = Field(..., description="이력서/포트폴리오 입력")
    job_posting: DocumentInput = Field(..., description="채용공고 입력")
    model: Literal["gemini", "vllm"] | None = Field(
        "gemini", description="사용할 모델 (gemini: Gemini Vision API OCR, vllm: EasyOCR + Llama)"
    )

    @validator("resume", "job_posting")
    def validate_documents(cls, v):
        """resume과 job_posting은 필수"""
        if not v:
            raise ValueError("resume과 job_posting은 필수 입력해야 합니다")
        return v

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "파일 업로드 방식 (S3 key)",
                    "value": {
                        "task_id": 1,
                        "model": "gemini",
                        "room_id": 23,
                        "user_id": 12,
                        "resume": {
                            "file_id": 23,
                            "s3_key": "uploads/2026/01/9eb3907b-resume.pdf",
                            "file_type": "application/pdf",
                            "text": None,
                        },
                        "job_posting": {
                            "file_id": 24,
                            "s3_key": "uploads/2026/01/abc123-job_posting.png",
                            "file_type": "image/png",
                            "text": None,
                        },
                    },
                },
                {
                    "name": "텍스트 직접 입력 방식",
                    "value": {
                        "task_id": 2,
                        "model": "gemini",
                        "room_id": 23,
                        "user_id": 12,
                        "resume": {
                            "file_id": None,
                            "s3_key": None,
                            "file_type": None,
                            "text": "이름: 홍길동\\n경력: 3년\\n기술스택: React, TypeScript...",
                        },
                        "job_posting": {
                            "file_id": None,
                            "s3_key": None,
                            "file_type": None,
                            "text": "카카오 백엔드 개발자 채용\\n자격요건: Java, Spring...",
                        },
                    },
                },
                {
                    "name": "혼합 방식 (S3 key + 텍스트)",
                    "value": {
                        "task_id": 3,
                        "model": "gemini",
                        "room_id": 23,
                        "user_id": 12,
                        "resume": {
                            "file_id": 23,
                            "s3_key": "uploads/2026/01/9eb3907b-resume.pdf",
                            "file_type": "application/pdf",
                            "text": None,
                        },
                        "job_posting": {
                            "file_id": None,
                            "s3_key": None,
                            "file_type": None,
                            "text": "카카오 백엔드 개발자 채용\\n자격요건: Java, Spring...",
                        },
                    },
                },
            ]
        }


class PageText(BaseModel):
    """페이지별 텍스트"""

    page: int = Field(..., ge=1, description="페이지 번호")
    text: str = Field(..., description="페이지 텍스트")


class DocumentExtractResult(BaseModel):
    """문서별 텍스트 추출 결과"""

    file_id: int | None = Field(None, description="파일 ID")
    extracted_text: str = Field(..., description="추출된 전체 텍스트")
    pages: list[PageText] | None = Field(None, description="페이지별 텍스트 (파일 입력 시만 제공)")


class TextExtractResult(BaseModel):
    """텍스트 추출 + 분석 결과 (이력서 + 채용공고)

    명세서에 따른 응답 구조:
    - resume_ocr: 이력서 OCR 텍스트
    - job_posting_ocr: 채용공고 OCR 텍스트
    - resume_analysis: 이력서 분석 결과
    - posting_analysis: 채용공고 분석 결과
    """

    success: bool = Field(True, description="성공 여부")
    resume_ocr: str = Field(..., description="이력서 OCR 텍스트")
    job_posting_ocr: str = Field(..., description="채용공고 OCR 텍스트")
    resume_analysis: dict[str, Any] | None = Field(
        None, description="이력서 분석 결과 (strengths, weaknesses, suggestions)"
    )
    posting_analysis: dict[str, Any] | None = Field(
        None,
        description="채용공고 분석 결과 (company, position, required_skills, preferred_skills)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "resume_ocr": "이름: 홍길동\n경력: 3년\n기술스택: React, TypeScript...",
                "job_posting_ocr": "카카오 백엔드 개발자 채용\n자격요건: Java, Spring...",
                "resume_analysis": {
                    "strengths": ["React 숙련도", "프로젝트 경험"],
                    "weaknesses": ["백엔드 경험 부족"],
                    "suggestions": ["Spring 학습 권장"],
                },
                "posting_analysis": {
                    "company": "카카오",
                    "position": "백엔드 개발자",
                    "required_skills": ["Java", "Spring", "MySQL"],
                    "preferred_skills": ["Docker", "Kubernetes"],
                },
            }
        }
