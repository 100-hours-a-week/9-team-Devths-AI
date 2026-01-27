from pydantic import BaseModel, Field, validator


class CalendarParseRequest(BaseModel):
    """캘린더 일정 파싱 요청 (API 8)"""

    s3_key: str | None = Field(None, description="S3 파일 URL 또는 키 (선택)")
    text: str | None = Field(None, description="채용공고 텍스트 (선택)")

    @validator("text", always=True)
    def validate_input_source(cls, v, values):
        """s3_key 또는 text 중 하나는 필수"""
        s3_key = values.get("s3_key")

        # 둘 다 없으면 에러
        if not v and not s3_key:
            raise ValueError("s3_key 또는 text 중 하나는 필수입니다")

        return v

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "name": "파일 업로드",
                    "value": {"s3_key": "https://s3.../job_posting.png", "text": None},
                },
                {
                    "name": "텍스트 입력",
                    "value": {
                        "s3_key": None,
                        "text": "카카오 백엔드 개발자 채용\n서류마감: 2026-01-15\n코딩테스트: 2026-01-20...",
                    },
                },
            ]
        }


class ScheduleStage(BaseModel):
    """일정 단계"""

    stage: str = Field(..., description="단계 (서류마감, 코딩테스트, 1차면접, 2차면접, 최종발표)")
    date: str = Field(..., description="일정 날짜 (YYYY-MM-DD)")
    time: str | None = Field(None, description="시간 (HH:MM)")


class CalendarParseResponse(BaseModel):
    """캘린더 일정 파싱 응답 (API 8)"""

    success: bool = Field(True, description="성공 여부")
    company: str = Field(..., description="회사명")
    position: str = Field(..., description="포지션")
    schedules: list[ScheduleStage] = Field(..., description="일정 목록")
    hashtags: list[str] = Field(..., description="해시태그")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "company": "카카오",
                "position": "백엔드 개발자",
                "schedules": [
                    {"stage": "서류 마감", "date": "2026-01-15", "time": None},
                    {"stage": "코딩테스트", "date": "2026-01-20", "time": "14:00"},
                    {"stage": "1차 면접", "date": "2026-01-25", "time": None},
                ],
                "hashtags": ["#카카오", "#백엔드", "#신입"],
            }
        }
