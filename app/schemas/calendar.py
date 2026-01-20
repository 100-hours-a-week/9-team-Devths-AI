from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional


class CalendarParseRequest(BaseModel):
    """캘린더 일정 파싱 요청 (API 8)"""
    file_url: Optional[HttpUrl] = Field(None, description="파일 URL (선택)")
    text: Optional[str] = Field(None, description="채용공고 텍스트 (선택)")

    class Config:
        json_schema_extra = {
            "example": {
                "file_url": "https://s3.amazonaws.com/bucket/job_posting.png",
                "text": None
            }
        }


class ScheduleStage(BaseModel):
    """일정 단계"""
    stage: str = Field(..., description="단계 (서류마감, 코딩테스트, 1차면접, 2차면접, 최종발표)")
    date: str = Field(..., description="일정 날짜 (YYYY-MM-DD)")
    time: Optional[str] = Field(None, description="시간 (HH:MM)")


class CalendarParseResponse(BaseModel):
    """캘린더 일정 파싱 응답 (API 8)"""
    success: bool = Field(True, description="성공 여부")
    company: str = Field(..., description="회사명")
    position: str = Field(..., description="포지션")
    schedules: List[ScheduleStage] = Field(..., description="일정 목록")
    hashtags: List[str] = Field(..., description="해시태그")

    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "company": "카카오",
                "position": "백엔드 개발자",
                "schedules": [
                    {"stage": "서류 마감", "date": "2026-01-15", "time": None},
                    {"stage": "코딩테스트", "date": "2026-01-20", "time": "14:00"},
                    {"stage": "1차 면접", "date": "2026-01-25", "time": None}
                ],
                "hashtags": ["#카카오", "#백엔드", "#신입"]
            }
        }
