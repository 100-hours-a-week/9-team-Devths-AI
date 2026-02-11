"""
v2 캘린더 일정 파싱 API

POST /ai/calendar/parse - 캘린더 일정 정보 추출
"""

from fastapi import APIRouter

from app.schemas.calendar import CalendarParseRequest, CalendarParseResponse

router = APIRouter()


@router.post(
    "/calendar/parse",
    response_model=CalendarParseResponse,
    summary="캘린더 일정 정보 파싱",
    description="""
    캘린더 모달에서 파일/텍스트를 분석하여 일정 정보를 추출합니다 (폼 자동 채우기용).

    **처리 방식:** 동기 - 간단한 파싱 작업

    **Pydantic AI 사용:** CalendarParseResult

    **사용 시나리오:**
    - 모달에서 채용공고 파일/텍스트 첨부 → 일정 정보 추출
    - Frontend가 모달 폼에 자동 채워넣음
    - 사용자 확인/수정 → 저장 → Backend가 Google Calendar에 추가
    """,
)
async def calendar_parse(request: CalendarParseRequest):  # noqa: ARG001
    """캘린더 일정 파싱"""
    return CalendarParseResponse(
        success=True,
        company="카카오",
        position="백엔드 개발자",
        schedules=[
            {"stage": "서류 마감", "date": "2026-01-15", "time": None},
            {"stage": "코딩테스트", "date": "2026-01-20", "time": "14:00"},
            {"stage": "1차 면접", "date": "2026-01-25", "time": None},
        ],
        hashtags=["#카카오", "#백엔드", "#신입"],
    )
