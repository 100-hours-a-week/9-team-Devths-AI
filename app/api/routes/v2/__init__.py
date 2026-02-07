"""
API v2 routes - 도메인별 분리된 모듈화 엔드포인트

v2 라우트는 기존 /ai prefix를 유지하며 도메인별로 파일이 분리되어 있습니다.
"""

from fastapi import APIRouter

from app.api.routes.v2 import calendar, chat, masking, task, text_extract

router = APIRouter(
    prefix="/ai",
    tags=["AI APIs v2"],
    responses={404: {"description": "Not found"}},
)

# 도메인별 라우터 통합
router.include_router(text_extract.router, tags=["Text Extract"])
router.include_router(task.router, tags=["Task Status"])
router.include_router(chat.router, tags=["Chat"])
router.include_router(calendar.router, tags=["Calendar"])
router.include_router(masking.router, tags=["PII Masking"])
