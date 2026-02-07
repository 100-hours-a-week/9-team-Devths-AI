"""
v2 비동기 작업 상태 조회 API

GET /ai/task/{task_id} - 비동기 작업 상태 조회 (text_extract, masking 등)
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status

from app.config.dependencies import get_legacy_task_storage
from app.schemas.common import ErrorCode, TaskStatusResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/task/{task_id}",
    response_model=TaskStatusResponse,
    summary="비동기 작업 상태 조회",
    description="비동기 처리 작업의 상태를 조회하고 결과를 확인합니다.",
    responses={
        200: {
            "description": "성공",
            "content": {
                "application/json": {
                    "examples": {
                        "processing": {
                            "summary": "처리 중",
                            "value": {
                                "task_id": 32,
                                "status": "processing",
                                "progress": None,
                                "message": None,
                                "result": None,
                                "error": None,
                            },
                        },
                        "completed": {
                            "summary": "완료 (text_extract)",
                            "value": {
                                "task_id": 32,
                                "status": "completed",
                                "progress": 100,
                                "message": None,
                                "result": {
                                    "success": True,
                                    "summary": "카카오 | 백엔드 개발자",
                                    "resume_ocr": "이름: 홍길동\n경력: 3년...",
                                    "job_posting_ocr": "카카오 백엔드 채용\n자격요건: Java...",
                                    "resume_analysis": {
                                        "strengths": ["Java/Spring 숙련도", "프로젝트 경험"],
                                        "weaknesses": ["클라우드 경험 부족"],
                                        "suggestions": ["AWS 학습 권장"],
                                    },
                                    "posting_analysis": {
                                        "company": "카카오",
                                        "position": "백엔드 개발자",
                                        "required_skills": ["Java", "Spring", "MySQL"],
                                        "preferred_skills": ["Docker", "Kubernetes"],
                                    },
                                    "formatted_text": "지원 회사 및 직무 : 카카오 | 백엔드 개발자\n\n이력서 분석\n...",
                                    "room_id": 23,
                                },
                                "error": None,
                            },
                        },
                        "failed": {
                            "summary": "실패",
                            "value": {
                                "task_id": 32,
                                "status": "failed",
                                "progress": None,
                                "message": None,
                                "result": None,
                                "error": {
                                    "code": "OCR_FAILED",
                                    "message": "이미지에서 텍스트를 추출할 수 없습니다",
                                },
                            },
                        },
                    }
                }
            },
        },
        404: {
            "description": "Not Found",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "code": "TASK_NOT_FOUND",
                            "message": "작업을 찾을 수 없습니다: 12",
                        }
                    }
                }
            },
        },
    },
)
async def get_task_status(
    task_id: str,
    task_storage=Depends(get_legacy_task_storage),
):
    """통합 비동기 작업 상태 조회 (text_extract, masking 등)"""
    task_key = str(task_id)
    task = task_storage.get(task_key)

    if task is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": ErrorCode.TASK_NOT_FOUND.value,
                "message": f"작업을 찾을 수 없습니다: {task_id}",
            },
        )

    # room_id를 result에 포함
    result = task.get("result")
    if result and "room_id" not in result:
        result["room_id"] = task.get("room_id")

    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        message=task.get("message"),
        result=result,
        error=task.get("error"),
    )
