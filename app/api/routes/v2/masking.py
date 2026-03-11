"""
v2 PII 마스킹 API

POST /ai/masking/draft - 게시판 첨부파일 PII 마스킹
GET /ai/masking/task/{task_id} - 마스킹 작업 상태 조회
GET /ai/masking/health - 마스킹 서비스 헬스 체크

ADR-102: asyncio.create_task() → Celery 태스크로 이관하여 504 타임아웃 해결.
ADR-102 Fallback: Celery 브로커 연결 불가 시 asyncio.create_task()로 fallback.
"""

import asyncio
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from app.config.dependencies import get_legacy_task_storage
from app.schemas.common import (
    AsyncTaskResponse,
    ErrorCode,
    PollingHint,
    TaskStatus,
    TaskStatusResponse,
)
from app.schemas.masking import MaskingDraftRequest
from app.services.chandra_masking import get_chandra_masking_service
from app.services.gemini_masking import get_gemini_masking_service
from app.utils.log_sanitizer import sanitize_log_input

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/masking/draft",
    response_model=AsyncTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="게시판 첨부파일 1차 마스킹",
    description="""
    게시글 작성 시 첨부파일에서 개인정보를 자동으로 감지하고 1차 마스킹 처리합니다.

    **처리 방식:** 비동기 (VLM 처리 시간 소요)

    **지원 파일 타입:**
    - PDF (.pdf)
    - 이미지 (.png, .jpg, .jpeg)

    **감지 항목:**

    **Gemini (얼굴 전용):**
    - 얼굴 사진 (face)

    **Chandra (텍스트 전용):**
    - 이름 (name)
    - 전화번호 (phone_number)
    - 이메일 (email_address)
    - 주소 (address)
    - 대학교명 (university)
    - 학과명 (major)
    - URL (url)

    **사용 가능 모델:**
    - gemini: Google Gemini 3 Flash Preview (얼굴 감지 전용)
    - chandra: datalab-to/chandra (텍스트 PII 감지 전용)
    """,
)
async def masking_draft(
    request: MaskingDraftRequest,
    task_storage=Depends(get_legacy_task_storage),
):
    """게시판 첨부파일 마스킹

    ADR-102: Celery 태스크로 이관하여 504 타임아웃 해결.
    """
    task_id = f"task_masking_{uuid.uuid4().hex[:12]}"

    logger.info("[MASKING_DRAFT] Creating new task: %s", task_id)

    # Redis에 초기 상태 저장
    task_data = {
        "type": "masking",
        "status": TaskStatus.PROCESSING,
        "created_at": datetime.now().isoformat(),
        "progress": 0,
        "message": "마스킹 작업을 시작합니다...",
        "request": request.model_dump(),
    }
    task_storage.save(task_id, task_data)

    # ADR-102: Celery 태스크로 이관 (asyncio.create_task 대신)
    # ADR-102 Fallback: Celery 브로커 연결 불가 시 asyncio.create_task()로 fallback
    from app.tasks.celery_utils import is_celery_available

    model_value = request.model.value if hasattr(request.model, "value") else str(request.model)

    if is_celery_available():
        # Celery 태스크 비동기 실행 (즉시 반환, 블로킹 없음)
        from app.tasks.masking_tasks import process_masking_task

        process_masking_task.delay(
            task_id=task_id,
            file_url=str(request.file_url) if request.file_url else "",
            s3_key=str(request.s3_key) if request.s3_key else "",
            file_type=request.file_type,
            model=model_value,
        )
        logger.info("[Masking] Celery 태스크 등록 완료: %s", task_id)
    else:
        # Fallback: asyncio.create_task로 실행 (Celery Worker 미실행 시)
        from app.tasks.masking_tasks import _process_masking_async

        asyncio.create_task(
            _process_masking_async(
                task_id=task_id,
                file_url=str(request.file_url) if request.file_url else "",
                s3_key=str(request.s3_key) if request.s3_key else "",
                file_type=request.file_type,
                model=model_value,
            )
        )
        logger.warning(
            "[Masking] Celery 불가 → asyncio fallback: %s (Celery Worker 실행 권장)",
            task_id,
        )

    return AsyncTaskResponse(
        task_id=task_id,
        status=TaskStatus.PROCESSING,
        message=f"마스킹 작업을 시작했습니다. /ai/task/{task_id}로 진행 상태를 확인하세요.",
        polling=PollingHint(interval_ms=2000, max_attempts=150),  # 최대 5분(150 × 2초)
    )


@router.get(
    "/masking/task/{task_id}",
    response_model=TaskStatusResponse,
    summary="마스킹 작업 상태 조회",
    description="""
    비동기 마스킹 작업의 상태를 조회합니다.

    **통합 엔드포인트:** `/ai/task/{task_id}`로도 조회 가능합니다.
    """,
)
async def get_masking_task_status(
    task_id: str,
    task_storage=Depends(get_legacy_task_storage),
):
    """마스킹 작업 상태 조회"""
    safe_task_id = sanitize_log_input(task_id)
    task = task_storage.get(task_id)
    if task is None:
        logger.warning("[GET_STATUS] Task not found: %s", safe_task_id)
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.TASK_NOT_FOUND, "message": "작업을 찾을 수 없습니다."},
        )

    logger.info("[GET_STATUS] Task %s status: %s", safe_task_id, task["status"])

    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        message=task.get("message"),
        result=task.get("result"),
        error=task.get("error"),
    )


@router.get(
    "/masking/health",
    summary="PII 마스킹 서비스 헬스 체크",
    description="PII 마스킹 서비스 상태 확인",
)
async def masking_health_check():
    """PII 마스킹 서비스 헬스 체크"""
    health_status = {"status": "healthy", "service": "pii-masking", "models": {}}

    try:
        get_gemini_masking_service()
        health_status["models"]["gemini"] = {
            "status": "available",
            "provider": "Google Gemini 3 Flash Preview",
        }
    except Exception as e:
        logger.error(f"Gemini health check failed: {e}")
        health_status["models"]["gemini"] = {"status": "error", "error": str(e)}

    try:
        get_chandra_masking_service()
        health_status["models"]["chandra"] = {
            "status": "available",
            "provider": "datalab-to/chandra",
        }
    except Exception as e:
        logger.error(f"Chandra health check failed: {e}")
        health_status["models"]["chandra"] = {"status": "error", "error": str(e)}

    if all(m.get("status") == "error" for m in health_status["models"].values()):
        health_status["status"] = "error"
        health_status["message"] = "모든 마스킹 모델이 사용 불가능합니다."
    else:
        health_status["message"] = "PII 마스킹 서비스가 정상 작동 중입니다."

    return health_status
