"""
v2 PII 마스킹 API

POST /ai/masking/draft - 게시판 첨부파일 PII 마스킹
GET /ai/masking/task/{task_id} - 마스킹 작업 상태 조회
GET /ai/masking/health - 마스킹 서비스 헬스 체크
"""

import asyncio
import base64
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, status

from app.config.dependencies import get_legacy_task_storage
from app.schemas.common import AsyncTaskResponse, ErrorCode, TaskStatus, TaskStatusResponse
from app.schemas.masking import (
    DetectedPII,
    MaskingDraftRequest,
    MaskingDraftResult,
    MaskingModelType,
)
from app.services.chandra_masking import get_chandra_masking_service
from app.services.gemini_masking import get_gemini_masking_service
from app.utils.log_sanitizer import sanitize_log_input

logger = logging.getLogger(__name__)

router = APIRouter()

# 백그라운드 작업 추적 (가비지 컬렉션 방지)
_background_tasks_set = set()


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
    """게시판 첨부파일 마스킹"""
    task_id = f"task_masking_{uuid.uuid4().hex[:12]}"

    logger.info("[MASKING_DRAFT] Creating new task: %s", task_id)

    task_data = {
        "type": "masking",
        "status": TaskStatus.PROCESSING,
        "created_at": datetime.now(),
        "progress": 0,
        "message": "마스킹 작업을 시작합니다...",
        "request": request.model_dump(),
    }
    task_storage.save(task_id, task_data)

    async def process_masking(store):
        logger.info("[PROCESS_MASKING] Starting masking task %s", task_id)
        logger.info(f"[PROCESS_MASKING] Using model: {request.model}")
        try:
            if request.model == MaskingModelType.CHANDRA:
                service = get_chandra_masking_service()
                model_name = "Chandra"
            else:
                service = get_gemini_masking_service()
                model_name = "Gemini"

            task_data = store.get(task_id)
            task_data["progress"] = 10
            task_data["message"] = "파일을 다운로드 중입니다..."
            store.save(task_id, task_data)
            logger.info(f"Task {task_id}: Downloading file")

            if request.file_type == "pdf":
                task_data = store.get(task_id)
                task_data["progress"] = 30
                task_data["message"] = "PDF를 이미지로 변환 중입니다..."
                store.save(task_id, task_data)

                task_data = store.get(task_id)
                task_data["progress"] = 50
                task_data["message"] = f"{model_name} API로 PII를 감지 중입니다..."
                store.save(task_id, task_data)

                masked_bytes, thumbnail_bytes, detections = await service.mask_pdf(
                    file_url=str(request.file_url)
                )
            else:
                task_data = store.get(task_id)
                task_data["progress"] = 30
                task_data["message"] = "이미지에서 PII를 감지 중입니다..."
                store.save(task_id, task_data)

                task_data = store.get(task_id)
                task_data["progress"] = 50
                task_data["message"] = f"{model_name} API로 PII를 감지 중입니다..."
                store.save(task_id, task_data)

                masked_bytes, thumbnail_bytes, detections = await service.mask_image_file(
                    file_url=str(request.s3_key)
                )

            task_data = store.get(task_id)
            task_data["progress"] = 80
            task_data["message"] = "마스킹된 파일을 저장 중입니다..."
            store.save(task_id, task_data)

            masked_base64 = base64.b64encode(masked_bytes).decode("utf-8")
            thumbnail_base64 = base64.b64encode(thumbnail_bytes).decode("utf-8")

            file_ext = "pdf" if request.file_type == "pdf" else "png"
            mime_type = f"application/{file_ext}" if request.file_type == "pdf" else "image/png"

            masked_url = f"data:{mime_type};base64,{masked_base64}"
            thumbnail_url = f"data:image/png;base64,{thumbnail_base64}"

            detected_pii = []
            for det in detections:
                pii_type = det.get("type", "unknown")
                try:
                    detected_pii.append(
                        DetectedPII(
                            type=pii_type,
                            coordinates=det.get("coordinates", [0, 0, 0, 0]),
                            confidence=det.get("confidence", 0.0),
                        )
                    )
                except ValueError:
                    logger.warning(f"Unknown PII type: {pii_type}")

            task_data = store.get(task_id)
            task_data["status"] = TaskStatus.COMPLETED
            task_data["progress"] = 100
            task_data["message"] = "마스킹 작업이 완료되었습니다."
            task_data["result"] = MaskingDraftResult(
                success=True,
                original_url=request.s3_key,
                masked_url=masked_url,
                thumbnail_url=thumbnail_url,
                detected_pii=detected_pii,
            ).model_dump()
            store.save(task_id, task_data)

            logger.info(f"Task {task_id} completed with {len(detected_pii)} PII detections")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
            task_data = store.get(task_id) or {}
            task_data["status"] = TaskStatus.FAILED
            task_data["message"] = f"마스킹 작업 실패: {str(e)}"
            task_data["error"] = {"code": ErrorCode.PROCESSING_ERROR, "message": str(e)}
            store.save(task_id, task_data)

    task = asyncio.create_task(process_masking(task_storage))
    _background_tasks_set.add(task)
    task.add_done_callback(lambda t: _background_tasks_set.discard(t))

    logger.info(f"Created background task {task_id}, current tasks: {len(_background_tasks_set)}")

    return AsyncTaskResponse(
        task_id=task_id,
        status=TaskStatus.PROCESSING,
        message="마스킹 작업을 시작했습니다. /ai/task/{task_id}로 진행 상태를 확인하세요.",
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
