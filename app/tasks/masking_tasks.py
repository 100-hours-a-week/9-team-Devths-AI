"""
Masking Tasks — ADR-102.

Celery 태스크: PII 마스킹 처리.
asyncio.create_task() 대신 Celery Worker에서 실행하여 504 타임아웃 해결.
"""

import asyncio
import base64
import logging

from app.tasks.celery_app import celery_app
from app.utils.log_sanitizer import sanitize_log_input

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.masking_tasks.process_masking_task")
def process_masking_task(
    self,
    task_id: str,
    file_url: str,
    s3_key: str,
    file_type: str,
    model: str,
):
    """PII 마스킹 Celery 태스크 (ADR-102).

    Args:
        task_id: 태스크 ID
        file_url: 파일 URL
        s3_key: S3 키
        file_type: 파일 타입 (pdf, image)
        model: 마스킹 모델 (gemini, chandra)
    """
    safe_task_id = sanitize_log_input(task_id)
    logger.info("[MaskingTask] 태스크 시작: %s", safe_task_id)

    try:
        result = asyncio.run(
            _process_masking_async(
                task_id=task_id,
                file_url=file_url,
                s3_key=s3_key,
                file_type=file_type,
                model=model,
            )
        )
        logger.info("[MaskingTask] 태스크 완료: %s", safe_task_id)
        return result

    except Exception as e:
        logger.error("[MaskingTask] 태스크 실패: %s — %s", safe_task_id, type(e).__name__)
        raise self.retry(exc=e, countdown=60, max_retries=2) from e


async def _process_masking_async(
    task_id: str,
    file_url: str,
    s3_key: str,
    file_type: str,
    model: str,
) -> dict:
    """비동기 마스킹 로직 (기존 process_masking 로직 이관)."""
    from app.config.dependencies import get_legacy_task_storage
    from app.schemas.common import ErrorCode, TaskStatus
    from app.schemas.masking import DetectedPII, MaskingDraftResult, MaskingModelType
    from app.services.chandra_masking import get_chandra_masking_service
    from app.services.gemini_masking import get_gemini_masking_service

    task_storage = get_legacy_task_storage()

    logger.info("[PROCESS_MASKING] Starting masking task %s", task_id)
    logger.info("[PROCESS_MASKING] Using model: %s", model)

    try:
        if model == MaskingModelType.CHANDRA or model == "chandra":
            service = get_chandra_masking_service()
            model_name = "Chandra"
        else:
            service = get_gemini_masking_service()
            model_name = "Gemini"

        task_data = task_storage.get(task_id) or {}
        task_data["progress"] = 10
        task_data["message"] = "파일을 다운로드 중입니다..."
        task_storage.save(task_id, task_data)
        logger.info("Task %s: Downloading file", task_id)

        if file_type == "pdf":
            task_data = task_storage.get(task_id) or {}
            task_data["progress"] = 30
            task_data["message"] = "PDF를 이미지로 변환 중입니다..."
            task_storage.save(task_id, task_data)

            task_data = task_storage.get(task_id) or {}
            task_data["progress"] = 50
            task_data["message"] = f"{model_name} API로 PII를 감지 중입니다..."
            task_storage.save(task_id, task_data)

            masked_bytes, thumbnail_bytes, detections = await service.mask_pdf(
                file_url=str(file_url)
            )
        else:
            task_data = task_storage.get(task_id) or {}
            task_data["progress"] = 30
            task_data["message"] = "이미지에서 PII를 감지 중입니다..."
            task_storage.save(task_id, task_data)

            task_data = task_storage.get(task_id) or {}
            task_data["progress"] = 50
            task_data["message"] = f"{model_name} API로 PII를 감지 중입니다..."
            task_storage.save(task_id, task_data)

            masked_bytes, thumbnail_bytes, detections = await service.mask_image_file(
                file_url=str(s3_key)
            )

        task_data = task_storage.get(task_id) or {}
        task_data["progress"] = 80
        task_data["message"] = "마스킹된 파일을 저장 중입니다..."
        task_storage.save(task_id, task_data)

        masked_base64 = base64.b64encode(masked_bytes).decode("utf-8")
        thumbnail_base64 = base64.b64encode(thumbnail_bytes).decode("utf-8")

        file_ext = "pdf" if file_type == "pdf" else "png"
        mime_type = f"application/{file_ext}" if file_type == "pdf" else "image/png"

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
                logger.warning("Unknown PII type: %s", pii_type)

        task_data = task_storage.get(task_id) or {}
        task_data["status"] = TaskStatus.COMPLETED
        task_data["progress"] = 100
        task_data["message"] = "마스킹 작업이 완료되었습니다."
        task_data["result"] = MaskingDraftResult(
            success=True,
            original_url=s3_key,
            masked_url=masked_url,
            thumbnail_url=thumbnail_url,
            detected_pii=detected_pii,
        ).model_dump()
        task_storage.save(task_id, task_data)

        logger.info("Task %s completed with %d PII detections", task_id, len(detected_pii))

        return {
            "status": "completed",
            "task_id": task_id,
            "pii_count": len(detected_pii),
        }

    except Exception as e:
        logger.error("Task %s failed: %s", task_id, str(e), exc_info=True)
        task_data = task_storage.get(task_id) or {}
        task_data["status"] = TaskStatus.FAILED
        task_data["message"] = f"마스킹 작업 실패: {str(e)}"
        task_data["error"] = {"code": ErrorCode.PROCESSING_ERROR, "message": str(e)}
        task_storage.save(task_id, task_data)
        raise
