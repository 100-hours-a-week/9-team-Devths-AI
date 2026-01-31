"""
PII Masking API Routes

게시판 첨부파일에서 개인정보를 자동으로 감지하고 마스킹 처리하는 API
"""

import asyncio
import base64
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, status

from app.schemas.common import AsyncTaskResponse, ErrorCode, TaskStatus, TaskStatusResponse
from app.schemas.masking import (
    DetectedPII,
    MaskingDraftRequest,
    MaskingDraftResult,
    MaskingModelType,
)
from app.services.chandra_masking import get_chandra_masking_service
from app.services.gemini_masking import get_gemini_masking_service
from app.utils.task_store import get_task_store

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai/masking",
    tags=["PII Masking"],
    responses={404: {"description": "Not found"}},
)

# 파일 기반 작업 저장소 (uvicorn --reload에서도 유지됨)
task_store = get_task_store()

# 백그라운드 작업 추적 (가비지 컬렉션 방지)
background_tasks_set = set()


async def verify_api_key(x_api_key: str | None = Header(None)):
    """API 키 검증"""
    if x_api_key != "your-api-key-here":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return x_api_key


@router.post(
    "/draft",
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

    **처리 플로우:**
    1. PDF/이미지 다운로드
    2. AI 모델로 PII 감지
       - Gemini: 얼굴 감지 (Vision API)
       - Chandra: 텍스트 PII 감지 (OCR + Pattern Matching)
    3. PII 마스킹
    4. 마스킹된 파일 + 썸네일 생성
    5. S3 업로드 (실제로는 별도 처리)
    6. 결과 반환

    **사용 가능 모델:**
    - gemini: Google Gemini 3 Flash Preview (얼굴 감지 전용)
    - chandra: datalab-to/chandra (텍스트 PII 감지 전용)

    **주의:**
    - 사용자 수정은 프론트엔드에서 처리 (AI 불필요)
    """,
)
async def masking_draft(request: MaskingDraftRequest):
    """
    게시판 첨부파일 마스킹

    Args:
        request: 마스킹 요청 (s3_key, file_type, model)

    Returns:
        AsyncTaskResponse: task_id와 처리 상태
    """
    task_id = f"task_masking_{uuid.uuid4().hex[:12]}"

    logger.info("=" * 80)
    logger.info(f"[MASKING_DRAFT] Creating new task: {task_id}")
    logger.info(f"[MASKING_DRAFT] Current tasks in store: {task_store.list_all()}")

    # 초기 상태를 즉시 저장 (통합 저장소 사용)
    task_data = {
        "type": "masking",  # 작업 타입 구분
        "status": TaskStatus.PROCESSING,
        "created_at": datetime.now(),
        "progress": 0,
        "message": "마스킹 작업을 시작합니다...",
        "request": request.model_dump(),
    }
    task_store.save(task_id, task_data)

    logger.info(f"[MASKING_DRAFT] Task {task_id} saved to file store")
    logger.info(f"[MASKING_DRAFT] Task exists: {task_store.exists(task_id)}")
    logger.info(f"[MASKING_DRAFT] Task data: {task_store.get(task_id)}")
    logger.info("=" * 80)

    # 백그라운드에서 처리
    async def process_masking():
        logger.info(f"[PROCESS_MASKING] Starting masking task {task_id}")
        logger.info(f"[PROCESS_MASKING] Task exists in store: {task_store.exists(task_id)}")
        logger.info(f"[PROCESS_MASKING] Using model: {request.model}")
        try:
            # 모델 선택
            if request.model == MaskingModelType.CHANDRA:
                service = get_chandra_masking_service()
                model_name = "Chandra"
            else:  # GEMINI
                service = get_gemini_masking_service()
                model_name = "Gemini"

            # 진행 상태 업데이트
            task_data = task_store.get(task_id)
            task_data["progress"] = 10
            task_data["message"] = "파일을 다운로드 중입니다..."
            task_store.save(task_id, task_data)
            logger.info(f"Task {task_id}: Downloading file")

            # 파일 타입에 따라 처리
            if request.file_type == "pdf":
                task_data = task_store.get(task_id)
                task_data["progress"] = 30
                task_data["message"] = "PDF를 이미지로 변환 중입니다..."
                task_store.save(task_id, task_data)

                task_data = task_store.get(task_id)
                task_data["progress"] = 50
                task_data["message"] = f"{model_name} API로 PII를 감지 중입니다..."
                task_store.save(task_id, task_data)

                masked_bytes, thumbnail_bytes, detections = await service.mask_pdf(
                    file_url=str(request.file_url)
                )

            else:  # image
                task_data = task_store.get(task_id)
                task_data["progress"] = 30
                task_data["message"] = "이미지에서 PII를 감지 중입니다..."
                task_store.save(task_id, task_data)

                task_data = task_store.get(task_id)
                task_data["progress"] = 50
                task_data["message"] = f"{model_name} API로 PII를 감지 중입니다..."
                task_store.save(task_id, task_data)

                masked_bytes, thumbnail_bytes, detections = await service.mask_image_file(
                    file_url=str(request.s3_key)
                )

            task_data = task_store.get(task_id)
            task_data["progress"] = 80
            task_data["message"] = "마스킹된 파일을 저장 중입니다..."
            task_store.save(task_id, task_data)

            # 실제로는 S3에 업로드하고 URL 반환
            # 여기서는 base64 data URL로 반환 (데모용)
            masked_base64 = base64.b64encode(masked_bytes).decode("utf-8")
            thumbnail_base64 = base64.b64encode(thumbnail_bytes).decode("utf-8")

            # data URL 형식 (전체 base64 포함)
            file_ext = "pdf" if request.file_type == "pdf" else "png"
            mime_type = f"application/{file_ext}" if request.file_type == "pdf" else "image/png"

            masked_url = f"data:{mime_type};base64,{masked_base64}"
            thumbnail_url = f"data:image/png;base64,{thumbnail_base64}"

            # 결과 포맷팅
            detected_pii = []
            for det in detections:
                # PIIType enum에 없는 타입은 처리하지 않거나 unknown으로 설정
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
                    # enum에 없는 타입은 건너뛰기
                    logger.warning(f"Unknown PII type: {pii_type}")

            task_data = task_store.get(task_id)
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
            task_store.save(task_id, task_data)

            logger.info(f"Task {task_id} completed with {len(detected_pii)} PII detections")

        except Exception as e:
            logger.error(f"Task {task_id} failed: {str(e)}", exc_info=True)
            task_data = task_store.get(task_id) or {}
            task_data["status"] = TaskStatus.FAILED
            task_data["message"] = f"마스킹 작업 실패: {str(e)}"
            task_data["error"] = {"code": ErrorCode.PROCESSING_ERROR, "message": str(e)}
            task_store.save(task_id, task_data)

    # asyncio.create_task로 작업 생성 및 추적
    task = asyncio.create_task(process_masking())
    background_tasks_set.add(task)

    # 작업 완료 시 자동으로 set에서 제거
    task.add_done_callback(lambda t: background_tasks_set.discard(t))

    logger.info(f"Created background task {task_id}, current tasks: {len(background_tasks_set)}")

    return AsyncTaskResponse(
        task_id=task_id,
        status=TaskStatus.PROCESSING,
        message="마스킹 작업을 시작했습니다. /ai/task/{task_id}로 진행 상태를 확인하세요.",
    )


@router.get(
    "/task/{task_id}",
    response_model=TaskStatusResponse,
    summary="마스킹 작업 상태 조회",
    description="""
    비동기 마스킹 작업의 상태를 조회합니다.

    **통합 엔드포인트:** `/ai/task/{task_id}`로도 조회 가능합니다.

    **상태 종류:**
    - processing: 처리 중
    - completed: 완료
    - failed: 실패

    **progress:** 0-100 (진행률)
    """,
)
async def get_masking_task_status(task_id: str):
    """
    마스킹 작업 상태 조회

    Args:
        task_id: 작업 ID

    Returns:
        TaskStatusResponse: 작업 상태 및 결과
    """
    logger.info("=" * 80)
    logger.info(f"[GET_STATUS] Looking for task: {task_id}")
    logger.info(f"[GET_STATUS] All tasks in store: {task_store.list_all()}")
    logger.info(f"[GET_STATUS] Task exists: {task_store.exists(task_id)}")
    logger.info("=" * 80)

    task = task_store.get(task_id)
    if task is None:
        logger.error(f"[GET_STATUS] Task {task_id} NOT FOUND!")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.TASK_NOT_FOUND, "message": "작업을 찾을 수 없습니다."},
        )

    logger.info(f"[GET_STATUS] Task {task_id} found! Status: {task['status']}")

    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        message=task.get("message"),
        result=task.get("result"),
        error=task.get("error"),
    )


@router.get(
    "/health", summary="PII 마스킹 서비스 헬스 체크", description="PII 마스킹 서비스 상태 확인"
)
async def masking_health_check():
    """PII 마스킹 서비스 헬스 체크"""
    health_status = {"status": "healthy", "service": "pii-masking", "models": {}}

    # Gemini 체크
    try:
        get_gemini_masking_service()
        health_status["models"]["gemini"] = {
            "status": "available",
            "provider": "Google Gemini 3 Flash Preview",
        }
    except Exception as e:
        logger.error(f"Gemini health check failed: {e}")
        health_status["models"]["gemini"] = {"status": "error", "error": str(e)}

    # Chandra 체크
    try:
        get_chandra_masking_service()
        health_status["models"]["chandra"] = {
            "status": "available",
            "provider": "datalab-to/chandra",
        }
    except Exception as e:
        logger.error(f"Chandra health check failed: {e}")
        health_status["models"]["chandra"] = {"status": "error", "error": str(e)}

    # 모든 모델이 실패하면 전체 상태 error
    if all(m.get("status") == "error" for m in health_status["models"].values()):
        health_status["status"] = "error"
        health_status["message"] = "모든 마스킹 모델이 사용 불가능합니다."
    else:
        health_status["message"] = "PII 마스킹 서비스가 정상 작동 중입니다."

    return health_status
