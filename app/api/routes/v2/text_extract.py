"""
v2 텍스트 추출 + 임베딩 API

POST /ai/text/extract - 이력서 + 채용공고 텍스트 추출 및 분석

ADR-102: asyncio.create_task() → Celery 태스크로 이관하여 504 타임아웃 해결.
ADR-102 Fallback: Celery 브로커 연결 불가 시 asyncio.create_task()로 fallback.
"""

import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, status

from app.config.dependencies import get_legacy_task_storage
from app.schemas.common import AsyncTaskResponse, PollingHint, TaskStatus
from app.schemas.text_extract import TextExtractRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/text/extract",
    response_model=AsyncTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="텍스트 추출 + 임베딩 저장 (이력서 + 채용공고)",
    description="""
    이력서와 채용공고에서 텍스트를 추출하고 내부에서 임베딩까지 처리합니다.

    **요청 구조:**
    - `resume`: 이력서/포트폴리오 입력 (필수)
    - `job_posting`: 채용공고 입력 (필수)
    - 각 문서는 파일 업로드(`s3_key` + `file_type`) 또는 텍스트 입력(`text`) 중 하나 선택

    **처리 방식:** 비동기 (task_id 반환 → 폴링 필요)

    **내부 처리 흐름:**
    1. 이력서 처리: 파일이면 OCR/VLM으로 텍스트 추출, 텍스트면 그대로 사용
    2. 채용공고 처리: 파일이면 OCR/VLM으로 텍스트 추출, 텍스트면 그대로 사용
    3. 각 텍스트 청킹 (500 tokens, 50 overlap)
    4. Gemini Embedding 생성
    5. VectorDB에 저장 (resume, job_posting 컬렉션)
    6. 분석 리포트 생성 (이력서/채용공고 분석)
    7. 추출된 텍스트 + 분석 결과 반환
    """,
    responses={
        400: {
            "description": "Bad Request",
            "content": {
                "application/json": {
                    "examples": {
                        "invalid_request": {
                            "value": {
                                "detail": {
                                    "code": "INVALID_REQUEST",
                                    "message": "resume과 job_posting 는 필수 입력해야합니다",
                                }
                            }
                        },
                        "invalid_file_type": {
                            "value": {
                                "detail": {
                                    "code": "INVALID_FILE_TYPE",
                                    "message": "file_type은 pdf 또는 image만 가능합니다",
                                    "field": "resume.file_type",
                                }
                            }
                        },
                        "invalid_document": {
                            "value": {
                                "detail": {
                                    "code": "INVALID_DOCUMENT",
                                    "message": "s3_key 또는 text 중 하나는 필수입니다",
                                    "field": "resume",
                                }
                            }
                        },
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {"code": "UNAUTHORIZED", "message": "유효하지 않은 API Key입니다"}
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
                            "code": "FILE_NOT_FOUND",
                            "message": "파일을 찾을 수 없습니다: users/12/resume/abc123.pdf",
                        }
                    }
                }
            },
        },
        422: {
            "description": "Unprocessable Entity",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "code": "OCR_FAILED",
                            "message": "이미지에서 텍스트를 추출할 수 없습니다",
                        }
                    }
                }
            },
        },
        429: {
            "description": "Too Many Requests",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "code": "RATE_LIMIT_EXCEEDED",
                            "message": "요청 한도 초과. 1분 후 재시도하세요",
                        }
                    }
                }
            },
        },
        500: {
            "description": "Internal Server Error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "code": "INTERNAL_ERROR",
                            "message": "내부 서버 오류가 발생했습니다",
                        }
                    }
                }
            },
        },
        503: {
            "description": "Service Unavailable",
            "content": {
                "application/json": {
                    "examples": {
                        "llm_unavailable": {
                            "value": {
                                "detail": {
                                    "code": "LLM_UNAVAILABLE",
                                    "message": "AI 서비스에 연결할 수 없습니다",
                                }
                            }
                        },
                        "s3_unavailable": {
                            "value": {
                                "detail": {
                                    "code": "S3_UNAVAILABLE",
                                    "message": "파일 스토리지에 연결할 수 없습니다",
                                }
                            }
                        },
                    }
                }
            },
        },
    },
)
async def text_extract(
    request: TextExtractRequest,
    task_storage=Depends(get_legacy_task_storage),
):
    """텍스트 추출 + 임베딩 저장 (통합) - 이력서 + 채용공고

    ADR-102: Celery 태스크로 이관하여 504 타임아웃 해결.
    """
    task_id = request.task_id

    # 비동기 작업 시작 — Redis에 초기 상태 저장
    task_key = str(task_id)
    task_storage.save(
        task_key,
        {
            "type": "text_extract",
            "status": TaskStatus.PROCESSING,
            "created_at": datetime.now().isoformat(),
            "room_id": request.room_id,
            "request": request.model_dump(),
        },
    )

    # ADR-102: Celery 태스크로 이관 (asyncio.create_task 대신)
    # ADR-102 Fallback: Celery 브로커 연결 불가 시 asyncio.create_task()로 fallback
    from app.tasks.celery_utils import is_celery_available
    from app.utils.log_sanitizer import sanitize_log_input

    model = request.model if hasattr(request, "model") and request.model else "gemini"
    safe_task_key = sanitize_log_input(task_key)

    if is_celery_available():
        # Celery 태스크 비동기 실행 (즉시 반환, 블로킹 없음)
        from app.tasks.text_extract_tasks import process_text_extract_task

        process_text_extract_task.delay(
            task_id=task_key,
            user_id=request.user_id,
            resume_data=request.resume.model_dump(),
            job_posting_data=request.job_posting.model_dump(),
            model=model,
        )
        logger.info("[TextExtract] Celery 태스크 등록 완료: %s", safe_task_key)
    else:
        # Fallback: asyncio.create_task로 실행 (Celery Worker 미실행 시)
        from app.tasks.text_extract_tasks import _process_text_extract_async

        asyncio.create_task(
            _process_text_extract_async(
                task_id=task_key,
                user_id=request.user_id,
                resume_data=request.resume.model_dump(),
                job_posting_data=request.job_posting.model_dump(),
                model=model,
            )
        )
        logger.warning(
            "[TextExtract] Celery 불가 → asyncio fallback: %s (Celery Worker 실행 권장)",
            safe_task_key,
        )

    return AsyncTaskResponse(
        task_id=task_id,
        status=TaskStatus.PROCESSING,
        polling=PollingHint(interval_ms=2000, max_attempts=150),  # 최대 5분(150 × 2초)
    )
