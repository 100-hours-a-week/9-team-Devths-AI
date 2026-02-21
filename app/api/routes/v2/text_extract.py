"""
v2 텍스트 추출 + 임베딩 API

POST /ai/text/extract - 이력서 + 채용공고 텍스트 추출 및 분석
"""

import asyncio
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, status

from app.api.routes.v2._helpers import format_analysis_text, get_services
from app.config.dependencies import get_legacy_task_storage
from app.prompts import get_extract_title_prompt, get_opening_prompt
from app.schemas.common import AsyncTaskResponse, ErrorCode, TaskStatus
from app.schemas.text_extract import (
    DocumentExtractResult,
    DocumentInput,
    PageText,
    TextExtractRequest,
    TextExtractResult,
)
from app.services.text_splitter_service import TextSplitterService
from app.services.web_loader_service import WebLoaderService
from app.utils.log_sanitizer import safe_info, sanitize_log_input

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
    """텍스트 추출 + 임베딩 저장 (통합) - 이력서 + 채용공고"""
    task_id = request.task_id

    # 모니터링 메트릭 전송 (PLG 적용 과정에서 CloudWatch 제거됨)

    # 비동기 작업 시작
    task_key = str(task_id)
    task_storage.save(
        task_key,
        {
            "type": "text_extract",
            "status": TaskStatus.PROCESSING,
            "created_at": datetime.now(),
            "room_id": request.room_id,
            "request": request.model_dump(),
        },
    )

    import time
    enqueued_time = time.time()

    async def process_text_extract(store):
        try:
            from app.core.monitoring import CELERY_TASK_WAIT_TIME, CELERY_TASKS_ACTIVE
            wait_time = time.time() - enqueued_time
            CELERY_TASK_WAIT_TIME.labels(task_name="text_extract").observe(wait_time)
            CELERY_TASKS_ACTIVE.labels(task_name="text_extract").inc()
        except Exception:
            pass

        try:
            rag = get_services()

            model = request.model if hasattr(request, "model") and request.model else "gemini"
            logger.info("")
            logger.info(f"{'='*80}")
            logger.info("=== 📄 텍스트 추출 시작 (이력서 + 채용공고) ===")
            logger.info(f"{'='*80}")
            logger.info(
                f"📌 OCR 전략: {'GEMINI (V1 Temporary)' if model == 'auto' else model.upper()}"
            )
            safe_info(logger, "📌 사용자 ID: %s", request.user_id)
            logger.info(f"📌 vLLM 서비스: {'✅ 사용 가능' if rag.vllm else '❌ 사용 불가'}")
            logger.info("")

            async def extract_document(
                doc_input: DocumentInput, doc_type: str
            ) -> DocumentExtractResult:
                """문서 추출 헬퍼 함수"""
                logger.info(f"📄 [{doc_type.upper()}] 처리 시작")

                if doc_input.s3_key:
                    file_type = doc_input.get_file_type_simple() or "pdf"
                    logger.info(f"   → 파일 타입 (MIME): {doc_input.file_type}")
                    logger.info(f"   → 파일 타입 (단순): {file_type}")
                    safe_s3_key = sanitize_log_input(doc_input.s3_key)
                    logger.info("   → S3 키: %s", safe_s3_key)

                    logger.info("   🔍 [OCRService] CLOVA OCR 우선 → Gemini Fallback 시작")
                    ocr_result = await rag.ocr.extract_text(
                        file_url=str(doc_input.s3_key),
                        file_type=file_type,
                        user_id=str(request.user_id),
                        fallback_enabled=True,
                    )
                    ocr_engine = ocr_result.get("ocr_engine") or "gemini"
                    fallback_reason = ocr_result.get("fallback_reason")
                    extracted_text = ocr_result.get("extracted_text", "")
                    pages = [PageText(**page) for page in ocr_result.get("pages", [])]

                    if fallback_reason:
                        logger.info(
                            f"   ✅ [{ocr_engine.upper()} OCR] 추출 완료 (폴백 사유: {fallback_reason}): "
                            f"{len(extracted_text)}자 (페이지: {len(pages)})"
                        )
                    else:
                        logger.info(
                            f"   ✅ [{ocr_engine.upper()} OCR] 추출 완료: "
                            f"{len(extracted_text)}자 (페이지: {len(pages)})"
                        )
                elif doc_input.url:
                    logger.info("   → URL 입력: %s", doc_input.url[:80])
                    extracted_text = await WebLoaderService.extract_text_from_url(doc_input.url)
                    pages = None
                    if extracted_text:
                        logger.info(
                            "   ✅ [WebLoader] URL 텍스트 추출 완료: %d자",
                            len(extracted_text),
                        )
                    else:
                        logger.warning("   ⚠️ URL에서 텍스트를 추출할 수 없습니다")
                else:
                    extracted_text = doc_input.text or ""
                    pages = None
                    logger.info(f"   → 텍스트 직접 입력: {len(extracted_text)} characters")

                if extracted_text:
                    document_id = f"{doc_type}_{uuid.uuid4().hex[:12]}"
                    doc_metadata = {
                        "user_id": request.user_id,
                        "file_id": doc_input.file_id,
                        "created_at": datetime.now().isoformat(),
                    }

                    # 텍스트 분할 (ADR-060 Phase 1)
                    splitter = TextSplitterService()
                    batch_docs = splitter.split_to_batch_docs(
                        text=extracted_text,
                        document_id=document_id,
                        metadata=doc_metadata,
                    )

                    if len(batch_docs) == 1:
                        # 단일 청크: 기존 add_document() 사용
                        await rag.vectordb.add_document(
                            document_id=batch_docs[0]["id"],
                            text=batch_docs[0]["text"],
                            collection_type=doc_type,
                            metadata=batch_docs[0]["metadata"],
                        )
                    else:
                        # 다중 청크: add_documents_batch() 사용
                        await rag.vectordb.add_documents_batch(
                            documents=batch_docs,
                            collection_type=doc_type,
                        )

                    safe_document_id = sanitize_log_input(document_id)
                    logger.info(
                        "   ✅ VectorDB 저장 완료: %s (%d 청크)",
                        safe_document_id,
                        len(batch_docs),
                    )

                return DocumentExtractResult(
                    file_id=doc_input.file_id, extracted_text=extracted_text, pages=pages
                )

            resume_result, job_posting_result = await asyncio.gather(
                extract_document(request.resume, "resume"),
                extract_document(request.job_posting, "job_posting"),
            )

            # 분석 리포트 생성
            logger.info("")
            logger.info("📊 분석 리포트 생성 시작...")
            analysis_failed = False
            try:
                analysis_result = await rag.llm.generate_analysis(
                    resume_text=resume_result.extracted_text,
                    posting_text=job_posting_result.extracted_text,
                    user_id=str(request.user_id),
                )
                logger.info("✅ 분석 리포트 생성 완료")
                # 문서 설계: 분석 완료 시 analysis_results 컬렉션에 저장
                try:
                    matching = analysis_result.get("matching", {})
                    analysis_doc_id = f"analysis_{request.user_id}_{uuid.uuid4().hex[:8]}"
                    await rag.vectordb.add_document(
                        document_id=analysis_doc_id,
                        text=format_analysis_text(
                            resume_analysis=analysis_result.get("resume_analysis"),
                            posting_analysis=analysis_result.get("posting_analysis"),
                            summary="",
                        )
                        or "분석 결과 없음",
                        collection_type="analysis_results",
                        metadata={
                            "user_id": str(request.user_id),
                            "analysis_type": "full",
                            "score": matching.get("score"),
                            "grade": matching.get("grade", ""),
                            "created_at": datetime.now().isoformat(),
                        },
                    )
                    logger.info("   ✅ VectorDB analysis_results 저장 완료")
                except Exception as ex:
                    logger.warning("   ⚠️ analysis_results 저장 건너뜀: %s", ex)
            except Exception as e:
                analysis_failed = True
                logger.warning(
                    "⚠️ 분석 리포트 생성 실패 (오프닝 메시지에 분석 내용이 비어 보일 수 있음): %s",
                    e,
                    exc_info=True,
                )
                analysis_result = {
                    "resume_analysis": {"strengths": [], "weaknesses": [], "suggestions": []},
                    "posting_analysis": {
                        "company": "알 수 없음",
                        "position": "알 수 없음",
                        "required_skills": [],
                        "preferred_skills": [],
                    },
                    "matching": {
                        "score": 0,
                        "grade": "F",
                        "matched_skills": [],
                        "missing_skills": [],
                    },
                }

            # 채팅방 제목 추출
            chat_title = ""
            try:
                logger.info("📝 채팅방 제목 추출 중...")
                posting_text = job_posting_result.extracted_text[:1000]
                title_prompt = f"""{get_extract_title_prompt()}

## 채용공고 텍스트
{posting_text}
"""
                title_response = ""
                async for chunk in rag.llm.generate_response(
                    user_message=title_prompt,
                    context=None,
                    history=[],
                    system_prompt="당신은 채용공고에서 회사명과 직무를 정확히 추출하는 AI입니다.",
                ):
                    title_response += chunk

                chat_title = title_response.strip()
                logger.info(f"✅ 채팅방 제목: {chat_title}")
            except Exception as e:
                logger.error(f"❌ 채팅방 제목 추출 실패: {e}")
            logger.info("")

            task_data = store.get(task_key) or {}
            task_data["status"] = TaskStatus.COMPLETED

            formatted_text = format_analysis_text(
                resume_analysis=analysis_result.get("resume_analysis"),
                posting_analysis=analysis_result.get("posting_analysis"),
                summary=chat_title,
            )
            formatted_text = formatted_text or "분석 결과가 없습니다."
            if analysis_failed:
                formatted_text += (
                    "\n\n(상세 분석이 일시적으로 반영되지 않았습니다. "
                    "이력서·채용공고 텍스트는 저장되었으니, 궁금한 점을 질문해 주세요.)"
                )

            # 오프닝 메시지 생성
            logger.info("🤖 오프닝 메시지 생성 시작...")
            ai_message = ""
            try:
                opening_prompt = get_opening_prompt(formatted_text)
                async for chunk in rag.llm.generate_response(
                    user_message=opening_prompt,
                    context=None,
                    history=[],
                    system_prompt="당신은 도움을 주는 친절한 취업 어시스턴트입니다.",
                ):
                    ai_message += chunk
                logger.info("✅ 오프닝 메시지 생성 완료")
            except Exception as e:
                logger.error(f"❌ 오프닝 메시지 생성 실패: {e}")
                ai_message = f"안녕하세요! 지원하신 {chat_title or '직무'}에 대한 분석이 완료되었습니다. 결과를 확인하시고 궁금한 점이 있다면 언제든 물어봐주세요!"

            task_data["result"] = TextExtractResult(
                success=True,
                summary=chat_title or None,
                resume_ocr=resume_result.extracted_text,
                job_posting_ocr=job_posting_result.extracted_text,
                resume_analysis=analysis_result.get("resume_analysis"),
                posting_analysis=analysis_result.get("posting_analysis"),
                formatted_text=formatted_text,
                ai_message=ai_message,
            ).model_dump()
            store.save(task_key, task_data)

            logger.info("")
            logger.info("✅ 텍스트 추출 + 분석 완료!")
            logger.info(f"   → 이력서 OCR: {len(resume_result.extracted_text)}자")
            logger.info(f"   → 채용공고 OCR: {len(job_posting_result.extracted_text)}자")

        except Exception as e:
            logger.error(f"텍스트 추출 오류: {e}", exc_info=True)
            task_data = store.get(task_key) or {}
            task_data["status"] = TaskStatus.FAILED
            task_data["error"] = {"code": ErrorCode.PROCESSING_ERROR, "message": str(e)}
            store.save(task_key, task_data)
        finally:
            try:
                from app.core.monitoring import CELERY_TASKS_ACTIVE
                CELERY_TASKS_ACTIVE.labels(task_name="text_extract").dec()
            except Exception:
                pass

    asyncio.create_task(process_text_extract(task_storage))

    return AsyncTaskResponse(task_id=task_id, status=TaskStatus.PROCESSING)
