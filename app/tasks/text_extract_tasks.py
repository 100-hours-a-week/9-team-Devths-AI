"""
Text Extract Tasks — ADR-102.

Celery 태스크: 이력서/채용공고 텍스트 추출 + 임베딩 저장.
asyncio.create_task() 대신 Celery Worker에서 실행하여 504 타임아웃 해결.
"""

import asyncio
import logging
import uuid
from datetime import datetime

from app.tasks.celery_app import celery_app
from app.utils.log_sanitizer import sanitize_log_input
from app.utils.url_validator import validate_url

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, name="app.tasks.text_extract_tasks.process_text_extract_task")
def process_text_extract_task(
    self,
    task_id: str,
    user_id: str,
    resume_data: dict,
    job_posting_data: dict,
    model: str = "gemini",
):
    """텍스트 추출 + 임베딩 저장 Celery 태스크 (ADR-102).

    Args:
        task_id: 태스크 ID
        user_id: 사용자 ID
        resume_data: 이력서 입력 데이터 (s3_key, file_type, text, file_id 등)
        job_posting_data: 채용공고 입력 데이터
        model: OCR 모델 선택 (gemini, auto 등)
    """
    safe_task_id = sanitize_log_input(task_id)
    logger.info("[TextExtractTask] 태스크 시작: %s", safe_task_id)

    try:
        result = asyncio.run(
            _process_text_extract_async(
                task_id=task_id,
                user_id=user_id,
                resume_data=resume_data,
                job_posting_data=job_posting_data,
                model=model,
            )
        )
        logger.info("[TextExtractTask] 태스크 완료: %s", safe_task_id)
        return result

    except Exception as e:
        logger.error("[TextExtractTask] 태스크 실패: %s — %s", safe_task_id, type(e).__name__)
        raise self.retry(exc=e, countdown=60, max_retries=2) from e


async def _process_text_extract_async(
    task_id: str,
    user_id: str,
    resume_data: dict,
    job_posting_data: dict,
    model: str,
) -> dict:
    """비동기 텍스트 추출 로직 (기존 process_text_extract 로직 이관)."""
    from app.api.routes.v2._helpers import format_analysis_text, get_services
    from app.config.dependencies import get_legacy_task_storage
    from app.prompts import get_extract_title_prompt, get_opening_prompt
    from app.schemas.common import ErrorCode, TaskStatus
    from app.schemas.text_extract import (
        DocumentExtractResult,
        DocumentInput,
        PageText,
        TextExtractResult,
    )
    from app.services.text_splitter_service import TextSplitterService
    from app.services.web_loader_service import WebLoaderService

    task_storage = get_legacy_task_storage()
    task_key = task_id

    try:
        rag = get_services()

        logger.info("")
        logger.info("=" * 80)
        logger.info("=== 📄 텍스트 추출 시작 (Celery Worker) ===")
        logger.info("=" * 80)
        safe_user_id = sanitize_log_input(user_id)
        logger.info("📌 OCR 전략: %s", model.upper())
        logger.info("📌 사용자 ID: %s", safe_user_id)
        logger.info("")

        async def extract_document(doc_data: dict, doc_type: str) -> DocumentExtractResult:
            """문서 추출 헬퍼 함수"""
            logger.info("📄 [%s] 처리 시작", doc_type.upper())

            doc_input = DocumentInput(**doc_data)
            extracted_text = ""
            pages = None

            if doc_input.s3_key:
                file_type = doc_input.get_file_type_simple() or "pdf"
                logger.info("   → 파일 타입: %s", file_type)
                logger.info("   → S3 키: %s", doc_input.s3_key[:50] if doc_input.s3_key else "N/A")

                logger.info("   🔍 [OCRService] CLOVA OCR 우선 → Gemini Fallback 시작")
                ocr_result = await rag.ocr.extract_text(
                    file_url=str(doc_input.s3_key),
                    file_type=file_type,
                    user_id=str(user_id),
                    fallback_enabled=True,
                )
                ocr_engine = ocr_result.get("ocr_engine") or "gemini"
                extracted_text = ocr_result.get("extracted_text", "")
                pages = [PageText(**page) for page in ocr_result.get("pages", [])]

                logger.info(
                    "   ✅ [%s OCR] 추출 완료: %d자 (페이지: %d)",
                    ocr_engine.upper(),
                    len(extracted_text),
                    len(pages),
                )
            elif doc_input.url:
                safe_url_log = sanitize_log_input(doc_input.url[:80] if doc_input.url else "N/A")
                logger.info("   → URL 입력: %s", safe_url_log)

                # SSRF 방어: URL 검증
                is_valid, error_msg = validate_url(doc_input.url)
                if not is_valid:
                    logger.warning("   ⚠️ 허용되지 않은 URL 차단: %s", safe_url_log)
                    extracted_text = ""
                else:
                    extracted_text = await WebLoaderService.extract_text_from_url(doc_input.url)
                    if extracted_text:
                        logger.info(
                            "   ✅ [WebLoader] URL 텍스트 추출 완료: %d자", len(extracted_text)
                        )
                    else:
                        logger.warning("   ⚠️ URL에서 텍스트를 추출할 수 없습니다")
            else:
                extracted_text = doc_input.text or ""
                logger.info("   → 텍스트 직접 입력: %d characters", len(extracted_text))

            if extracted_text:
                document_id = f"{doc_type}_{uuid.uuid4().hex[:12]}"
                doc_metadata = {
                    "user_id": user_id,
                    "file_id": doc_input.file_id,
                    "created_at": datetime.now().isoformat(),
                }

                splitter = TextSplitterService()
                batch_docs = splitter.split_to_batch_docs(
                    text=extracted_text,
                    document_id=document_id,
                    metadata=doc_metadata,
                )

                if len(batch_docs) == 1:
                    await rag.vectordb.add_document(
                        document_id=batch_docs[0]["id"],
                        text=batch_docs[0]["text"],
                        collection_type=doc_type,
                        metadata=batch_docs[0]["metadata"],
                    )
                else:
                    await rag.vectordb.add_documents_batch(
                        documents=batch_docs,
                        collection_type=doc_type,
                    )

                logger.info("   ✅ VectorDB 저장 완료: %s (%d 청크)", document_id, len(batch_docs))

            return DocumentExtractResult(
                file_id=doc_input.file_id, extracted_text=extracted_text, pages=pages
            )

        resume_result, job_posting_result = await asyncio.gather(
            extract_document(resume_data, "resume"),
            extract_document(job_posting_data, "job_posting"),
        )

        # 분석 리포트 생성
        logger.info("")
        logger.info("📊 분석 리포트 생성 시작...")
        analysis_failed = False
        try:
            analysis_result = await rag.llm.generate_analysis(
                resume_text=resume_result.extracted_text,
                posting_text=job_posting_result.extracted_text,
                user_id=str(user_id),
            )
            logger.info("✅ 분석 리포트 생성 완료")

            try:
                matching = analysis_result.get("matching", {})
                analysis_doc_id = f"analysis_{user_id}_{uuid.uuid4().hex[:8]}"
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
                        "user_id": str(user_id),
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
            logger.warning("⚠️ 분석 리포트 생성 실패: %s", e, exc_info=True)
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
            logger.info("✅ 채팅방 제목: %s", chat_title)
        except Exception as e:
            logger.error("❌ 채팅방 제목 추출 실패: %s", e)

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
            logger.error("❌ 오프닝 메시지 생성 실패: %s", e)
            ai_message = f"안녕하세요! 지원하신 {chat_title or '직무'}에 대한 분석이 완료되었습니다. 결과를 확인하시고 궁금한 점이 있다면 언제든 물어봐주세요!"

        # 결과 저장
        task_data = task_storage.get(task_key) or {}
        task_data["status"] = TaskStatus.COMPLETED
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
        task_storage.save(task_key, task_data)

        logger.info("")
        logger.info("✅ 텍스트 추출 + 분석 완료!")
        logger.info("   → 이력서 OCR: %d자", len(resume_result.extracted_text))
        logger.info("   → 채용공고 OCR: %d자", len(job_posting_result.extracted_text))

        return {
            "status": "completed",
            "task_id": task_id,
            "resume_length": len(resume_result.extracted_text),
            "job_posting_length": len(job_posting_result.extracted_text),
        }

    except Exception as e:
        logger.error("텍스트 추출 오류: %s", e, exc_info=True)
        task_data = task_storage.get(task_key) or {}
        task_data["status"] = TaskStatus.FAILED
        task_data["error"] = {"code": ErrorCode.PROCESSING_ERROR, "message": str(e)}
        task_storage.save(task_key, task_data)
        raise
