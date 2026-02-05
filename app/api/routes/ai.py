import asyncio
import json
import logging
import os
import time
import uuid
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from app.prompts import (
    create_tech_followup_prompt,
    create_tech_interview_init_prompt,
    format_conversation_history,
    get_extract_title_prompt,
    get_opening_prompt,
    # 기술 면접 5단계 프롬프트
    get_system_tech_interview,
)
from app.schemas.calendar import CalendarParseRequest, CalendarParseResponse
from app.schemas.chat import (
    ChatMode,
    ChatRequest,
    InterviewQuestionState,
    InterviewSession,
)
from app.schemas.common import AsyncTaskResponse, ErrorCode, TaskStatus, TaskStatusResponse
from app.schemas.text_extract import (
    DocumentExtractResult,
    DocumentInput,
    PageText,
    TextExtractRequest,
    TextExtractResult,
)
from app.services.cloudwatch_service import CloudWatchService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.vectordb_service import VectorDBService
from app.services.vllm_service import VLLMService
from app.utils.log_sanitizer import safe_info, safe_warning, sanitize_log_input
from app.utils.prompt_guard import RiskLevel, check_prompt_injection
from app.utils.task_store import get_task_store

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai",
    tags=["AI APIs (v3.0)"],
    responses={404: {"description": "Not found"}},
)

# 통합 작업 저장소 (파일 기반, 서버 재시작 시에도 유지)
# 백엔드에서 전달받은 task_id를 키로 사용
task_store = get_task_store()


# ============================================================================
# 면접 세션 캐시 (메모리 기반)
# - Spring 백엔드가 세션 상태를 전달하지 않는 문제의 임시 해결책
# - 운영 환경에서는 Redis 권장
# ============================================================================
interview_sessions: dict[str, InterviewSession] = {}


def get_session_key(user_id: int, interview_id: int | None) -> str:
    """면접 세션 캐시 키 생성"""
    return f"interview:{user_id}:{interview_id or 'default'}"


# Initialize services
llm_service = None
vllm_service = None
vectordb_service = None
rag_service = None


def get_services():
    """Get or initialize AI services"""
    global llm_service, vllm_service, vectordb_service, rag_service

    if llm_service is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        llm_service = LLMService(api_key=api_key)
        vectordb_service = VectorDBService(api_key=api_key)

        # Initialize vLLM service (GCP GPU server)
        gcp_vllm_url = os.getenv("GCP_VLLM_BASE_URL")

        try:
            if gcp_vllm_url:
                logger.info(f"🌐 GCP vLLM 서버 연결: {gcp_vllm_url}")
                vllm_service = VLLMService()
                logger.info("✅ vLLM service initialized (GCP GPU server)")
            else:
                # GCP URL 없으면 OCR 전용 모드
                logger.info("💰 GCP URL 없음 - OCR 전용 모드로 초기화")
                vllm_service = VLLMService(ocr_only=True)
                logger.info("✅ vLLM service initialized (OCR-only mode)")
        except Exception as e:
            logger.warning(f"vLLM service initialization failed: {e}")
            vllm_service = None

        rag_service = RAGService(llm_service, vectordb_service, vllm_service)

    return rag_service


def format_analysis_text(
    resume_analysis: dict | None,
    posting_analysis: dict | None,
    summary: str | None,
) -> str:
    """분석 결과를 plain text로 포맷팅 (마크다운 없이)

    백엔드에서 바로 화면에 표시할 수 있도록 포맷팅된 텍스트를 생성합니다.
    """
    lines = []

    # 회사/직무
    if summary:
        lines.append(f"지원 회사 및 직무 : {summary}")
        lines.append("")

    # 이력서 분석
    if resume_analysis:
        lines.append("이력서 분석")
        lines.append("")
        lines.append("장점")
        strengths = resume_analysis.get("strengths", [])
        for i, strength in enumerate(strengths[:5], 1):
            lines.append(f"{i}. {strength}")
        lines.append("")
        lines.append("단점")
        weaknesses = resume_analysis.get("weaknesses", [])
        for i, weakness in enumerate(weaknesses[:5], 1):
            lines.append(f"{i}. {weakness}")
        lines.append("")

    # 채용공고 분석
    if posting_analysis:
        lines.append("채용 공고 분석")
        lines.append("")
        company = posting_analysis.get("company", "")
        position = posting_analysis.get("position", "")
        lines.append("기업 / 포지션")
        lines.append(f"{company} / {position}")
        lines.append("")
        lines.append("필수 역량")
        for skill in posting_analysis.get("required_skills", [])[:5]:
            lines.append(f"- {skill}")
        lines.append("")
        lines.append("우대 사항")
        for skill in posting_analysis.get("preferred_skills", [])[:5]:
            lines.append(f"- {skill}")
        lines.append("")

    # 매칭도
    if resume_analysis and posting_analysis:
        lines.append("매칭도")
        lines.append("")
        lines.append("나와 지원 직무에 맞는 점")
        # matches 필드가 있으면 사용, 없으면 strengths에서 가져옴
        matches = resume_analysis.get("matches", resume_analysis.get("strengths", [])[:3])
        for match in matches[:3] if matches else []:
            lines.append(f"- {match}")
        lines.append("")
        lines.append("나와 지원 직무에 맞지 않는 점")
        # gaps 필드가 있으면 사용, 없으면 weaknesses에서 가져옴
        gaps = resume_analysis.get("gaps", resume_analysis.get("weaknesses", [])[:3])
        for gap in gaps[:3] if gaps else []:
            lines.append(f"- {gap}")

    return "\n".join(lines)


async def verify_api_key(x_api_key: str | None = Header(None)):
    """API 키 검증"""
    # 실제로는 환경변수나 DB에서 확인
    if x_api_key != "your-api-key-here":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key")
    return x_api_key


# ============================================================================
# API 1: 텍스트 추출 + 임베딩 (통합) (비동기)
# ============================================================================


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
async def text_extract(request: TextExtractRequest):
    """텍스트 추출 + 임베딩 저장 (통합) - 이력서 + 채용공고"""
    task_id = request.task_id  # 백엔드에서 전달받은 task_id 사용

    # 모니터링 메트릭 전송
    try:
        cw = CloudWatchService.get_instance()
        # fire-and-forget (await 안함, 배경 실행)
        asyncio.create_task(cw.put_metric("AI_Job_Count", 1, "Count", {"Type": "text_extract"}))
    except Exception:
        pass

    # 비동기 작업 시작 (통합 task_store 사용)
    task_key = str(task_id)  # 파일 기반 저장소는 문자열 키 사용
    task_store.save(
        task_key,
        {
            "type": "text_extract",
            "status": TaskStatus.PROCESSING,
            "created_at": datetime.now(),
            "room_id": request.room_id,
            "request": request.model_dump(),
        },
    )

    # 백그라운드에서 처리
    async def process_text_extract():
        try:
            rag = get_services()

            # 모델 선택 (기본값: gemini)
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

                # 파일 업로드 방식
                if doc_input.s3_key:
                    # MIME 타입을 단순 타입으로 변환 (pdf/image)
                    file_type = doc_input.get_file_type_simple() or "pdf"
                    logger.info(f"   → 파일 타입 (MIME): {doc_input.file_type}")
                    logger.info(f"   → 파일 타입 (단순): {file_type}")
                    safe_s3_key = sanitize_log_input(doc_input.s3_key)
                    logger.info("   → S3 키: %s", safe_s3_key)

                    # OCRService: CLOVA OCR 우선 → Gemini Fallback (EasyOCR는 GPU 인스턴스 없음으로 미사용)
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

                # 텍스트 직접 입력
                else:
                    extracted_text = doc_input.text or ""
                    pages = None
                    logger.info(f"   → 텍스트 직접 입력: {len(extracted_text)} characters")

                # VectorDB에 임베딩 저장
                if extracted_text:
                    document_id = f"{doc_type}_{uuid.uuid4().hex[:12]}"
                    await rag.vectordb.add_document(
                        document_id=document_id,
                        text=extracted_text,
                        collection_type=doc_type,
                        metadata={
                            "user_id": request.user_id,
                            "file_id": doc_input.file_id,
                            "created_at": datetime.now().isoformat(),
                        },
                    )
                    safe_document_id = sanitize_log_input(document_id)
                    logger.info("   ✅ VectorDB 저장 완료: %s", safe_document_id)

                return DocumentExtractResult(
                    file_id=doc_input.file_id, extracted_text=extracted_text, pages=pages
                )

            # 이력서와 채용공고 각각 처리
            resume_result = await extract_document(request.resume, "resume")
            job_posting_result = await extract_document(request.job_posting, "job_posting")

            # 분석 리포트 생성 (명세서 요구사항)
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

            # 채팅방 제목 추출 (회사명/채용직무)
            chat_title = ""
            try:
                logger.info("📝 채팅방 제목 추출 중...")
                # 채용공고 텍스트 (앞 1000자만)
                posting_text = job_posting_result.extracted_text[:1000]
                title_prompt = f"""{get_extract_title_prompt()}

## 채용공고 텍스트
{posting_text}
"""
                # Gemini로 제목 추출
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
                # 실패해도 계속 진행
            logger.info("")

            # 결과 저장 (명세서에 따른 응답 구조)
            task_data = task_store.get(task_key) or {}
            task_data["status"] = TaskStatus.COMPLETED

            # formatted_text 생성 (백엔드에서 바로 표시용)
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

            # 오프닝 메시지 생성 (Gemini) - 대화 시작용
            logger.info("🤖 오프닝 메시지 생성 시작...")
            ai_message = ""
            try:
                opening_prompt = get_opening_prompt(formatted_text)
                # Gemini로 오프닝 생성
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
                # 실패 시 기본 메시지
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
            task_store.save(task_key, task_data)

            logger.info("")
            logger.info("✅ 텍스트 추출 + 분석 완료!")
            logger.info(f"   → 이력서 OCR: {len(resume_result.extracted_text)}자")
            logger.info(f"   → 채용공고 OCR: {len(job_posting_result.extracted_text)}자")

        except Exception as e:
            logger.error(f"텍스트 추출 오류: {e}", exc_info=True)
            task_data = task_store.get(task_key) or {}
            task_data["status"] = TaskStatus.FAILED
            task_data["error"] = {"code": ErrorCode.PROCESSING_ERROR, "message": str(e)}
            task_store.save(task_key, task_data)

    asyncio.create_task(process_text_extract())

    return AsyncTaskResponse(task_id=task_id, status=TaskStatus.PROCESSING)


# ============================================================================
# 비동기 작업 상태 조회
# ============================================================================


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
                                    "formatted_text": "지원 회사 및 직무 : 카카오 | 백엔드 개발자\n\n이력서 분석\n\n장점\n1. Java/Spring 숙련도\n2. 프로젝트 경험\n\n단점\n1. 클라우드 경험 부족\n\n채용 공고 분석\n\n기업 / 포지션\n카카오 / 백엔드 개발자\n\n필수 역량\n- Java\n- Spring\n- MySQL\n\n우대 사항\n- Docker\n- Kubernetes",
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
                            "code": "TASK_NOT_FOUND",
                            "message": "작업을 찾을 수 없습니다: 12",
                        }
                    }
                }
            },
        },
    },
)
async def get_task_status(task_id: str):
    """통합 비동기 작업 상태 조회 (text_extract, masking 등)"""
    task_key = str(task_id)  # 문자열 키로 통일
    task = task_store.get(task_key)

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


# ============================================================================
# API 2: 채팅 (통합: 대화/분석/면접) (스트리밍)
# ============================================================================


# ============================================================================
# API 2: 채팅 (통합: 대화/분석/면접) (스트리밍)
# ============================================================================


async def generate_chat_stream(request: ChatRequest):
    """채팅 응답 스트리밍 생성"""

    # =========================================================================
    # 프롬프트 인젝션 검사 (API 호출 전 사전 필터링)
    # =========================================================================
    user_message_raw = request.message or ""
    guard_result = check_prompt_injection(user_message_raw)

    if guard_result.risk_level == RiskLevel.BLOCK:
        # 차단: 안전한 응답 반환 (LLM 호출 없이)
        safe_warning(
            logger,
            "🚨 프롬프트 인젝션 차단: user_id=%s, patterns=%s",
            request.user_id,
            str(guard_result.matched_patterns),
        )
        blocked_response = guard_result.message
        yield f"data: {json.dumps({'content': blocked_response}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
        return

    if guard_result.risk_level == RiskLevel.WARNING:
        # 경고: 로깅만 하고 계속 진행
        safe_warning(
            logger,
            "⚠️ 의심스러운 입력 감지: user_id=%s, patterns=%s",
            request.user_id,
            str(guard_result.matched_patterns),
        )

    # context에서 모드 결정 (normal 또는 interview)
    mode = request.context.mode if request.context else ChatMode.NORMAL

    rag = get_services()
    newline = "\n"
    sse_end = "\n\n"

    # 모니터링 시작
    start_time = time.time()


    # 모델 선택 (gemini 또는 vllm)
    model = request.model.value if hasattr(request.model, "value") else str(request.model)

    # 메트릭 차원 정의
    dims = {"Model": model, "Mode": str(mode)}

    logger.info("")
    logger.info(f"{'='*80}")
    logger.info("=== 💬 채팅 요청 시작 ===")
    logger.info(f"{'='*80}")
    safe_info(logger, "📌 요청 모델: %s", model.upper())
    safe_info(logger, "📌 채팅 모드: %s", mode)
    safe_info(logger, "📌 사용자 ID: %s", request.user_id)
    safe_info(logger, "📌 채팅방 ID: %s", request.room_id)
    logger.info(f"📌 vLLM 서비스: {'✅ 사용 가능' if rag.vllm else '❌ 사용 불가'}")
    logger.info("")

    # 1. 일반 대화 (RAG 활용)
    if mode == ChatMode.NORMAL:
        full_response = ""

        try:
            # 히스토리 없이 단일 요청/응답 처리 (명세서 기준)
            history_dict = []

            # Determine if this is an analysis request
            user_message = request.message or ""
            is_analysis = any(
                keyword in user_message for keyword in ["분석", "매칭", "적합", "평가", "비교"]
            )

            # 면접 모드 여부 확인
            is_followup = (
                request.interview_id is not None and request.context.mode == ChatMode.INTERVIEW
            )

            if is_analysis:
                # ===================================================================
                # 분석 요청: vLLM과 Gemini 완전 분리
                # ===================================================================
                # 사용자 메시지는 로그에 포함하지 않음 (보안)
                logger.info("🔍 분석 요청 감지")
                logger.info("")

                # 채팅방 제목 추출 (회사명/채용직무)
                chat_title = ""
                try:
                    logger.info("📝 [0/3] 채팅방 제목 추출 중...")
                    # VectorDB에서 채용공고만 가져오기
                    job_posting_docs = await rag.retrieve_all_documents(
                        user_id=request.user_id, context_types=["job_posting"]
                    )

                    if job_posting_docs:
                        # 채용공고 텍스트 (앞 1000자만)
                        posting_text = job_posting_docs[:1000]
                        title_prompt = f"""{get_extract_title_prompt()}

## 채용공고 텍스트
{posting_text}
"""
                        # Gemini로 제목 추출 (빠르고 정확)
                        title_response = ""
                        async for chunk in rag.llm.generate_response(
                            user_message=title_prompt,
                            context=None,
                            history=[],
                            system_prompt="당신은 채용공고에서 회사명과 직무를 정확히 추출하는 AI입니다.",
                        ):
                            title_response += chunk

                        chat_title = title_response.strip()
                        logger.info(f"✅ [0/3] 채팅방 제목: {chat_title}")

                        # 채팅방 제목을 SSE로 전송
                        yield f"data: {json.dumps({'summary': chat_title}, ensure_ascii=False)}{sse_end}"
                    else:
                        logger.warning("⚠️ 채용공고를 찾을 수 없어 제목 추출 생략")
                except Exception as e:
                    logger.error(f"❌ 채팅방 제목 추출 실패: {e}")
                    # 실패해도 계속 진행
                logger.info("")

                # ---------------------------------------------------------------
                # vLLM 모드 (가성비): EasyOCR → VectorDB → Llama 분석
                # ---------------------------------------------------------------
                if model == "vllm" and rag.vllm:
                    logger.info("💰 [vLLM 가성비 모드] 분석 시작")
                    logger.info("   프로세스: EasyOCR → VectorDB 저장 → VectorDB 조회 → Llama 분석")
                    logger.info("")

                    # 1. VectorDB에서 OCR로 추출된 모든 문서 가져오기
                    logger.info("📂 [1/3] VectorDB에서 업로드된 문서 조회 중...")
                    full_context = await rag.retrieve_all_documents(
                        user_id=request.user_id, context_types=["resume", "job_posting"]
                    )

                    if not full_context:
                        error_msg = "❌ 업로드된 이력서 또는 채용공고를 찾을 수 없습니다.\n먼저 파일을 업로드해주세요."
                        # 사용자 ID는 로그에 포함하지 않음 (보안)
                        logger.error("⚠️ VectorDB에 문서가 없습니다")
                        yield f"data: {json.dumps({'chunk': error_msg}, ensure_ascii=False)}{sse_end}"
                        full_response = error_msg
                    else:
                        logger.info(f"✅ [1/3] VectorDB 조회 완료: {len(full_context)}자")
                        logger.info("")

                        # 2. Llama 모델로 분석
                        logger.info("🤖 [2/3] Llama 모델 분석 시작...")
                        analysis_prompt = f"""다음 이력서와 채용공고를 분석하여 아래 형식으로 응답해주세요:

{full_context}

[중요] 전체 응답은 반드시 1500자 이내로 간결하게 작성하세요.

아래 형식 그대로 출력하세요:

지원 회사 및 직무 : [회사명] | [직무명]

이력서 분석

장점
1. [구체적인 장점 1]
2. [구체적인 장점 2]
3. [구체적인 장점 3]

단점
1. [구체적인 단점 또는 보완점 1]
2. [구체적인 단점 또는 보완점 2]
3. [구체적인 단점 또는 보완점 3]

채용 공고 분석

필수 역량
- [필수 역량 1]
- [필수 역량 2]
- [필수 역량 3]

매칭도

맞는 점
- [매칭되는 역량 1]
- [매칭되는 역량 2]

보완할 점
- [부족한 역량 1]
- [부족한 역량 2]

절대 금지:
- # ## ### 제목 기호 사용 금지
- ** __ 볼드/이탤릭 기호 사용 금지
- 1500자 초과 금지

간결하게 작성하세요."""

                        async for chunk in rag.vllm.generate_response(
                            user_message=analysis_prompt,
                            context=None,
                            history=[],
                            system_prompt="당신은 채용 전문가입니다. 마크다운 문법(#, ##, **, ```)을 절대 사용하지 말고 일반 텍스트로만 응답하세요.",
                        ):
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

                        logger.info(f"✅ [3/3] Llama 분석 완료 (응답 길이: {len(full_response)}자)")

                # ---------------------------------------------------------------
                # Gemini 모드 (고성능): Gemini Vision API로 직접 파일 읽고 분석
                # ---------------------------------------------------------------
                else:
                    if model == "vllm" and not rag.vllm:
                        logger.warning("⚠️ vLLM 서비스 사용 불가 → Gemini로 자동 변경")

                    logger.info("🚀 [Gemini 고성능 모드] 분석 시작")
                    logger.info("   프로세스: RAG 검색 → Gemini 분석 (원래 방식)")
                    logger.info("")

                    # RAG로 컨텍스트 검색하여 Gemini로 분석
                    logger.info("📂 [1/2] RAG 검색 중...")
                    async for chunk in rag.chat_with_rag(
                        user_message=user_message,
                        user_id=request.user_id,
                        history=history_dict,
                        use_rag=True,
                        context_types=["resume", "job_posting"],
                        model="gemini",
                    ):
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

                    logger.info(f"✅ [2/2] Gemini 분석 완료 (응답 길이: {len(full_response)}자)")
            else:
                # ===================================================================
                # 일반 대화: RAG 검색 사용
                # ===================================================================
                logger.info("💬 일반 대화 모드")
                logger.info("")

                # 면접 세션에서 꼬리질문 생성 또는 피드백 (interview_id가 있고, 이전 질문-답변 쌍이 있는 경우)
                if is_followup:
                    # 면접 Q&A 개수 카운트 (assistant=질문, user=답변)
                    interview_qa_count = len([h for h in history_dict if h.get("role") == "user"])

                    logger.info(f"📊 현재 면접 답변 수: {interview_qa_count}개")

                    # 5번째 답변 후 → 피드백 생성
                    if interview_qa_count >= 5:
                        logger.info("🎯 [면접 종료] 5개 답변 완료 → 피드백 생성 시작")

                        # 종료 메시지
                        end_msg = "면접이 종료되었습니다. 답변 평가를 시작합니다.\n\n"
                        yield f"data: {json.dumps({'chunk': end_msg}, ensure_ascii=False)}{sse_end}"
                        full_response += end_msg

                        # Q&A 목록 생성
                        qa_pairs = []
                        for i in range(0, len(history_dict), 2):
                            if i + 1 < len(history_dict):
                                qa_pairs.append(
                                    {
                                        "question": history_dict[i].get("content", ""),
                                        "answer": history_dict[i + 1].get("content", ""),
                                    }
                                )

                        # 피드백 프롬프트
                        feedback_prompt = (
                            "다음 면접 Q&A에 대해 각 답변마다 피드백을 제공해주세요:\n\n"
                        )
                        for i, qa in enumerate(qa_pairs[:5], 1):
                            feedback_prompt += (
                                f"질문 {i}: {qa['question']}\n답변 {i}: {qa['answer']}\n\n"
                            )

                        feedback_prompt += """각 답변에 대해 다음 형식으로 피드백해주세요:

질문 1
[질문 내용]

답변 1
[답변 내용]

평가
- 잘한 점: [구체적으로]
- 개선점: [구체적으로]
- 추천 답변: [더 나은 답변 예시]

(질문 2~5도 같은 형식으로)

마크다운 문법(#, **, ```)을 사용하지 마세요."""

                        # 피드백 생성
                        async for chunk in rag.llm.generate_response(
                            user_message=feedback_prompt,
                            context=None,
                            history=[],
                            system_prompt="당신은 전문 면접관입니다. 지원자의 답변을 평가하고 구체적인 피드백을 제공합니다. 마크다운 문법을 사용하지 마세요.",
                            user_id=request.user_id,
                        ):
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

                        logger.info(
                            f"✅ [면접 피드백] 생성 완료 (응답 길이: {len(full_response)}자)"
                        )

                    else:
                        # 5개 미만 → 꼬리질문 계속 생성
                        original_question = history_dict[-2].get("content", "")
                        candidate_answer = history_dict[-1].get("content", "")

                        logger.info("🔍 [꼬리질문 생성] 감지")
                        logger.info(f"   원본 질문: {original_question[:50]}...")
                        logger.info(f"   답변: {candidate_answer[:50]}...")
                        logger.info("")

                        # 간단한 STAR 분석
                        star_analysis = {
                            "situation": "unknown",
                            "task": "unknown",
                            "action": "unknown",
                            "result": "unknown",
                        }

                        # 꼬리질문 생성
                        async for chunk in rag.generate_followup_question(
                            original_question=original_question,
                            candidate_answer=candidate_answer,
                            star_analysis=star_analysis,
                            model=model,
                            user_id=request.user_id,
                        ):
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

                        logger.info(f"✅ [꼬리질문 생성] 완료 (응답 길이: {len(full_response)}자)")

                else:
                    # 일반 대화 또는 면접 질문 요청
                    if (
                        "면접 질문" in user_message
                        or "면접질문" in user_message
                        or "면접" in user_message
                    ):
                        # 면접 질문 요청 시 portfolio(면접 질문 데이터)만 검색
                        context_types = ["portfolio"]
                        logger.info("🎯 면접 질문 요청 감지 → portfolio 컬렉션만 검색")
                    else:
                        # 일반 요청 시 모든 컬렉션 검색
                        context_types = ["resume", "job_posting", "portfolio"]
                        logger.info("📚 일반 대화 → 모든 컬렉션 검색")

                    logger.info("")

                    # RAG를 사용하여 컨텍스트 검색 및 응답 생성
                    # 모델명은 로그에 포함하지 않음 (보안)
                    logger.info("🔍 RAG 검색 및 응답 생성 시작...")
                    async for chunk in rag.chat_with_rag(
                        user_message=user_message,
                        user_id=request.user_id,
                        history=history_dict,
                        use_rag=True,  # RAG 활성화
                        context_types=context_types,
                        model=model,
                    ):
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

                    # 모델명은 로그에 포함하지 않음 (보안)
                    logger.info("✅ 일반 대화 완료 (응답 길이: %d자)", len(full_response))

        except Exception as e:
            error_msg = f"오류가 발생했습니다: {str(e)}"
            yield f"data: {json.dumps({'chunk': error_msg}, ensure_ascii=False)}{sse_end}"
            full_response = error_msg

        yield f"data: [DONE]{sse_end}"

    # 2. 면접 모드 - 5개 질문 × 최대 3 depths 꼬리질문
    elif mode == ChatMode.INTERVIEW:
        try:
            interview_type = request.context.interview_type or "tech"
            interview_type_kr = "기술" if interview_type == "tech" else "인성"

            # 세션 캐시 키 생성
            session_key = get_session_key(request.user_id, request.interview_id)
            user_message = request.message or ""

            # 세션 상태 확인 (요청에 없으면 캐시에서 조회)
            session = request.context.interview_session
            if session is None:
                session = interview_sessions.get(session_key)
                if session:
                    safe_info(
                        logger,
                        "📦 [면접] 캐시에서 세션 복원: %s, phase=%s, Q%d/5",
                        session_key,
                        session.phase,
                        session.current_question_id,
                    )

            # vLLM 또는 Gemini 선택
            model_choice = (
                request.model.value if hasattr(request.model, "value") else str(request.model)
            )

            # ===================================================================
            # PHASE 1: 세션 초기화 (첫 요청 시 5개 질문 세트 생성)
            # ===================================================================
            if session is None or session.phase == "init":
                logger.info("🎯 [면접] 세션 초기화 - 5개 질문 세트 생성 시작")

                # RAG로 사용자 정보 검색
                context = await rag.retrieve_context(
                    query=f"{interview_type_kr} 면접 질문을 위한 사용자 정보",
                    user_id=request.user_id,
                    context_types=["resume", "portfolio", "job_posting"],
                    n_results=3,
                )

                resume_ocr = request.context.resume_ocr if request.context else None
                job_posting_ocr = request.context.job_posting_ocr if request.context else None
                portfolio_text = request.context.portfolio_text if request.context else None

                # 5개 질문 세트 생성 프롬프트
                init_prompt = create_tech_interview_init_prompt(
                    resume_text=resume_ocr or context or "정보 없음",
                    job_posting_text=job_posting_ocr or "정보 없음",
                    portfolio_text=portfolio_text or context or "정보 없음",
                )

                # 방안 2: 비스트리밍으로 JSON 생성 (빠름) → 질문을 스트리밍으로 출력 (타이핑 효과)
                system_prompt = get_system_tech_interview()

                if model_choice == "vllm" and rag.vllm:
                    # vLLM은 기존 스트리밍 방식 유지 (비스트리밍 미지원)
                    full_response = ""
                    async for chunk in rag.vllm.generate_response(
                        user_message=init_prompt,
                        context=None,
                        history=[],
                        system_prompt=system_prompt,
                    ):
                        full_response += chunk
                else:
                    # Gemini: 비스트리밍으로 JSON 빠르게 생성
                    full_response = await rag.llm.generate_response_non_stream(
                        user_message=init_prompt,
                        context=None,
                        system_prompt=system_prompt,
                        user_id=request.user_id,
                    )

                # JSON 파싱하여 세션 생성
                try:
                    # 마크다운 코드블록 제거 및 JSON 추출
                    import re

                    # ```json ... ``` 형식 제거
                    json_content = re.sub(r"```json\s*", "", full_response)
                    json_content = re.sub(r"```\s*$", "", json_content)
                    json_content = json_content.strip()

                    # JSON 부분만 추출 (중괄호 기준)
                    json_start = json_content.find("{")
                    json_end = json_content.rfind("}") + 1

                    if json_start != -1 and json_end > json_start:
                        json_str = json_content[json_start:json_end]

                        # 디버깅용 로그 (실제 JSON 문자열 일부 출력)
                        logger.info(f"JSON 파싱 시도 (첫 200자): {json_str[:200]}")

                        questions_data = json.loads(json_str)

                        # 세션 생성
                        new_session = InterviewSession(
                            session_id=str(uuid.uuid4()),
                            interview_type=interview_type,
                            questions=[
                                InterviewQuestionState(
                                    id=q["id"],
                                    category=q["category"],
                                    category_name=q["category_name"],
                                    question=q["question"],
                                    intent=q.get("intent", ""),
                                    keywords=q.get("keywords", []),
                                )
                                for q in questions_data.get("questions", [])
                            ],
                            current_question_id=1,
                            phase="questioning",
                        )

                        logger.info("✅ 면접 질문 세트 생성 완료: %d개", len(new_session.questions))

                        # 세션 캐시에 저장
                        interview_sessions[session_key] = new_session
                        safe_info(logger, "💾 [면접] 세션 캐시 저장: %s", session_key)

                        # 첫 번째 질문 출력 (헤더: [기술면접 1/5]) - 타이핑 효과
                        first_q = new_session.questions[0] if new_session.questions else None
                        if first_q:
                            question_text = (
                                f"[{interview_type_kr}면접 1/5]{newline}{first_q.question}"
                            )
                            # 방안 2: 질문을 한 글자씩 스트리밍 출력 (타이핑 효과)
                            for char in question_text:
                                yield f"data: {json.dumps({'chunk': char}, ensure_ascii=False)}{sse_end}"
                                await asyncio.sleep(0.015)  # 15ms 딜레이로 타이핑 효과

                            # 세션 상태 전달 (메타데이터로)
                            session_meta = {
                                "type": "session_state",
                                "session": new_session.model_dump(),
                            }
                            yield f"data: {json.dumps(session_meta, ensure_ascii=False)}{sse_end}"
                    else:
                        raise ValueError("JSON 형식을 찾을 수 없습니다")

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"질문 세트 파싱 실패: {e}")
                    logger.error(f"원본 응답 (첫 500자): {full_response[:500]}")

                    # SSE 에러 이벤트 전송 (HTTP 200 유지, payload에 status=500)
                    error_response = {
                        "type": "error",
                        "error": {
                            "code": "PARSE_FAILED",
                            "status": 500,
                        },
                        "fallback": "면접 질문 세트 생성 중 오류가 발생했습니다. 다시 시도해주세요.",
                    }
                    yield f"data: {json.dumps(error_response, ensure_ascii=False)}{sse_end}"

                yield f"data: [DONE]{sse_end}"

            # ===================================================================
            # PHASE 2: 꼬리질문 또는 다음 질문 생성 (답변 수신 후)
            # ===================================================================
            elif session.phase in ["questioning", "followup"]:
                current_q_id = session.current_question_id
                current_q = next((q for q in session.questions if q.id == current_q_id), None)

                if not current_q:
                    yield f"data: {json.dumps({'chunk': '세션 오류: 현재 질문을 찾을 수 없습니다.'}, ensure_ascii=False)}{sse_end}"
                    yield f"data: [DONE]{sse_end}"
                    return

                # 지원자 답변을 대화 이력에 추가
                current_q.conversation.append(
                    {
                        "role": "candidate",
                        "content": user_message,
                    }
                )
                current_q.current_depth += 1

                # 꼬리질문 생성 여부 판단 (최대 3 depths)
                if current_q.current_depth < current_q.max_depth:
                    # 꼬리질문 생성
                    followup_prompt = create_tech_followup_prompt(
                        question_id=current_q.id,
                        category_name=current_q.category_name,
                        original_question=current_q.question,
                        conversation_history=format_conversation_history(current_q.conversation),
                        last_answer=user_message,
                        current_depth=current_q.current_depth,
                    )

                    full_response = ""
                    system_prompt = get_system_tech_interview()

                    if model_choice == "vllm" and rag.vllm:
                        async for chunk in rag.vllm.generate_response(
                            user_message=followup_prompt,
                            context=None,
                            history=[],
                            system_prompt=system_prompt,
                        ):
                            full_response += chunk
                    else:
                        async for chunk in rag.llm.generate_response(
                            user_message=followup_prompt,
                            context=None,
                            history=[],
                            system_prompt=system_prompt,
                            user_id=request.user_id,
                        ):
                            full_response += chunk

                    # 꼬리질문 응답 파싱
                    try:
                        json_start = full_response.find("{")
                        json_end = full_response.rfind("}") + 1
                        if json_start != -1 and json_end > json_start:
                            followup_data = json.loads(full_response[json_start:json_end])

                            if followup_data.get("should_continue", True) and followup_data.get(
                                "followup"
                            ):
                                # 꼬리질문 출력 (헤더: [기술면접 2-1/5] 형식)
                                followup_q = followup_data["followup"]["question"]
                                current_q.conversation.append(
                                    {
                                        "role": "interviewer",
                                        "content": followup_q,
                                    }
                                )
                                session.phase = "followup"

                                # 꼬리질문 헤더: [기술면접 {질문번호}-{꼬리질문번호}/5] - 타이핑 효과
                                followup_header = f"[{interview_type_kr}면접 {current_q_id}-{current_q.current_depth}/5]"
                                followup_text = f"{followup_header}{newline}{followup_q}"
                                # 한 글자씩 스트리밍 출력 (타이핑 효과)
                                for char in followup_text:
                                    yield f"data: {json.dumps({'chunk': char}, ensure_ascii=False)}{sse_end}"
                                    await asyncio.sleep(0.015)
                            else:
                                # 다음 주제로 이동
                                current_q.is_completed = True
                    except json.JSONDecodeError as e:
                        # 꼬리질문 파싱 실패 - SSE 에러 전송 후 다음 질문으로 이동
                        logger.error(f"꼬리질문 파싱 실패: {e}")
                        error_response = {
                            "type": "error",
                            "error": {
                                "code": "PARSE_FAILED",
                                "status": 500,
                            },
                            "fallback": "꼬리질문 생성 중 오류가 발생했습니다. 다음 질문으로 넘어갑니다.",
                        }
                        yield f"data: {json.dumps(error_response, ensure_ascii=False)}{sse_end}"
                        current_q.is_completed = True

                else:
                    # 최대 깊이 도달 - 다음 주제로 이동
                    current_q.is_completed = True

                # 다음 질문으로 이동해야 하는 경우
                if current_q.is_completed:
                    next_q_id = current_q_id + 1
                    if next_q_id <= session.total_questions:
                        next_q = next((q for q in session.questions if q.id == next_q_id), None)
                        if next_q:
                            session.current_question_id = next_q_id
                            session.phase = "questioning"

                            # 다음 질문 출력 (헤더: [기술면접 2/5] 형식) - 타이핑 효과
                            question_header = f"[{interview_type_kr}면접 {next_q_id}/5]"
                            question_text = f"{question_header}{newline}{next_q.question}"
                            # 한 글자씩 스트리밍 출력 (타이핑 효과)
                            for char in question_text:
                                yield f"data: {json.dumps({'chunk': char}, ensure_ascii=False)}{sse_end}"
                                await asyncio.sleep(0.015)

                            next_q.conversation.append(
                                {
                                    "role": "interviewer",
                                    "content": next_q.question,
                                }
                            )
                    else:
                        # 모든 질문 완료 - 타이핑 효과
                        session.phase = "completed"
                        complete_msg = f"{newline}{newline}면접 결과 리포트를 생성 중입니다. 잠시만 기다려 주세요."
                        for char in complete_msg:
                            yield f"data: {json.dumps({'chunk': char}, ensure_ascii=False)}{sse_end}"
                            await asyncio.sleep(0.015)

                # 세션 캐시 업데이트 (PHASE 2 블록 내부, if current_q.is_completed 외부)
                interview_sessions[session_key] = session
                safe_info(
                    logger,
                    "💾 [면접] 세션 캐시 업데이트: %s, phase=%s, Q%d/5",
                    session_key,
                    session.phase,
                    session.current_question_id,
                )

                # 면접 완료 시 세션 정리
                if session.phase == "completed":
                    interview_sessions.pop(session_key, None)
                    safe_info(logger, "🗑️ [면접] 완료된 세션 삭제: %s", session_key)

                # 업데이트된 세션 상태 전달
                session_meta = {
                    "type": "session_state",
                    "session": session.model_dump(),
                }
                yield f"data: {json.dumps(session_meta, ensure_ascii=False)}{sse_end}"
                yield f"data: [DONE]{sse_end}"

            # ===================================================================
            # PHASE 3: 면접 완료
            # ===================================================================
            elif session.phase == "completed":
                # 이미 완료된 세션 - 캐시에서 삭제
                interview_sessions.pop(session_key, None)
                complete_msg = "면접이 이미 완료되었습니다. 리포트 모드에서 결과를 확인하세요."
                yield f"data: {json.dumps({'chunk': complete_msg}, ensure_ascii=False)}{sse_end}"
                yield f"data: [DONE]{sse_end}"

        except Exception as e:
            logger.error(f"Interview error: {e}", exc_info=True)
            error_msg = f"면접 진행 중 오류가 발생했습니다: {str(e)}"
            yield f"data: {json.dumps({'chunk': error_msg}, ensure_ascii=False)}{sse_end}"
            yield f"data: [DONE]{sse_end}"

    # 3. 리포트 모드 - 면접 평가 리포트 생성
    elif mode == ChatMode.REPORT:
        try:
            interview_type = request.context.interview_type or "tech"
            interview_type_kr = "기술" if interview_type == "tech" else "인성"
            qa_list = request.context.qa_list or []

            if not qa_list:
                yield f"data: [DONE]{sse_end}"
                return

            content = f"{interview_type_kr} 면접 평가 리포트를 생성 중입니다...{newline}"
            yield f"data: {json.dumps({'chunk': content}, ensure_ascii=False)}{sse_end}"

            # Q&A 목록을 텍스트로 변환
            qa_text = ""
            for i, qa in enumerate(qa_list, 1):
                q = qa.get("question", "")
                a = qa.get("answer", "")
                qa_text += f"질문 {i}: {q}\n답변 {i}: {a}\n\n"

            # 평가 리포트 프롬프트
            report_prompt = f"""
다음은 {interview_type_kr} 면접 Q&A 기록입니다:

{qa_text}

위 면접 내용을 바탕으로 상세한 평가 리포트를 작성해주세요.
다음 항목을 포함해주세요:
1. 각 답변에 대한 개별 평가 (잘한 점, 개선점)
2. 전체적인 강점 패턴
3. 전체적인 약점 패턴
4. 향후 학습 가이드
"""

            full_report = ""

            # vLLM 또는 Gemini 선택
            model_choice = (
                request.model.value if hasattr(request.model, "value") else str(request.model)
            )

            if model_choice == "vllm" and rag.vllm:
                logger.info("📊 [vLLM] 면접 평가 리포트 생성 시작")
                async for chunk in rag.vllm.generate_response(
                    user_message=report_prompt,
                    context=None,
                    history=[],
                    system_prompt="당신은 면접 평가 전문가입니다. 상세하고 건설적인 피드백을 제공합니다.",
                ):
                    full_report += chunk
                    yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"
            else:
                logger.info("📊 [Gemini] 면접 평가 리포트 생성 시작")
                async for chunk in rag.llm.generate_response(
                    user_message=report_prompt,
                    context=None,
                    history=[],
                    system_prompt="당신은 면접 평가 전문가입니다. 상세하고 건설적인 피드백을 제공합니다.",
                    user_id=request.user_id,
                ):
                    full_report += chunk
                    yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

            {
                "success": True,
                "mode": "report",
                "response": full_report.strip(),
                "interview_type": interview_type,
                "interview_id": request.interview_id,
            }
            yield f"data: [DONE]{sse_end}"

        except Exception as e:
            logger.error(f"Interview report generation error: {e}")
            {"success": False, "mode": "report", "error": str(e)}
            yield f"data: [DONE]{sse_end}"

    # Latency 측정 종료 및 전송
    try:
        duration = (time.time() - start_time) * 1000
        cw = CloudWatchService.get_instance()
        asyncio.create_task(cw.put_metric("AI_Chat_Latency", duration, "Milliseconds", dims))
    except Exception as e:
        logger.error(f"Failed to record latency metric: {e}")


@router.post(
    "/chat",
    summary="채팅 스트리밍 (일반/면접/리포트)",
    description="""
    채팅 스트리밍 응답을 처리합니다.

    **처리 방식:** 스트리밍 (SSE)

    **모드:**
    - normal: 일반 대화
    - interview: 면접 질문 생성
    - report: 면접 평가 리포트 생성

    **면접 타입 (context.interview_type):**
    - behavior: 인성 면접
    - tech: 기술 면접
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
                                    "message": "room_id는 필수입니다",
                                    "field": "room_id",
                                }
                            }
                        },
                        "invalid_mode": {
                            "value": {
                                "detail": {
                                    "code": "INVALID_MODE",
                                    "message": "mode는 normal 또는 interview 중 하나여야 합니다",
                                    "field": "context.mode",
                                }
                            }
                        },
                        "invalid_interview_type": {
                            "value": {
                                "detail": {
                                    "code": "INVALID_INTERVIEW_TYPE",
                                    "message": "interview_type은 behavior 또는 tech만 가능합니다",
                                    "field": "context.interview_type",
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
        422: {
            "description": "Unprocessable Entity",
            "content": {
                "application/json": {
                    "example": {
                        "detail": {
                            "code": "MISSING_CONTEXT",
                            "message": "면접 모드 시 context.resume_ocr 또는 context.job_posting_ocr이 필요합니다",
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
                    "example": {
                        "detail": {
                            "code": "LLM_UNAVAILABLE",
                            "message": "AI 서비스에 연결할 수 없습니다",
                        }
                    }
                }
            },
        },
    },
)
async def chat(request: ChatRequest):
    """채팅 처리 (일반/면접)"""
    return StreamingResponse(
        generate_chat_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # nginx 버퍼링 비활성화
        },
    )


# ============================================================================
# API 3: 캘린더 일정 파싱 (동기)
# ============================================================================


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
    # validator에서 이미 검증됨

    # Pydantic AI로 구조화된 결과 반환
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


# ============================================================================
# API 4: 게시판 첨부파일 마스킹 (비동기)
# ============================================================================
# 이 API는 app/api/routes/masking.py로 이동되었습니다.
# masking.py에서 파일 기반 저장소와 실제 Gemini API를 사용합니다.
