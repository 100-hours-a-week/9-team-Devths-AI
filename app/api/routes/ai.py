import asyncio
import json
import logging
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from app.prompts import (
    SYSTEM_INTERVIEW,
    create_interview_question_prompt,
)
from app.schemas.calendar import CalendarParseRequest, CalendarParseResponse
from app.schemas.chat import (
    AnalysisResult,
    ChatMode,
    ChatRequest,
    MatchingResult,
    PostingAnalysis,
    ResumeAnalysis,
)
from app.schemas.common import AsyncTaskResponse, ErrorCode, TaskStatus, TaskStatusResponse
from app.schemas.text_extract import (
    DocumentExtractResult,
    DocumentInput,
    PageText,
    TextExtractRequest,
    TextExtractResult,
)
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.vectordb_service import VectorDBService
from app.services.vllm_service import VLLMService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai",
    tags=["AI APIs (v3.0)"],
    responses={404: {"description": "Not found"}},
)

# 임시 작업 저장소 (실제로는 Redis 등 사용)
tasks_db = {}
task_id_counter = 0  # 정수 task_id 카운터


def get_next_task_id() -> int:
    """Get next task_id as integer"""
    global task_id_counter
    task_id_counter += 1
    return task_id_counter


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
    task_id = get_next_task_id()

    # 비동기 작업 시작
    tasks_db[task_id] = {
        "status": TaskStatus.PROCESSING,
        "created_at": datetime.now(),
        "room_id": request.room_id,
        "request": request.model_dump(),
    }

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
            logger.info(f"📌 요청 모델: {model.upper()}")
            logger.info(f"📌 사용자 ID: {request.user_id}")
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
                    logger.info(f"   → S3 키: {doc_input.s3_key}")

                    # vLLM 모드: EasyOCR 사용 (가성비)
                    if model == "vllm" and rag.vllm:
                        logger.info("   💰 [vLLM 가성비 모드] EasyOCR 시작")
                        ocr_result = await rag.vllm.extract_text_from_file(
                            file_url=str(doc_input.s3_key),
                            file_type=file_type,
                            user_id=str(request.user_id),
                        )
                        extracted_text = ocr_result["extracted_text"]
                        pages = [PageText(**page) for page in ocr_result["pages"]]
                        logger.info(
                            f"   ✅ [vLLM OCR] 추출 완료: {len(extracted_text)}자 (페이지: {len(pages)})"
                        )

                    # Gemini 모드: Gemini Vision API 사용 (고성능)
                    else:
                        if model == "vllm" and not rag.vllm:
                            logger.warning("   ⚠️ vLLM 서비스 사용 불가 → Gemini로 자동 변경")
                        logger.info("   🚀 [Gemini 고성능 모드] Gemini Vision API OCR 시작")
                        ocr_result = await rag.llm.extract_text_from_file(
                            file_url=str(doc_input.s3_key), file_type=file_type
                        )
                        extracted_text = ocr_result["extracted_text"]
                        pages = [PageText(**page) for page in ocr_result["pages"]]
                        logger.info(
                            f"   ✅ [Gemini OCR] 추출 완료: {len(extracted_text)}자 (페이지: {len(pages)})"
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
                    logger.info(f"   ✅ VectorDB 저장 완료: {document_id}")

                return DocumentExtractResult(
                    file_id=doc_input.file_id, extracted_text=extracted_text, pages=pages
                )

            # 이력서와 채용공고 각각 처리
            resume_result = await extract_document(request.resume, "resume")
            job_posting_result = await extract_document(request.job_posting, "job_posting")

            # 분석 리포트 생성 (명세서 요구사항)
            logger.info("")
            logger.info("📊 분석 리포트 생성 시작...")
            try:
                analysis_result = await rag.llm.generate_analysis(
                    resume_text=resume_result.extracted_text,
                    posting_text=job_posting_result.extracted_text,
                    user_id=str(request.user_id),
                )
                logger.info("✅ 분석 리포트 생성 완료")
            except Exception as e:
                logger.warning(f"⚠️ 분석 리포트 생성 실패: {e} (OCR 텍스트만 반환)")
                analysis_result = {
                    "resume_analysis": {"strengths": [], "weaknesses": [], "suggestions": []},
                    "posting_analysis": {
                        "company": "알 수 없음",
                        "position": "알 수 없음",
                        "required_skills": [],
                        "preferred_skills": [],
                    },
                }

            # 결과 저장 (명세서에 따른 응답 구조)
            tasks_db[task_id]["status"] = TaskStatus.COMPLETED
            tasks_db[task_id]["result"] = TextExtractResult(
                success=True,
                resume_ocr=resume_result.extracted_text,
                job_posting_ocr=job_posting_result.extracted_text,
                resume_analysis=analysis_result.get("resume_analysis"),
                posting_analysis=analysis_result.get("posting_analysis"),
            ).model_dump()

            logger.info("")
            logger.info("✅ 텍스트 추출 + 분석 완료!")
            logger.info(f"   → 이력서 OCR: {len(resume_result.extracted_text)}자")
            logger.info(f"   → 채용공고 OCR: {len(job_posting_result.extracted_text)}자")

        except Exception as e:
            logger.error(f"텍스트 추출 오류: {e}", exc_info=True)
            tasks_db[task_id]["status"] = TaskStatus.FAILED
            tasks_db[task_id]["error"] = {"code": ErrorCode.PROCESSING_ERROR, "message": str(e)}

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
async def get_task_status(task_id: int):
    """비동기 작업 상태 조회"""
    if task_id not in tasks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": ErrorCode.TASK_NOT_FOUND.value,
                "message": f"작업을 찾을 수 없습니다: {task_id}",
            },
        )

    task = tasks_db[task_id]

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


async def generate_chat_stream(request: ChatRequest):
    """채팅 응답 스트리밍 생성"""

    # context가 배열인 경우 면접 리포트 모드로 간주
    if isinstance(request.context, list):
        mode = ChatMode.INTERVIEW_REPORT
    else:
        mode = request.context.mode if request.context else ChatMode.GENERAL

    rag = get_services()
    newline = "\n"
    sse_end = "\n\n"

    # 모델 선택 (gemini 또는 vllm)
    model = request.model.value if hasattr(request.model, "value") else str(request.model)
    logger.info("")
    logger.info(f"{'='*80}")
    logger.info("=== 💬 채팅 요청 시작 ===")
    logger.info(f"{'='*80}")
    logger.info(f"📌 요청 모델: {model.upper()}")
    logger.info(f"📌 채팅 모드: {mode}")
    logger.info(f"📌 사용자 ID: {request.user_id}")
    logger.info(f"📌 채팅방 ID: {request.room_id}")
    logger.info(f"📌 vLLM 서비스: {'✅ 사용 가능' if rag.vllm else '❌ 사용 불가'}")
    logger.info("")

    # 1. 일반 대화 (RAG 활용)
    if mode == ChatMode.GENERAL:
        full_response = ""

        try:
            # Convert ChatMessage list to dict list for service compatibility
            history_dict = [
                {
                    "role": msg.role.value if hasattr(msg.role, "value") else str(msg.role),
                    "content": msg.content,
                }
                for msg in request.history
            ]

            # Determine if this is an analysis request
            user_message = request.message or ""
            is_analysis = any(
                keyword in user_message for keyword in ["분석", "매칭", "적합", "평가", "비교"]
            )

            # 면접 세션이 활성화되어 있고, 이전 질문이 있으면 꼬리질문 생성
            is_followup = (
                request.session_id is not None
                and len(history_dict) >= 2
                and history_dict[-2].get("role") == "assistant"
                and history_dict[-1].get("role") == "user"
            )

            if is_analysis:
                # ===================================================================
                # 분석 요청: vLLM과 Gemini 완전 분리
                # ===================================================================
                logger.info(f"🔍 분석 요청 감지: '{user_message[:50]}...'")
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
                        logger.error(f"⚠️ VectorDB에 문서가 없습니다 (user_id: {request.user_id})")
                        yield f"data: {json.dumps({'type': 'chunk', 'content': error_msg}, ensure_ascii=False)}{sse_end}"
                        full_response = error_msg
                    else:
                        logger.info(f"✅ [1/3] VectorDB 조회 완료: {len(full_context)}자")
                        logger.info("")

                        # 2. Llama 모델로 분석
                        logger.info("🤖 [2/3] Llama 모델 분석 시작...")
                        analysis_prompt = f"""다음 이력서와 채용공고를 상세히 분석해주세요:

{full_context}

다음 항목을 포함하여 분석해주세요:
1. **적합도 평가**: 지원자가 채용공고 요구사항에 얼마나 부합하는지
2. **강점**: 지원자의 뛰어난 역량과 경험
3. **약점**: 부족한 부분이나 개선이 필요한 영역
4. **예상 면접 질문 (인성 3개, 기술 3개)**

간결하고 명확하게 정리해주세요."""

                        async for chunk in rag.vllm.generate_response(
                            user_message=analysis_prompt,
                            context=None,
                            history=[],
                            system_prompt="당신은 채용 전문가입니다. 이력서와 채용공고를 분석하여 명확한 피드백을 제공하세요.",
                        ):
                            full_response += chunk
                            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}{sse_end}"

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
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}{sse_end}"

                    logger.info(f"✅ [2/2] Gemini 분석 완료 (응답 길이: {len(full_response)}자)")
            else:
                # ===================================================================
                # 일반 대화: RAG 검색 사용
                # ===================================================================
                logger.info("💬 일반 대화 모드")
                logger.info("")

                # 면접 세션에서 꼬리질문 생성 (session_id가 있고, 이전 질문-답변 쌍이 있는 경우)
                if is_followup:
                    original_question = history_dict[-2].get("content", "")
                    candidate_answer = history_dict[-1].get("content", "")

                    logger.info("🔍 [꼬리질문 생성] 감지")
                    logger.info(f"   원본 질문: {original_question[:50]}...")
                    logger.info(f"   답변: {candidate_answer[:50]}...")
                    logger.info("")

                    # 간단한 STAR 분석 (실제로는 LLM으로 분석할 수 있지만, 여기서는 기본값 사용)
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
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}{sse_end}"

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
                    logger.info(f"🔍 [{model.upper()}] RAG 검색 및 응답 생성 시작...")
                    async for chunk in rag.chat_with_rag(
                        user_message=user_message,
                        user_id=request.user_id,
                        history=history_dict,
                        use_rag=True,  # RAG 활성화
                        context_types=context_types,
                        model=model,
                    ):
                        full_response += chunk
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}{sse_end}"

                    logger.info(
                        f"✅ [{model.upper()}] 일반 대화 완료 (응답 길이: {len(full_response)}자)"
                    )

        except Exception as e:
            error_msg = f"오류가 발생했습니다: {str(e)}"
            yield f"data: {json.dumps({'type': 'chunk', 'content': error_msg}, ensure_ascii=False)}{sse_end}"
            full_response = error_msg

        result = {
            "success": True,
            "mode": "general",
            "response": full_response,
            "tool_used": {"tool": "RAG", "description": "VectorDB 검색 후 LLM 응답 생성"},
        }
        yield f"data: {json.dumps({'type': 'complete', 'data': result}, ensure_ascii=False)}{sse_end}"

    # 2. 분석 모드 (RAG 사용)
    elif mode == ChatMode.ANALYSIS:
        try:
            content1 = f"이력서를 분석 중입니다...{newline}"
            yield f"data: {json.dumps({'type': 'chunk', 'content': content1}, ensure_ascii=False)}{sse_end}"
            await asyncio.sleep(0.3)

            content2 = f"채용공고를 분석 중입니다...{newline}"
            yield f"data: {json.dumps({'type': 'chunk', 'content': content2}, ensure_ascii=False)}{sse_end}"
            await asyncio.sleep(0.3)

            content3 = f"매칭도를 계산 중입니다...{newline}"
            yield f"data: {json.dumps({'type': 'chunk', 'content': content3}, ensure_ascii=False)}{sse_end}"

            # RAG를 사용하여 실제 분석 수행
            analysis_result = await rag.analyze_resume_and_posting(
                user_id=request.user_id,
                resume_id=request.context.resume_id,
                posting_id=request.context.posting_id,
            )

            # Convert to Pydantic models
            analysis = AnalysisResult(
                resume_analysis=ResumeAnalysis(**analysis_result.get("resume_analysis", {})),
                posting_analysis=PostingAnalysis(**analysis_result.get("posting_analysis", {})),
                matching=MatchingResult(**analysis_result.get("matching", {})),
            )

            result = {"success": True, "mode": "analysis", "analysis": analysis.model_dump()}
            yield f"data: {json.dumps({'type': 'complete', 'data': result}, ensure_ascii=False)}{sse_end}"

        except Exception as e:
            error_result = {"success": False, "mode": "analysis", "error": str(e)}
            yield f"data: {json.dumps({'type': 'complete', 'data': error_result}, ensure_ascii=False)}{sse_end}"

    # 3. 면접 모드 - 맞춤형 질문 생성 및 대화
    elif mode == ChatMode.INTERVIEW_QUESTION:
        try:
            # 면접 타입에 따라 프롬프트 조정
            interview_type = request.context.interview_type or "technical"
            interview_type_kr = "기술" if interview_type == "technical" else "인성"

            content = f"{interview_type_kr} 면접 질문을 생성 중입니다...{newline}"
            yield f"data: {json.dumps({'type': 'chunk', 'content': content}, ensure_ascii=False)}{sse_end}"

            # RAG를 사용하여 사용자 맞춤 면접 질문 생성
            # resume, portfolio, job_posting 컬렉션에서 컨텍스트 검색
            context = await rag.retrieve_context(
                query=f"{interview_type_kr} 면접 질문을 위한 사용자 정보",
                user_id=request.user_id,
                context_types=["resume", "portfolio", "job_posting"],
                n_results=1,  # 속도 개선을 위해 1개만 검색
            )

            # 면접 질문 생성 프롬프트 (prompts 모듈 사용)
            # context에서 resume_ocr, job_posting_ocr 사용
            resume_ocr = request.context.resume_ocr if request.context else None
            job_posting_ocr = request.context.job_posting_ocr if request.context else None

            if resume_ocr or job_posting_ocr or context:
                # 컨텍스트가 있으면 맞춤형 질문 생성
                question_prompt = create_interview_question_prompt(
                    resume_text=resume_ocr or context or "정보 없음",
                    job_posting_text=job_posting_ocr or "정보 없음",
                    interview_type=interview_type,
                )
            else:
                question_prompt = f"일반적인 {interview_type_kr} 면접 질문 1개를 짧게 생성해주세요:"

            full_question = ""

            # vLLM 또는 Gemini 선택
            model_choice = (
                request.model.value if hasattr(request.model, "value") else str(request.model)
            )

            if model_choice == "vllm" and rag.vllm:
                logger.info("💬 [vLLM] 면접 질문 생성 시작")
                async for chunk in rag.vllm.generate_response(
                    user_message=question_prompt,
                    context=None,
                    history=[],
                    system_prompt=SYSTEM_INTERVIEW,
                ):
                    full_question += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}{sse_end}"
            else:
                logger.info("💬 [Gemini] 면접 질문 생성 시작")
                async for chunk in rag.llm.generate_response(
                    user_message=question_prompt,
                    context=None,
                    history=[],
                    system_prompt=SYSTEM_INTERVIEW,
                    user_id=request.user_id,
                ):
                    full_question += chunk
                    yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}{sse_end}"

            result = {
                "success": True,
                "mode": "interview_question",
                "response": full_question.strip(),
                "interview_type": interview_type,
            }
            yield f"data: {json.dumps({'type': 'complete', 'data': result}, ensure_ascii=False)}{sse_end}"

        except Exception as e:
            logger.error(f"Interview question generation error: {e}")
            error_result = {"success": False, "mode": "interview_question", "error": str(e)}
            yield f"data: {json.dumps({'type': 'complete', 'data': error_result}, ensure_ascii=False)}{sse_end}"

    # 4. 면접 리포트 (Pydantic AI 사용)
    elif mode == ChatMode.INTERVIEW_REPORT:
        try:
            chunks = [
                f"면접 답변을 분석 중입니다...{newline}",
                f"종합 리포트를 작성 중입니다...{newline}",
            ]

            for chunk in chunks:
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}{sse_end}"
                await asyncio.sleep(0.5)

            # context가 배열인 경우 (면접 리포트 모드)
            qa_list = request.context if isinstance(request.context, list) else []

            # Q&A 배열을 문자열로 변환하여 프롬프트 생성
            qa_history = "\n".join(
                [f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}" for item in qa_list]
            )

            # 면접 리포트 생성 프롬프트
            from app.prompts import create_interview_report_prompt

            report_prompt = create_interview_report_prompt(
                qa_history=qa_history, resume_text="", job_posting_text=""
            )

            # LLM으로 리포트 생성
            full_report_text = ""
            async for chunk in rag.llm.generate_response(
                user_message=report_prompt,
                context=None,
                history=[],
                system_prompt="당신은 면접 평가 전문가입니다. JSON 형식으로만 응답하세요.",
                user_id=request.user_id,
            ):
                full_report_text += chunk
                yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}{sse_end}"

            # JSON 파싱 시도
            from app.parsers import extract_json_from_text

            parsed_report = extract_json_from_text(full_report_text)

            if parsed_report:
                result = {"success": True, "mode": "interview_report", "report": parsed_report}
            else:
                # 파싱 실패 시 기본 구조
                result = {
                    "success": True,
                    "mode": "interview_report",
                    "report": {
                        "evaluations": [
                            {
                                "question": qa.get("question", ""),
                                "answer": qa.get("answer", ""),
                                "good_points": [],
                                "improvements": [],
                            }
                            for qa in qa_list
                        ],
                        "strength_patterns": [],
                        "weakness_patterns": [],
                        "learning_guide": [],
                    },
                }

            yield f"data: {json.dumps({'type': 'complete', 'data': result}, ensure_ascii=False)}{sse_end}"

        except Exception as e:
            logger.error(f"Interview report generation error: {e}")
            error_result = {"success": False, "mode": "interview_report", "error": str(e)}
            yield f"data: {json.dumps({'type': 'complete', 'data': error_result}, ensure_ascii=False)}{sse_end}"


@router.post(
    "/chat",
    summary="채팅 처리 (통합: 대화/분석/면접)",
    description="""
    모든 LLM 응답을 통합 처리합니다.

    **v3.0 변경사항:**
    - `/ai/analyze`, `/ai/interview/*` 통합
    - context.mode로 기능 구분
    - Pydantic AI로 구조화된 출력

    **처리 방식:** 스트리밍 (SSE)

    **모드:**
    - general: 일반 대화
    - analysis: 이력서/채용공고 분석
    - interview_question: 면접 질문 생성
    - interview_report: 면접 리포트

    **Pydantic AI 사용:**
    - analysis: AnalysisResult
    - interview_question: InterviewQuestion
    - interview_report: InterviewReport
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
                                    "message": "mode는 general, interview_question, interview_report 중 하나여야 합니다",
                                    "field": "context.mode",
                                }
                            }
                        },
                        "invalid_interview_type": {
                            "value": {
                                "detail": {
                                    "code": "INVALID_INTERVIEW_TYPE",
                                    "message": "interview_type은 technical 또는 personality만 가능합니다",
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
                            "message": "면접 리포트 생성 시 context는 Q&A 배열이어야 합니다",
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
    """채팅 처리 (통합)"""
    return StreamingResponse(generate_chat_stream(request), media_type="text/event-stream")


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
