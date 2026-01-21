import asyncio
import json
import logging
import os
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Header, HTTPException, status
from fastapi.responses import StreamingResponse

from app.schemas.calendar import CalendarParseRequest, CalendarParseResponse
from app.schemas.chat import (
    AnalysisResult,
    ChatMode,
    ChatRequest,
    InterviewReport,
    MatchingResult,
    PostingAnalysis,
    QAEvaluation,
    ResumeAnalysis,
)
from app.schemas.common import (
    AsyncTaskResponse,
    ErrorCode,
    TaskStatus,
    TaskStatusResponse,
)
from app.schemas.text_extract import PageText, TextExtractRequest, TextExtractResult
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService
from app.services.vectordb_service import VectorDBService
from app.services.vllm_service import VLLMService

logger = logging.getLogger(__name__)


def sanitize_for_log(value: str, max_length: int = 100) -> str:
    """
    Sanitize user input for safe logging to prevent log injection attacks.

    Args:
        value: The string value to sanitize
        max_length: Maximum length to truncate to

    Returns:
        Sanitized string safe for logging
    """
    if not value:
        return ""

    # Remove control characters and newlines that could be used for log injection
    sanitized = "".join(
        char if char.isprintable() and char not in "\n\r" else " " for char in value
    )

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length] + "..."

    return sanitized


router = APIRouter(
    prefix="/ai",
    tags=["AI APIs (v3.0)"],
    responses={404: {"description": "Not found"}},
)

# 임시 작업 저장소 (실제로는 Redis 등 사용)
tasks_db = {}

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
    summary="텍스트 추출 + 임베딩 저장 (통합)",
    description="""
    파일 또는 텍스트에서 텍스트를 추출하고 내부에서 임베딩까지 처리합니다.

    **v3.0 변경사항:**
    - `/ai/ocr/extract` + `/ai/file/embed` 통합
    - VectorDB 정보 제거 (AI 서버 내부에서만 사용)
    - 백엔드에는 extracted_text만 반환

    **처리 방식:** 비동기 (task_id 반환 → 폴링 필요)

    **내부 처리 흐름:**
    1. 파일 입력 시: OCR/VLM으로 텍스트 추출
    2. 텍스트 입력 시: 그대로 사용
    3. 텍스트 청킹 (500 tokens, 50 overlap)
    4. Gemini Embedding 생성
    5. VectorDB에 저장 (type에 따른 컬렉션)
    6. Backend에 extracted_text만 반환
    """,
    dependencies=[Depends(verify_api_key)],
)
async def text_extract(request: TextExtractRequest):
    """텍스트 추출 + 임베딩 저장 (통합)"""
    task_id = f"task_{uuid.uuid4().hex[:12]}"

    # 비동기 작업 시작
    tasks_db[task_id] = {
        "status": TaskStatus.PROCESSING,
        "created_at": datetime.now(),
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
            logger.info("=== 📄 텍스트 추출 시작 (파일 업로드) ===")
            logger.info(f"{'='*80}")
            logger.info(f"📌 요청 모델: {sanitize_for_log(model, 20).upper()}")
            logger.info(f"📌 문서 타입: {sanitize_for_log(request.type, 50)}")
            logger.info(f"📌 사용자 ID: {sanitize_for_log(request.user_id, 50)}")
            logger.info(f"📌 문서 ID: {sanitize_for_log(str(request.document_id), 50)}")
            logger.info(f"📌 vLLM 서비스: {'✅ 사용 가능' if rag.vllm else '❌ 사용 불가'}")

            # 파일 URL이 있으면 OCR 처리
            if request.file_url:
                file_type = request.file_type if hasattr(request, "file_type") else "pdf"
                logger.info(f"📌 파일 타입: {file_type}")
                logger.info("")

                # vLLM 모드: pytesseract OCR 사용 (가성비)
                if model == "vllm" and rag.vllm:
                    logger.info("💰 [vLLM 가성비 모드] pytesseract OCR 시작")
                    logger.info("   → 비용 절감을 위해 pytesseract 사용 (Gemini Vision API 대신)")
                    ocr_result = await rag.vllm.extract_text_from_file(
                        file_url=str(request.file_url), file_type=file_type
                    )
                    extracted_text = ocr_result["extracted_text"]
                    pages = [PageText(**page) for page in ocr_result["pages"]]
                    logger.info(
                        f"✅ [vLLM OCR] 추출 완료: {len(extracted_text)}자 (페이지: {len(pages)})"
                    )

                # Gemini 모드: Gemini Vision API 사용 (고성능)
                else:
                    if model == "vllm" and not rag.vllm:
                        logger.warning("⚠️ vLLM 서비스 사용 불가 → Gemini로 자동 변경")
                    logger.info("🚀 [Gemini 고성능 모드] Gemini Vision API OCR 시작")
                    logger.info("   → 고품질 OCR을 위해 Gemini Vision API 사용")
                    ocr_result = await rag.llm.extract_text_from_file(
                        file_url=str(request.file_url), file_type=file_type
                    )
                    extracted_text = ocr_result["extracted_text"]
                    pages = [PageText(**page) for page in ocr_result["pages"]]
                    logger.info(
                        f"✅ [Gemini OCR] 추출 완료: {len(extracted_text)}자 (페이지: {len(pages)})"
                    )

            # 텍스트 직접 입력
            else:
                extracted_text = request.text or ""
                pages = None
                logger.info(f"텍스트 직접 입력: {len(extracted_text)} characters")

            # VectorDB에 임베딩 저장
            if extracted_text:
                document_id = request.document_id or f"doc_{uuid.uuid4().hex[:12]}"

                await rag.vectordb.add_document(
                    document_id=document_id,
                    text=extracted_text,
                    collection_type=request.type,
                    metadata={"user_id": request.user_id, "created_at": datetime.now().isoformat()},
                )

                tasks_db[task_id]["status"] = TaskStatus.COMPLETED
                tasks_db[task_id]["result"] = TextExtractResult(
                    success=True, extracted_text=extracted_text, pages=pages
                ).model_dump()
            else:
                raise ValueError("No text extracted")

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
    dependencies=[Depends(verify_api_key)],
)
async def get_task_status(task_id: str):
    """비동기 작업 상태 조회"""
    if task_id not in tasks_db:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": ErrorCode.TASK_NOT_FOUND, "message": "작업을 찾을 수 없습니다."},
        )

    task = tasks_db[task_id]

    return TaskStatusResponse(
        task_id=task_id,
        status=task["status"],
        progress=task.get("progress"),
        message=task.get("message"),
        result=task.get("result"),
        error=task.get("error"),
    )


# ============================================================================
# API 2: 채팅 (통합: 대화/분석/면접) (스트리밍)
# ============================================================================


async def generate_chat_stream(request: ChatRequest):
    """채팅 응답 스트리밍 생성"""

    mode = request.context.mode
    rag = get_services()
    newline = "\n"
    sse_end = "\n\n"

    # 모델 선택 (gemini 또는 vllm)
    model = request.model.value if hasattr(request.model, "value") else str(request.model)
    logger.info("")
    logger.info(f"{'='*80}")
    logger.info("=== 💬 채팅 요청 시작 ===")
    logger.info(f"{'='*80}")
    logger.info(f"📌 요청 모델: {sanitize_for_log(model, 20).upper()}")
    logger.info(f"📌 채팅 모드: {sanitize_for_log(mode, 50)}")
    logger.info(f"📌 사용자 ID: {sanitize_for_log(request.user_id, 50)}")
    logger.info(f"📌 채팅방 ID: {sanitize_for_log(str(request.room_id), 50)}")
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

            if is_analysis:
                # ===================================================================
                # 분석 요청: vLLM과 Gemini 완전 분리
                # ===================================================================
                logger.info(f"🔍 분석 요청 감지: '{sanitize_for_log(user_message, 50)}'")
                logger.info("")

                # ---------------------------------------------------------------
                # vLLM 모드 (가성비): OCR(pytesseract) → VectorDB → Llama 분석
                # ---------------------------------------------------------------
                if model == "vllm" and rag.vllm:
                    logger.info("💰 [vLLM 가성비 모드] 분석 시작")
                    logger.info(
                        "   프로세스: pytesseract OCR → VectorDB 저장 → VectorDB 조회 → Llama 분석"
                    )
                    logger.info("")

                    # 1. VectorDB에서 OCR로 추출된 모든 문서 가져오기
                    logger.info("📂 [1/3] VectorDB에서 업로드된 문서 조회 중...")
                    full_context = await rag.retrieve_all_documents(
                        user_id=request.user_id, context_types=["resume", "job_posting"]
                    )

                    if not full_context:
                        error_msg = "❌ 업로드된 이력서 또는 채용공고를 찾을 수 없습니다.\n먼저 파일을 업로드해주세요."
                        logger.error(
                            f"⚠️ VectorDB에 문서가 없습니다 (user_id: {sanitize_for_log(request.user_id, 50)})"
                        )
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
                logger.info(
                    f"🔍 [{sanitize_for_log(model, 20).upper()}] RAG 검색 및 응답 생성 시작..."
                )
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
                    f"✅ [{sanitize_for_log(model, 20).upper()}] 일반 대화 완료 (응답 길이: {len(full_response)}자)"
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

            # 면접 질문 생성 프롬프트 (간소화)
            if context:
                question_prompt = f"{interview_type_kr} 면접 질문 1개를 생성해주세요.\n\n참고정보:\n{context}\n\n질문만 간단히 출력하세요:"
            else:
                question_prompt = f"일반적인 {interview_type_kr} 면접 질문 1개를 짧게 생성해주세요:"

            full_question = ""
            async for chunk in rag.llm.generate_response(
                user_message=question_prompt,
                context=None,
                history=[],
                system_prompt="면접관입니다. 간단명료하게 질문만 생성하세요.",
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
        chunks = [
            f"면접 답변을 분석 중입니다...{newline}",
            f"종합 리포트를 작성 중입니다...{newline}",
        ]

        for chunk in chunks:
            yield f"data: {json.dumps({'type': 'chunk', 'content': chunk}, ensure_ascii=False)}{sse_end}"
            await asyncio.sleep(0.5)

        # Pydantic AI 모델로 구조화된 리포트
        report = InterviewReport(
            evaluations=[
                QAEvaluation(
                    question="React의 Virtual DOM이 무엇인가요?",
                    answer="실제 DOM과 비교해서 변경된 부분만 업데이트하는 거예요",
                    good_points=["Virtual DOM의 기본 개념을 잘 이해하고 있음"],
                    improvements=["Reconciliation 알고리즘 설명 추가하면 좋음"],
                )
            ],
            strength_patterns=["기술 개념에 대한 이해도가 높음"],
            weakness_patterns=["심화 개념 설명이 부족함"],
            learning_guide=["React 심화 개념 학습 (Fiber, Concurrent Mode)"],
        )

        result = {"success": True, "mode": "interview_report", "report": report.model_dump()}
        yield f"data: {json.dumps({'type': 'complete', 'data': result}, ensure_ascii=False)}{sse_end}"


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
    dependencies=[Depends(verify_api_key)],
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
    dependencies=[Depends(verify_api_key)],
)
async def calendar_parse(request: CalendarParseRequest):
    """캘린더 일정 파싱"""
    if not request.file_url and not request.text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "code": ErrorCode.INVALID_REQUEST,
                "message": "file_url 또는 text 중 하나는 필수입니다.",
            },
        )

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
