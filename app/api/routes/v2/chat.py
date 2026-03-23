"""
v2 채팅 API (통합: 대화/분석/면접/리포트)

POST /ai/chat - 채팅 스트리밍 (SSE)
"""

import asyncio
import json
import logging
import random
import time
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.routes.v2._helpers import (
    extract_json_from_llm_response,
    get_services,
    get_session_key,
    stream_text_chars,
)
from app.api.routes.v2._sse_errors import sse_error_event
from app.config.dependencies import get_session_store
from app.config.settings import get_settings
from app.prompts import (
    get_extract_title_prompt,
)
from app.prompts.interview import (
    create_feedback_prompt,
    create_tech_followup_prompt,
    format_conversation_history,
    format_followup_question_label,
    format_main_question_label,
    get_system_tech_interview,
)
from app.prompts.loader import load_prompt_yaml
from app.prompts.persona_styles import get_personality_style_prompt, get_tech_style_prompt
from app.schemas.chat import (
    ChatMode,
    ChatRequest,
    InterviewQuestionState,
    InterviewSession,
)
from app.services.example_selector import get_few_shot_for_personality, get_few_shot_for_technical
from app.services.interview_dedup import cosine_similarity, is_mastered_quality
from app.services.web_loader_service import WebLoaderService
from app.utils.log_sanitizer import safe_info, safe_warning, sanitize_log_input
from app.utils.prompt_guard import RiskLevel, check_prompt_injection

logger = logging.getLogger(__name__)

# ── 면접 파라미터 상수 ───────────────────────────────────────
# 참고: total_questions(5), max_depth(3) 기본값은 schemas/chat.py Field default가 관리
PERSONALITY_VECTORDB_SELECT = 4  # 인성 면접 VectorDB에서 선택할 질문 수

# 인성 면접 질문 선택 관련 상수
PERSONALITY_SIMILARITY_THRESHOLD = 0.80  # 질문 간 유사도 임계값
PERSONALITY_QUERY_COUNT = 6  # 검색할 카테고리 수
PERSONALITY_RESULTS_PER_QUERY = 15  # 카테고리당 검색 결과 수

router = APIRouter()


async def generate_chat_stream(
    request: ChatRequest,
    session_store,
    http_request: Request,
):
    """채팅 응답 스트리밍 생성 (session_store: DI from get_session_store)"""

    # =========================================================================
    # 프롬프트 인젝션 검사
    # =========================================================================
    user_message_raw = request.message or ""
    guard_result = check_prompt_injection(user_message_raw)

    if guard_result.risk_level == RiskLevel.BLOCK:
        safe_warning(
            logger,
            "🚨 프롬프트 인젝션 차단: user_id=%s, patterns=%s",
            request.user_id,
            str(guard_result.matched_patterns),
        )
        try:
            from app.core.monitoring import AI_PROMPT_INJECTION_BLOCKED

            AI_PROMPT_INJECTION_BLOCKED.labels(endpoint="/ai/chat").inc()
        except Exception as e:
            logger.error(f"Prompt Injection Metric Error: {e}")

        yield sse_error_event(
            code="PROMPT_BLOCKED",
            status=400,
            message="프롬프트 인젝션이 감지되어 차단되었습니다.",
            fallback=guard_result.message,
        )
        yield "data: [DONE]\n\n"
        return

    if guard_result.risk_level == RiskLevel.WARNING:
        safe_warning(
            logger,
            "⚠️ 의심스러운 입력 감지: user_id=%s, patterns=%s",
            request.user_id,
            str(guard_result.matched_patterns),
        )

    mode = request.context.mode if request.context else ChatMode.NORMAL

    rag = get_services()
    if rag is None:
        logger.error("서비스 초기화 실패 (VectorDB 연결 불가) — 요청 거부")
        yield 'data: {"error": "서비스 초기화 실패. 잠시 후 다시 시도해주세요."}\n\n'
        return

    newline = "\n"
    sse_end = "\n\n"

    # 모니터링 시작
    start_time = time.time()
    first_token_time = None

    def record_ttft():
        nonlocal first_token_time
        if first_token_time is None:
            first_token_time = time.time()
            try:
                ttft = first_token_time - start_time
                from app.core.monitoring import AI_TIME_TO_FIRST_TOKEN

                AI_TIME_TO_FIRST_TOKEN.labels(model=model, endpoint="/ai/chat").observe(ttft)
            except Exception as e:
                logger.error(f"TTFT Metric Error: {e}")

    model = request.model.value if hasattr(request.model, "value") else str(request.model)

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

    # =========================================================================
    # 1. 일반 대화 (RAG 활용)
    # =========================================================================
    if mode == ChatMode.NORMAL:
        full_response = ""

        try:
            history_dict = []
            user_message = request.message or ""

            # URL 감지 및 웹 컨텍스트 추출 (ADR-059)
            user_message, web_context = await WebLoaderService.extract_chat_context(user_message)
            if web_context:
                logger.info("🌐 URL 웹 컨텍스트 추출 완료: %d자", len(web_context))
                user_message = f"참고 URL 내용:\n{web_context}\n\n사용자 질문: {user_message}"

            is_analysis = any(
                keyword in user_message for keyword in ["분석", "매칭", "적합", "평가", "비교"]
            )
            is_followup = (
                request.interview_id is not None and request.context.mode == ChatMode.INTERVIEW
            )

            if is_analysis:
                logger.info("🔍 분석 요청 감지")
                logger.info("")

                # 채팅방 제목 추출
                chat_title = ""
                try:
                    logger.info("📝 [0/3] 채팅방 제목 추출 중...")
                    job_posting_docs = await rag.retrieve_all_documents(
                        user_id=request.user_id, context_types=["job_posting"]
                    )

                    if job_posting_docs:
                        posting_text = job_posting_docs[:1000]
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
                            record_ttft()
                            title_response += chunk

                        chat_title = title_response.strip()
                        logger.info(f"✅ [0/3] 채팅방 제목: {chat_title}")
                        yield f"data: {json.dumps({'summary': chat_title}, ensure_ascii=False)}{sse_end}"
                    else:
                        logger.warning("⚠️ 채용공고를 찾을 수 없어 제목 추출 생략")
                except Exception as e:
                    logger.error(f"❌ 채팅방 제목 추출 실패: {e}")
                logger.info("")

                # vLLM 모드
                if model == "vllm" and rag.vllm:
                    logger.info("💰 [vLLM 가성비 모드] 분석 시작")
                    logger.info("   프로세스: EasyOCR → VectorDB 저장 → VectorDB 조회 → Llama 분석")
                    logger.info("")

                    logger.info("📂 [1/3] VectorDB에서 업로드된 문서 조회 중...")
                    full_context = await rag.retrieve_all_documents(
                        user_id=request.user_id, context_types=["resume", "job_posting"]
                    )

                    if not full_context:
                        logger.error("⚠️ VectorDB에 문서가 없습니다")
                        yield sse_error_event(
                            code="VECTORDB_ERROR",
                            status=404,
                            message="VectorDB에 업로드된 문서가 없습니다.",
                            fallback="업로드된 이력서 또는 채용공고를 찾을 수 없습니다. 먼저 파일을 업로드해주세요.",
                        )
                        full_response = ""
                    else:
                        logger.info(f"✅ [1/3] VectorDB 조회 완료: {len(full_context)}자")
                        logger.info("")

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
                            if await http_request.is_disconnected():
                                logger.info("[Chat][NORMAL] 클라이언트 연결 해제 — 스트림 종료")
                                return
                            record_ttft()
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

                        logger.info(f"✅ [3/3] Llama 분석 완료 (응답 길이: {len(full_response)}자)")

                # Gemini 모드
                else:
                    if model == "vllm" and not rag.vllm:
                        logger.warning("⚠️ vLLM 서비스 사용 불가 → Gemini로 자동 변경")

                    logger.info("🚀 [Gemini 고성능 모드] 분석 시작")
                    logger.info("   프로세스: RAG 검색 → Gemini 분석 (원래 방식)")
                    logger.info("")

                    logger.info("📂 [1/2] RAG 검색 중...")
                    async for chunk in rag.chat_with_rag(
                        user_message=user_message,
                        user_id=request.user_id,
                        history=history_dict,
                        use_rag=True,
                        context_types=["resume", "job_posting"],
                        model="gemini",
                        chat_mode=mode.value if hasattr(mode, "value") else str(mode),  # ADR-077
                    ):
                        if await http_request.is_disconnected():
                            logger.info("[Chat][NORMAL] 클라이언트 연결 해제 — 스트림 종료")
                            return
                        record_ttft()
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

                    logger.info(f"✅ [2/2] Gemini 분석 완료 (응답 길이: {len(full_response)}자)")
            else:
                # 일반 대화
                logger.info("💬 일반 대화 모드")
                logger.info("")

                if is_followup:
                    interview_qa_count = len([h for h in history_dict if h.get("role") == "user"])
                    logger.info(f"📊 현재 면접 답변 수: {interview_qa_count}개")

                    if interview_qa_count >= 5:
                        logger.info("🎯 [면접 종료] 5개 답변 완료 → 피드백 생성 시작")

                        end_msg = "면접이 종료되었습니다. 답변 평가를 시작합니다.\n\n"
                        yield f"data: {json.dumps({'chunk': end_msg}, ensure_ascii=False)}{sse_end}"
                        full_response += end_msg

                        qa_pairs = []
                        for i in range(0, len(history_dict), 2):
                            if i + 1 < len(history_dict):
                                qa_pairs.append(
                                    {
                                        "question": history_dict[i].get("content", ""),
                                        "answer": history_dict[i + 1].get("content", ""),
                                    }
                                )

                        evaluation_content = (
                            "다음 면접 Q&A에 대해 각 답변마다 피드백을 제공해주세요:\n\n"
                        )
                        for i, qa in enumerate(qa_pairs[:5], 1):
                            evaluation_content += (
                                f"질문 {i}: {qa['question']}\n답변 {i}: {qa['answer']}\n\n"
                            )
                        feedback_prompt = create_feedback_prompt(evaluation_content)

                        async for chunk in rag.llm.generate_response(
                            user_message=feedback_prompt,
                            context=None,
                            history=[],
                            system_prompt="당신은 전문 면접관입니다. 지원자의 답변을 평가하고 구체적인 피드백을 제공합니다. 마크다운 문법을 사용하지 마세요.",
                            user_id=request.user_id,
                        ):
                            if await http_request.is_disconnected():
                                logger.info("[Chat][NORMAL] 클라이언트 연결 해제 — 스트림 종료")
                                return
                            record_ttft()
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

                        logger.info(
                            f"✅ [면접 피드백] 생성 완료 (응답 길이: {len(full_response)}자)"
                        )

                    else:
                        original_question = history_dict[-2].get("content", "")
                        candidate_answer = history_dict[-1].get("content", "")

                        logger.info("🔍 [꼬리질문 생성] 감지")
                        logger.info(f"   원본 질문: {original_question[:50]}...")
                        logger.info(f"   답변: {candidate_answer[:50]}...")
                        logger.info("")

                        star_analysis = {
                            "situation": "unknown",
                            "task": "unknown",
                            "action": "unknown",
                            "result": "unknown",
                        }

                        async for chunk in rag.generate_followup_question(
                            original_question=original_question,
                            candidate_answer=candidate_answer,
                            star_analysis=star_analysis,
                            model=model,
                            user_id=request.user_id,
                        ):
                            if await http_request.is_disconnected():
                                logger.info("[Chat][NORMAL] 클라이언트 연결 해제 — 스트림 종료")
                                return
                            record_ttft()
                            full_response += chunk
                            yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

                        logger.info(f"✅ [꼬리질문 생성] 완료 (응답 길이: {len(full_response)}자)")

                else:
                    if (
                        "면접 질문" in user_message
                        or "면접질문" in user_message
                        or "면접" in user_message
                    ):
                        context_types = ["portfolio"]
                        logger.info("🎯 면접 질문 요청 감지 → portfolio 컬렉션만 검색")
                    else:
                        context_types = ["resume", "job_posting", "portfolio"]
                        logger.info("📚 일반 대화 → 모든 컬렉션 검색")

                    logger.info("")

                    logger.info("🔍 RAG 검색 및 응답 생성 시작...")
                    async for chunk in rag.chat_with_rag(
                        user_message=user_message,
                        user_id=request.user_id,
                        history=history_dict,
                        use_rag=True,
                        context_types=context_types,
                        model=model,
                        chat_mode=mode.value if hasattr(mode, "value") else str(mode),  # ADR-077
                    ):
                        if await http_request.is_disconnected():
                            logger.info("[Chat][NORMAL] 클라이언트 연결 해제 — 스트림 종료")
                            return
                        record_ttft()
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

                    logger.info("✅ 일반 대화 완료 (응답 길이: %d자)", len(full_response))

        except asyncio.CancelledError:
            logger.info("[Chat][NORMAL] 스트림 취소됨 (클라이언트 연결 해제)")
            return
        except Exception as e:
            logger.error("채팅 처리 오류: %s", str(e), exc_info=True)
            yield sse_error_event(
                code="INTERNAL_ERROR",
                status=500,
                message=str(e),
                fallback="일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            )
            full_response = ""

        yield f"data: [DONE]{sse_end}"

    # =========================================================================
    # 2. 면접 모드 - 5개 질문 × 최대 3 depths 꼬리질문
    # =========================================================================
    elif mode == ChatMode.INTERVIEW:
        try:
            # SSE 스트림 즉시 오픈: Redis/RAG 등 첫 await 전에 keepalive를 전송해
            # 프록시(Nginx) 및 메인 백엔드의 첫 청크 대기 타임아웃을 방지한다.
            if await http_request.is_disconnected():
                logger.info("[Chat][INTERVIEW] 클라이언트 연결 해제 — 스트림 종료")
                return
            yield ": keepalive\n\n"

            interview_type = request.context.interview_type or "tech"
            interview_type_kr = "기술" if interview_type == "tech" else "인성"

            session_key = get_session_key(request.user_id, request.interview_id)
            user_message = request.message or ""

            session = request.context.interview_session
            safe_info(
                logger,
                "🔍 [면접 진단] session_key=%s | request.interview_session=%s | user_msg_len=%s",
                session_key,
                f"phase={session.phase}" if session else "None",
                len(user_message),
            )
            if session is None:
                logger.debug(
                    "🔍 [면접] session_store.get() 호출 전: key=%s",
                    sanitize_log_input(session_key),
                )
                session_data = await session_store.get(session_key)
                logger.debug(
                    "🔍 [면접] session_store.get() 완료: data=%s",
                    session_data is not None,
                )
                session = InterviewSession.model_validate(session_data) if session_data else None
                if session:
                    safe_info(
                        logger,
                        "📦 [면접] 캐시에서 세션 복원: %s, phase=%s, Q%s/5",
                        session_key,
                        session.phase,
                        session.current_question_id,
                    )
            safe_info(
                logger,
                "🔍 [면접 진단] 최종 세션 상태: %s",
                f"phase={session.phase}, Q{session.current_question_id}/5"
                if session
                else "None → PHASE 1 시작",
            )

            model_choice = (
                request.model.value if hasattr(request.model, "value") else str(request.model)
            )

            # PHASE 1: 세션 초기화
            if session is None or session.phase == "init":
                logger.info("🎯 [면접] 세션 초기화 - 5개 질문 세트 생성 시작")

                context = await rag.retrieve_context(
                    query=f"{interview_type_kr} 면접 질문을 위한 사용자 정보",
                    user_id=request.user_id,
                    context_types=["resume", "portfolio", "job_posting"],
                    n_results=3,
                )

                resume_ocr = request.context.resume_ocr if request.context else None
                job_posting_ocr = request.context.job_posting_ocr if request.context else None
                portfolio_text = request.context.portfolio_text if request.context else None

                yaml_name = (
                    "personality_interview_init"
                    if interview_type == "behavior"
                    else "tech_interview_init"
                )
                init_prompts = load_prompt_yaml("interview", yaml_name)

                # ADR-098: 면접 스타일 프롬프트 적용
                # 허용된 스타일만 사용 (Log Injection 방지)
                allowed_styles = {"friendly", "standard", "challenging", "practical"}
                raw_style = (
                    request.context.interview_style
                    if request.context and request.context.interview_style
                    else "standard"
                )
                interview_style = raw_style if raw_style in allowed_styles else "standard"
                if interview_type == "behavior":
                    style_prompt = get_personality_style_prompt(interview_style)
                else:
                    style_prompt = get_tech_style_prompt(interview_style)

                safe_info(logger, "🎭 [면접] 스타일 적용: %s", interview_style)

                system_prompt = init_prompts["system"].format(style_prompt=style_prompt)
                init_prompt = init_prompts["human"].format(
                    resume_text=resume_ocr or context or "정보 없음",
                    job_posting_text=job_posting_ocr or "정보 없음",
                    portfolio_text=portfolio_text or context or "정보 없음",
                )

                # 인성 면접: SemanticSimilarityExampleSelector — interview_feedback에서 유사 Q&A 참고 예시 추가
                if yaml_name == "personality_interview_init":
                    try:
                        few_shot = await get_few_shot_for_personality(
                            rag.vectordb,
                            query_text=f"{interview_type_kr} 면접 질문 답변 예시",
                            k=2,
                            interview_type_filter="personality",
                        )
                        if few_shot:
                            system_prompt = (
                                system_prompt.rstrip()
                                + "\n\n## 참고 예시 (유사 Q&A)\n\n"
                                + few_shot
                                + "\n\n위 예시 톤을 참고하여 질문을 생성하세요."
                            )
                    except Exception as e:
                        logger.debug("Few-shot personality selection skipped: %s", e)

                # 기술 면접: interview_feedback에서 유사 기술 Q&A 참고 예시 추가
                elif yaml_name == "tech_interview_init":
                    try:
                        few_shot = await get_few_shot_for_technical(
                            rag.vectordb,
                            query_text="기술 면접 질문 답변 예시",
                            k=2,
                        )
                        if few_shot:
                            system_prompt = (
                                system_prompt.rstrip()
                                + "\n\n## 참고 예시 (유사 기술 Q&A)\n\n"
                                + few_shot
                                + "\n\n위 예시의 깊이와 구체성을 참고하여 질문을 생성하세요."
                            )
                    except Exception as e:
                        logger.debug("Few-shot tech selection skipped: %s", e)

                # 인성 면접: interview_feedback에서 미리 질문 로드 (LLM 스킵)
                behavior_questions_data: dict | None = None
                if interview_type == "behavior":
                    _fixed_q1 = {
                        "id": 1,
                        "category": "intro_self",
                        "category_name": "자기소개",
                        "question": "자기소개 해보세요.",
                        "intent": "지원자의 전반적인 역량과 경력 요약 파악",
                        "keywords": ["자기소개", "경력", "역량"],
                    }
                    try:
                        # 다양한 카테고리의 쿼리로 검색하여 질문 다양성 확보
                        _personality_queries = [
                            "팀워크 협업 경험",
                            "갈등 해결 문제 상황",
                            "리더십 경험 사례",
                            "실패 극복 경험",
                            "스트레스 관리 방법",
                            "장단점 자기 분석",
                            "목표 달성 성취 경험",
                            "의사소통 커뮤니케이션",
                            "동기부여 열정",
                            "적응력 변화 대응",
                        ]
                        _selected_queries = random.sample(
                            _personality_queries,
                            min(PERSONALITY_QUERY_COUNT, len(_personality_queries)),
                        )

                        # 병렬 VectorDB 쿼리로 성능 최적화
                        async def _query_personality(query_text: str) -> list[dict]:
                            return await rag.vectordb.query(
                                query_text=query_text,
                                collection_type="interview_feedback",
                                n_results=PERSONALITY_RESULTS_PER_QUERY,
                                where={"interview_type": "personality"},
                                max_distance=1.8,
                            )

                        _query_results = await asyncio.gather(
                            *[_query_personality(q) for q in _selected_queries]
                        )

                        _all_candidates = []
                        _seen_questions: set[str] = set()

                        for _fb in _query_results:
                            for r in _fb:
                                q_text = (r.get("metadata") or {}).get("question_only", "")
                                if q_text and q_text not in _seen_questions:
                                    _seen_questions.add(q_text)
                                    _all_candidates.append(r)

                        logger.info(
                            "📋 interview_feedback 인성 질문 후보 %d개 조회 (카테고리: %s)",
                            len(_all_candidates),
                            _selected_queries,
                        )

                        # 유사도 기반 중복 제거하며 PERSONALITY_VECTORDB_SELECT개 선택
                        if len(_all_candidates) >= PERSONALITY_VECTORDB_SELECT:
                            random.shuffle(_all_candidates)
                            _selected: list[dict] = []
                            _selected_embeddings: list[list[float]] = []

                            for candidate in _all_candidates:
                                if len(_selected) >= PERSONALITY_VECTORDB_SELECT:
                                    break

                                q_text = candidate["metadata"]["question_only"]
                                q_embedding = await rag.vectordb.create_embedding(q_text)

                                # 이미 선택된 질문들과 유사도 체크
                                is_similar = False
                                for existing_emb in _selected_embeddings:
                                    sim = cosine_similarity(q_embedding, existing_emb)
                                    if sim >= PERSONALITY_SIMILARITY_THRESHOLD:
                                        logger.debug(
                                            "🔄 유사 질문 스킵 (sim=%.2f): %s", sim, q_text[:30]
                                        )
                                        is_similar = True
                                        break

                                if not is_similar:
                                    _selected.append(candidate)
                                    _selected_embeddings.append(q_embedding)

                            if len(_selected) >= PERSONALITY_VECTORDB_SELECT:
                                _fb_qs = [
                                    {
                                        "id": i + 2,
                                        "category": f"personality_q{i + 2}",
                                        "category_name": "인성",
                                        "question": r["metadata"]["question_only"],
                                        "intent": "",
                                        "keywords": [],
                                    }
                                    for i, r in enumerate(_selected[:PERSONALITY_VECTORDB_SELECT])
                                ]
                                logger.info(
                                    "✅ interview_feedback 인성 질문 4개 선택 (유사도 필터 적용)"
                                )
                                behavior_questions_data = {"questions": [_fixed_q1] + _fb_qs}
                            else:
                                logger.warning(
                                    "⚠️ 유사도 필터 후 질문 부족 (%d개) → LLM 폴백",
                                    len(_selected),
                                )
                    except Exception as _e:
                        logger.warning("interview_feedback 선 조회 실패 → LLM 폴백: %s", _e)

                full_response = ""  # 기본값 (except block에서 full_response[:500] 참조)
                if behavior_questions_data is None:
                    if model_choice == "vllm" and rag.vllm:
                        async for chunk in rag.vllm.generate_response(
                            user_message=init_prompt,
                            context=None,
                            history=[],
                            system_prompt=system_prompt,
                        ):
                            record_ttft()
                            full_response += chunk
                    else:
                        # Gunicorn + Nginx 환경에서 PHASE 1 LLM 대기(60~120초) 중
                        # SSE idle timeout(proxy_read_timeout 기본 60s) 방지:
                        # 1) 즉시 "생성 중" 이벤트 전송 → Nginx idle timer 리셋
                        # 2) 25초마다 keepalive SSE comment 전송
                        thinking_event = json.dumps(
                            {"type": "thinking", "chunk": "⏳ 면접 질문을 생성하고 있습니다..."},
                            ensure_ascii=False,
                        )
                        yield f"data: {thinking_event}{sse_end}"

                        safe_info(
                            logger,
                            "⏳ [PHASE 1] LLM 비스트리밍 호출 시작 | user=%s room=%s",
                            request.user_id,
                            request.room_id,
                        )
                        llm_task = asyncio.create_task(
                            rag.llm.generate_response_non_stream(
                                user_message=init_prompt,
                                context=None,
                                system_prompt=system_prompt,
                                user_id=request.user_id,
                                max_tokens=get_settings().llm_max_tokens_interview,
                            )
                        )
                        try:
                            while not llm_task.done():
                                try:
                                    await asyncio.wait_for(asyncio.shield(llm_task), timeout=25)
                                    break
                                except asyncio.TimeoutError:
                                    logger.debug(
                                        "⏳ [PHASE 1] keepalive 전송 (LLM 응답 대기 중...)"
                                    )
                                    yield ": keepalive\n\n"  # SSE comment → Nginx idle timer 리셋
                            full_response = llm_task.result()
                            safe_info(
                                logger,
                                "✅ [PHASE 1] LLM 응답 수신 완료 (%s자) | user=%s room=%s",
                                len(full_response),
                                request.user_id,
                                request.room_id,
                            )
                        finally:
                            if not llm_task.done():
                                llm_task.cancel()

                # JSON 파싱 또는 vectordb 직접 구성
                try:
                    if behavior_questions_data is not None:
                        # 인성 면접 vectordb 경로: LLM 없이 직접 세션 구성
                        questions_data = behavior_questions_data
                    else:
                        # LLM 응답 파싱 (기존 경로)
                        questions_data = extract_json_from_llm_response(full_response)
                        if questions_data is None:
                            raise ValueError("JSON 형식을 찾을 수 없습니다")
                        logger.info("JSON 파싱 시도 (첫 200자): %s", str(questions_data)[:200])

                        # 인성 면접 LLM 폴백: Q1+Q2 하드코딩 + Q3-Q5 LLM
                        if interview_type == "behavior":
                            fixed_q1 = {
                                "id": 1,
                                "category": "intro_self",
                                "category_name": "자기소개",
                                "question": "자기소개 해보세요.",
                                "intent": "지원자의 전반적인 역량과 경력 요약 파악",
                                "keywords": ["자기소개", "경력", "역량"],
                            }
                            fixed_q2 = {
                                "id": 2,
                                "category": "intro_motivation",
                                "category_name": "지원동기",
                                "question": "우리 회사를 지원하는 이유가 뭔가요?",
                                "intent": "지원 동기와 회사/직무 이해도 확인",
                                "keywords": ["지원동기", "회사이해", "직무"],
                            }
                            llm_questions = [
                                q
                                for q in questions_data.get("questions", [])
                                if q.get("id", 0) >= 3
                            ]
                            questions_data["questions"] = [fixed_q1, fixed_q2] + llm_questions
                            logger.info("인성 면접 LLM 폴백 적용 (interview_feedback 결과 부족)")

                    # 공통: 세션 생성 + 스트리밍 + 저장
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

                    # 질문 스트리밍을 먼저 수행 (Redis 저장 실패해도 사용자에게 질문 전달)
                    first_q = new_session.questions[0] if new_session.questions else None
                    if first_q:
                        question_text = (
                            f"{format_main_question_label(1)}{newline}{first_q.question}"
                        )
                        async for chunk in stream_text_chars(question_text, sse_end):
                            if await http_request.is_disconnected():
                                logger.info("[Chat][INTERVIEW] 클라이언트 연결 해제 — 스트림 종료")
                                return
                            yield chunk

                        session_meta = {
                            "type": "session_state",
                            "session": new_session.model_dump(),
                        }
                        yield f"data: {json.dumps(session_meta, ensure_ascii=False)}{sse_end}"

                    # 세션 저장 (실패해도 질문은 이미 전달됨)
                    try:
                        await session_store.set(session_key, new_session.model_dump())
                        safe_info(logger, "💾 [면접] 세션 저장: %s", session_key)
                    except Exception as e:
                        logger.error(
                            "💾 [면접] 세션 저장 실패 (질문은 전달됨): %s",
                            type(e).__name__,
                        )

                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"질문 세트 파싱 실패: {e}")
                    logger.error(f"원본 응답 (첫 500자): {full_response[:500]}")

                    yield sse_error_event(
                        code="PARSE_FAILED",
                        status=500,
                        message=f"질문 세트 JSON 파싱 실패: {e}",
                        fallback="면접 질문 세트 생성 중 오류가 발생했습니다. 다시 시도해주세요.",
                    )

                yield f"data: [DONE]{sse_end}"

            # PHASE 2: 꼬리질문 또는 다음 질문 생성
            elif session.phase in ["questioning", "followup"]:
                current_q_id = session.current_question_id
                current_q = next((q for q in session.questions if q.id == current_q_id), None)

                if not current_q:
                    yield sse_error_event(
                        code="SESSION_NOT_FOUND",
                        status=404,
                        message=f"면접 세션에서 질문 ID {current_q_id}를 찾을 수 없습니다.",
                        fallback="세션 오류: 현재 질문을 찾을 수 없습니다.",
                    )
                    yield f"data: [DONE]{sse_end}"
                    return

                current_q.conversation.append(
                    {
                        "role": "candidate",
                        "content": user_message,
                    }
                )
                current_q.current_depth += 1

                if interview_type == "behavior":
                    # 인성 면접: 꼬리질문 없이 바로 다음 질문으로
                    logger.info("📋 [인성 면접] 꼬리질문 스킵 → Q%s 완료", current_q_id)
                    current_q.is_completed = True
                elif current_q.current_depth < current_q.max_depth:
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

                    safe_info(
                        logger,
                        "🔍 [꼬리질문 진단] Q%s depth=%s/%s → 꼬리질문 생성 시도",
                        current_q_id,
                        current_q.current_depth,
                        current_q.max_depth,
                    )
                    if await http_request.is_disconnected():
                        logger.info("[Chat][INTERVIEW] 클라이언트 연결 해제 — 스트림 종료")
                        return
                    yield ": keepalive\n\n"

                    if model_choice == "vllm" and rag.vllm:
                        async for chunk in rag.vllm.generate_response(
                            user_message=followup_prompt,
                            context=None,
                            history=[],
                            system_prompt=system_prompt,
                        ):
                            if await http_request.is_disconnected():
                                logger.info("[Chat][INTERVIEW] 클라이언트 연결 해제 — 스트림 종료")
                                return
                            record_ttft()
                            full_response += chunk
                            yield ": k\n\n"
                    else:
                        async for chunk in rag.llm.generate_response(
                            user_message=followup_prompt,
                            context=None,
                            history=[],
                            system_prompt=system_prompt,
                            user_id=request.user_id,
                        ):
                            if await http_request.is_disconnected():
                                logger.info("[Chat][INTERVIEW] 클라이언트 연결 해제 — 스트림 종료")
                                return
                            record_ttft()
                            full_response += chunk
                            yield ": k\n\n"

                    safe_info(
                        logger, "🔍 [꼬리질문 진단] LLM 응답 앞 400자: %s", full_response[:400]
                    )

                    try:
                        followup_data = extract_json_from_llm_response(full_response)
                        if followup_data is None:
                            # JSON 파싱 실패 — 사용자에게 에러 알림 후 현재 질문 완료 처리
                            logger.error(
                                "꼬리질문 JSON 파싱 실패 (LLM 응답 앞 200자): %s",
                                full_response[:200],
                            )
                            yield sse_error_event(
                                code="PARSE_FAILED",
                                status=500,
                                message="꼬리질문 JSON 파싱 실패",
                                fallback="꼬리질문 생성 중 오류가 발생했습니다. 다음 질문으로 넘어갑니다.",
                            )
                            current_q.is_completed = True
                        else:
                            safe_info(
                                logger,
                                "🔍 [꼬리질문 진단] should_continue=%s | followup 존재=%s",
                                followup_data.get("should_continue"),
                                bool(followup_data.get("followup")),
                            )

                            if followup_data.get("should_continue", True) and followup_data.get(
                                "followup"
                            ):
                                followup_q = followup_data["followup"]["question"]
                                current_q.conversation.append(
                                    {
                                        "role": "interviewer",
                                        "content": followup_q,
                                    }
                                )
                                session.phase = "followup"

                                followup_header = format_followup_question_label(
                                    current_q_id, current_q.current_depth
                                )
                                followup_text = f"{followup_header}{newline}{followup_q}"
                                async for chunk in stream_text_chars(followup_text, sse_end):
                                    if await http_request.is_disconnected():
                                        logger.info(
                                            "[Chat][INTERVIEW] 클라이언트 연결 해제 — 스트림 종료"
                                        )
                                        return
                                    yield chunk
                            else:
                                safe_info(
                                    logger,
                                    "🔍 [꼬리질문 진단] 꼬리질문 스킵 → is_completed=True (Q%s)",
                                    current_q_id,
                                )
                                current_q.is_completed = True

                                # ADR-066: answer_quality 파싱 → 마스터한 질문 수집
                                analysis = followup_data.get("analysis", {})
                                answer_quality = analysis.get("answer_quality", "good")
                                if interview_type == "tech" and is_mastered_quality(answer_quality):
                                    try:
                                        q_embedding = await rag.vectordb.create_embedding(
                                            current_q.question
                                        )
                                        session.mastered_questions.append(
                                            {
                                                "question": current_q.question,
                                                "embedding": q_embedding,
                                            }
                                        )
                                        logger.info(
                                            "ADR-066: 마스터 질문 등록 (quality=%s): %s",
                                            answer_quality,
                                            current_q.question[:50],
                                        )
                                    except Exception as e:
                                        logger.debug("ADR-066: 마스터 임베딩 실패: %s", e)
                    except Exception as e:
                        logger.error("꼬리질문 처리 중 예외 발생: %s", type(e).__name__)
                        yield sse_error_event(
                            code="PARSE_FAILED",
                            status=500,
                            message=f"꼬리질문 처리 실패: {type(e).__name__}",
                            fallback="꼬리질문 생성 중 오류가 발생했습니다. 다음 질문으로 넘어갑니다.",
                        )
                        current_q.is_completed = True

                else:
                    current_q.is_completed = True

                if current_q.is_completed:
                    next_q_id = current_q_id + 1
                    if next_q_id <= session.total_questions:
                        next_q = next((q for q in session.questions if q.id == next_q_id), None)
                        if next_q:
                            session.current_question_id = next_q_id
                            session.phase = "questioning"

                            question_header = format_main_question_label(next_q_id)
                            question_text = f"{question_header}{newline}{next_q.question}"
                            async for chunk in stream_text_chars(question_text, sse_end):
                                if await http_request.is_disconnected():
                                    logger.info(
                                        "[Chat][INTERVIEW] 클라이언트 연결 해제 — 스트림 종료"
                                    )
                                    return
                                yield chunk

                            next_q.conversation.append(
                                {
                                    "role": "interviewer",
                                    "content": next_q.question,
                                }
                            )
                    else:
                        session.phase = "completed"
                        complete_msg = f"{newline}{newline}면접 결과 리포트를 생성 중입니다. 잠시만 기다려 주세요."
                        async for chunk in stream_text_chars(complete_msg, sse_end):
                            if await http_request.is_disconnected():
                                logger.info("[Chat][INTERVIEW] 클라이언트 연결 해제 — 스트림 종료")
                                return
                            yield chunk

                await session_store.set(session_key, session.model_dump())
                safe_info(
                    logger,
                    "💾 [면접] 세션 업데이트: %s, phase=%s, Q%s/5",
                    session_key,
                    session.phase,
                    session.current_question_id,
                )

                if session.phase == "completed":
                    await session_store.delete(session_key)
                    safe_info(logger, "🗑️ [면접] 완료된 세션 삭제: %s", session_key)

                session_meta = {
                    "type": "session_state",
                    "session": session.model_dump(),
                }
                yield f"data: {json.dumps(session_meta, ensure_ascii=False)}{sse_end}"
                yield f"data: [DONE]{sse_end}"

            # PHASE 3: 면접 완료
            elif session.phase == "completed":
                await session_store.delete(session_key)
                complete_msg = "면접이 이미 완료되었습니다. 리포트 모드에서 결과를 확인하세요."
                yield f"data: {json.dumps({'chunk': complete_msg}, ensure_ascii=False)}{sse_end}"
                yield f"data: [DONE]{sse_end}"

        except asyncio.CancelledError:
            logger.info("[Chat][INTERVIEW] 스트림 취소됨 (클라이언트 연결 해제)")
            return
        except Exception as e:
            logger.error(f"Interview error: {e}", exc_info=True)
            yield sse_error_event(
                code="INTERNAL_ERROR",
                status=500,
                message=str(e),
                fallback="면접 진행 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            )
            yield f"data: [DONE]{sse_end}"

    # Latency 측정 및 로깅 (PLG 스택 로그 활용)
    try:
        total_time = time.time() - start_time
        duration_ms = total_time * 1000

        ttft_str = "N/A"
        gen_time_str = "N/A"

        if first_token_time is not None:
            ttft = first_token_time - start_time
            gen_time = time.time() - first_token_time
            ttft_str = f"{ttft:.2f}s"
            gen_time_str = f"{gen_time:.2f}s"
            try:
                from app.core.monitoring import AI_GENERATION_DURATION

                AI_GENERATION_DURATION.labels(model=model, endpoint="/ai/chat").observe(gen_time)
            except Exception:
                pass

        safe_info(
            logger,
            f"📊 [LLM Stats] Mode={mode} | Model={model} | TTFT={ttft_str} | GenTime={gen_time_str} | TotalTime={total_time:.2f}s | UID={request.user_id}",
        )
        safe_info(logger, "⏱️ 채팅 처리 완료: %sms", f"{duration_ms:.2f}")
    except Exception as e:
        logger.error(f"Failed to record latency metric: {e}")


@router.post(
    "/chat",
    summary="채팅 스트리밍 (일반/면접)",
    description="""
    채팅 스트리밍 응답을 처리합니다.

    **처리 방식:** 스트리밍 (SSE)

    **모드:**
    - normal: 일반 대화
    - interview: 면접 질문 생성

    **면접 평가 리포트는 `/ai/evaluation/analyze`를 사용하세요.**

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
async def chat(
    request: ChatRequest,
    http_request: Request,
    session_store=Depends(get_session_store),
):
    """채팅 처리 (일반/면접)"""
    return StreamingResponse(
        generate_chat_stream(request, session_store, http_request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
