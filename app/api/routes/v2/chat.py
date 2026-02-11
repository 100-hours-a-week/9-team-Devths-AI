"""
v2 채팅 API (통합: 대화/분석/면접/리포트)

POST /ai/chat - 채팅 스트리밍 (SSE)
"""

import asyncio
import json
import logging
import re
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.routes.v2._helpers import get_services, get_session_key
from app.api.routes.v2._sse_errors import sse_error_event
from app.config.dependencies import get_session_store
from app.prompts import (
    create_tech_followup_prompt,
    create_tech_interview_init_prompt,
    format_conversation_history,
    get_extract_title_prompt,
    get_system_tech_interview,
)
from app.schemas.chat import (
    ChatMode,
    ChatRequest,
    InterviewQuestionState,
    InterviewSession,
)
from app.services.cloudwatch_service import CloudWatchService
from app.utils.log_sanitizer import safe_info, safe_warning
from app.utils.prompt_guard import RiskLevel, check_prompt_injection

logger = logging.getLogger(__name__)

router = APIRouter()


async def generate_chat_stream(
    request: ChatRequest,
    session_store,
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
    newline = "\n"
    sse_end = "\n\n"

    # 모니터링 시작
    start_time = time.time()

    model = request.model.value if hasattr(request.model, "value") else str(request.model)
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

    # =========================================================================
    # 1. 일반 대화 (RAG 활용)
    # =========================================================================
    if mode == ChatMode.NORMAL:
        full_response = ""

        try:
            history_dict = []
            user_message = request.message or ""
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
                    ):
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
                    ):
                        full_response += chunk
                        yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}{sse_end}"

                    logger.info("✅ 일반 대화 완료 (응답 길이: %d자)", len(full_response))

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
            interview_type = request.context.interview_type or "tech"
            interview_type_kr = "기술" if interview_type == "tech" else "인성"

            session_key = get_session_key(request.user_id, request.interview_id)
            user_message = request.message or ""

            session = request.context.interview_session
            if session is None:
                session_data = await session_store.get(session_key)
                session = InterviewSession.model_validate(session_data) if session_data else None
                if session:
                    safe_info(
                        logger,
                        "📦 [면접] 캐시에서 세션 복원: %s, phase=%s, Q%d/5",
                        session_key,
                        session.phase,
                        session.current_question_id,
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

                init_prompt = create_tech_interview_init_prompt(
                    resume_text=resume_ocr or context or "정보 없음",
                    job_posting_text=job_posting_ocr or "정보 없음",
                    portfolio_text=portfolio_text or context or "정보 없음",
                )

                system_prompt = get_system_tech_interview()

                if model_choice == "vllm" and rag.vllm:
                    full_response = ""
                    async for chunk in rag.vllm.generate_response(
                        user_message=init_prompt,
                        context=None,
                        history=[],
                        system_prompt=system_prompt,
                    ):
                        full_response += chunk
                else:
                    full_response = await rag.llm.generate_response_non_stream(
                        user_message=init_prompt,
                        context=None,
                        system_prompt=system_prompt,
                        user_id=request.user_id,
                    )

                # JSON 파싱하여 세션 생성
                try:
                    json_content = re.sub(r"```json\s*", "", full_response)
                    json_content = re.sub(r"```\s*$", "", json_content)
                    json_content = json_content.strip()

                    json_start = json_content.find("{")
                    json_end = json_content.rfind("}") + 1

                    if json_start != -1 and json_end > json_start:
                        json_str = json_content[json_start:json_end]
                        logger.info(f"JSON 파싱 시도 (첫 200자): {json_str[:200]}")

                        questions_data = json.loads(json_str)

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

                        await session_store.set(session_key, new_session.model_dump())
                        safe_info(logger, "💾 [면접] 세션 저장: %s", session_key)

                        first_q = new_session.questions[0] if new_session.questions else None
                        if first_q:
                            question_text = (
                                f"[{interview_type_kr}면접 1/5]{newline}{first_q.question}"
                            )
                            for char in question_text:
                                yield f"data: {json.dumps({'chunk': char}, ensure_ascii=False)}{sse_end}"
                                await asyncio.sleep(0.015)

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

                if current_q.current_depth < current_q.max_depth:
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

                    try:
                        json_start = full_response.find("{")
                        json_end = full_response.rfind("}") + 1
                        if json_start != -1 and json_end > json_start:
                            followup_data = json.loads(full_response[json_start:json_end])

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

                                followup_header = f"[{interview_type_kr}면접 {current_q_id}-{current_q.current_depth}/5]"
                                followup_text = f"{followup_header}{newline}{followup_q}"
                                for char in followup_text:
                                    yield f"data: {json.dumps({'chunk': char}, ensure_ascii=False)}{sse_end}"
                                    await asyncio.sleep(0.015)
                            else:
                                current_q.is_completed = True
                    except json.JSONDecodeError as e:
                        logger.error(f"꼬리질문 파싱 실패: {e}")
                        yield sse_error_event(
                            code="PARSE_FAILED",
                            status=500,
                            message=f"꼬리질문 JSON 파싱 실패: {e}",
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

                            question_header = f"[{interview_type_kr}면접 {next_q_id}/5]"
                            question_text = f"{question_header}{newline}{next_q.question}"
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
                        session.phase = "completed"
                        complete_msg = f"{newline}{newline}면접 결과 리포트를 생성 중입니다. 잠시만 기다려 주세요."
                        for char in complete_msg:
                            yield f"data: {json.dumps({'chunk': char}, ensure_ascii=False)}{sse_end}"
                            await asyncio.sleep(0.015)

                await session_store.set(session_key, session.model_dump())
                safe_info(
                    logger,
                    "💾 [면접] 세션 업데이트: %s, phase=%s, Q%d/5",
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

        except Exception as e:
            logger.error(f"Interview error: {e}", exc_info=True)
            yield sse_error_event(
                code="INTERNAL_ERROR",
                status=500,
                message=str(e),
                fallback="면접 진행 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            )
            yield f"data: [DONE]{sse_end}"

    # =========================================================================
    # 3. 리포트 모드 - 면접 평가 리포트 생성
    # =========================================================================
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

            qa_text = ""
            for i, qa in enumerate(qa_list, 1):
                q = qa.get("question", "")
                a = qa.get("answer", "")
                qa_text += f"질문 {i}: {q}\n답변 {i}: {a}\n\n"

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

            yield f"data: [DONE]{sse_end}"

        except Exception as e:
            logger.error(f"Interview report generation error: {e}", exc_info=True)
            yield sse_error_event(
                code="LLM_ERROR",
                status=500,
                message=str(e),
                fallback="면접 리포트 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요.",
            )
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
    session_store=Depends(get_session_store),
):
    """채팅 처리 (일반/면접)"""
    return StreamingResponse(
        generate_chat_stream(request, session_store),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
