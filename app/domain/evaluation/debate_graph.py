"""
LangGraph Debate Workflow - 2단계 Gemini×OpenAI 토론.

사용자가 "심층 분석" 버튼을 눌렀을 때 실행되는 토론 그래프.
1단계 Gemini 분석 결과를 기반으로 GPT-4o가 독립 분석 후,
불일치 항목에 대해 토론하여 최종 합의를 도출합니다.

Flow:
  load_gemini_result → gpt4o_analyze → compare_analyses
    ├── (일치) → merge_results → END
    └── (불일치) → debate_round → synthesize_final → END
"""

import json
import logging
from typing import Any, Literal

from google import genai
from google.genai import types as genai_types
from langgraph.graph import END, StateGraph
from openai import AsyncOpenAI

from app.prompts.evaluation import (
    create_debate_rebuttal_prompt,
    create_gpt4o_analyze_prompt,
    create_synthesize_prompt,
)

from .entities import DebateResult, DebateState, InterviewAnalysis

logger = logging.getLogger(__name__)

# 불일치 판단 임계값: 점수 차이가 이 값 이상이면 불일치
SCORE_DISAGREEMENT_THRESHOLD = 2


def create_debate_graph(
    google_api_key: str,
    openai_api_key: str,
    gemini_model: str = "gemini-3-pro-preview",
    gpt_model: str = "gpt-4o",
    thinking_level: str = "HIGH",
) -> Any:
    """LangGraph 토론 그래프를 생성합니다.

    Args:
        google_api_key: Google API key
        openai_api_key: OpenAI API key
        gemini_model: Gemini model name
        gpt_model: GPT model name
        thinking_level: Gemini thinking level

    Returns:
        Compiled LangGraph state machine
    """
    # 클라이언트 초기화
    gemini_client = genai.Client(api_key=google_api_key)
    openai_client = AsyncOpenAI(api_key=openai_api_key)

    workflow = StateGraph(DebateState)

    # ========================================
    # Node 1: Gemini 분석 결과 로드
    # ========================================
    def load_gemini_result(state: DebateState) -> dict:
        """1단계에서 완료된 Gemini 분석 결과를 상태에 확인."""
        logger.info("Debate: Loading Gemini analysis result")
        gemini_analysis = state.get("gemini_analysis", {})

        if not gemini_analysis:
            logger.warning("No Gemini analysis found in state")

        return {
            "phase": "gpt4o_analyzing",
        }

    # ========================================
    # Node 2: GPT-4o 독립 분석
    # ========================================
    async def gpt4o_analyze(state: DebateState) -> dict:
        """GPT-4o가 동일 Q&A를 독립적으로 분석합니다."""
        logger.info("Debate: GPT-4o independent analysis starting")

        qa_pairs = state.get("qa_pairs", [])
        resume_text = state.get("resume_text", "")
        job_posting_text = state.get("job_posting_text", "")

        prompt = create_gpt4o_analyze_prompt(
            qa_pairs=qa_pairs,
            resume_text=resume_text,
            job_posting_text=job_posting_text,
        )

        try:
            response = await openai_client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {"role": "system", "content": "당신은 면접 답변 평가 전문가입니다. 반드시 JSON 형식으로만 응답해주세요."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
            )

            response_text = response.choices[0].message.content or ""
            gpt4o_analysis = _parse_json(response_text)

            logger.info(f"GPT-4o analysis complete: {len(gpt4o_analysis.get('questions', []))} questions")

            return {
                "gpt4o_analysis": gpt4o_analysis,
                "phase": "comparing",
            }
        except Exception as e:
            logger.error(f"GPT-4o analysis failed: {e}", exc_info=True)
            return {
                "gpt4o_analysis": None,
                "phase": "comparing",
            }

    # ========================================
    # Node 3: 분석 비교
    # ========================================
    def compare_analyses(state: DebateState) -> dict:
        """두 분석의 점수/verdict를 비교하여 불일치 항목 추출."""
        logger.info("Debate: Comparing analyses")

        gemini = state.get("gemini_analysis", {})
        gpt4o = state.get("gpt4o_analysis")

        # GPT-4o 분석이 실패한 경우
        if not gpt4o:
            logger.warning("GPT-4o analysis not available, using Gemini only")
            return {
                "disagreements": [],
                "agreements": gemini.get("questions", []),
                "phase": "done",
                "consensus_method": "single",
                "final_analysis": gemini,
            }

        gemini_questions = gemini.get("questions", [])
        gpt4o_questions = gpt4o.get("questions", [])

        disagreements = []
        agreements = []

        max_len = max(len(gemini_questions), len(gpt4o_questions))

        for i in range(max_len):
            g_q = gemini_questions[i] if i < len(gemini_questions) else {}
            o_q = gpt4o_questions[i] if i < len(gpt4o_questions) else {}

            g_score = g_q.get("score", 0)
            o_score = o_q.get("score", 0)
            score_diff = abs(g_score - o_score)

            g_verdict = g_q.get("verdict", "")
            o_verdict = o_q.get("verdict", "")

            if score_diff >= SCORE_DISAGREEMENT_THRESHOLD or g_verdict != o_verdict:
                disagreements.append({
                    "question_index": i,
                    "question": g_q.get("question", o_q.get("question", "")),
                    "gemini_score": g_score,
                    "gpt4o_score": o_score,
                    "score_diff": score_diff,
                    "gemini_verdict": g_verdict,
                    "gpt4o_verdict": o_verdict,
                    "gemini_reasoning": g_q.get("reasoning", ""),
                    "gpt4o_reasoning": o_q.get("reasoning", ""),
                })
            else:
                agreements.append({
                    "question_index": i,
                    "question": g_q.get("question", o_q.get("question", "")),
                    "agreed_score": round((g_score + o_score) / 2),
                    "agreed_verdict": g_verdict or o_verdict,
                })

        logger.info(
            f"Comparison: {len(agreements)} agreements, "
            f"{len(disagreements)} disagreements"
        )

        return {
            "disagreements": disagreements,
            "agreements": agreements,
            "phase": "debating" if disagreements else "done",
        }

    # ========================================
    # Node 4: 토론 라운드 (1회)
    # ========================================
    async def debate_round(state: DebateState) -> dict:
        """불일치 항목에 대해 Gemini와 GPT-4o가 각각 반박합니다."""
        logger.info("Debate: Starting debate round")

        disagreements = state.get("disagreements", [])
        gemini_analysis = state.get("gemini_analysis", {})
        gpt4o_analysis = state.get("gpt4o_analysis", {})

        if not disagreements:
            return {"phase": "synthesizing"}

        # 불일치 항목 포맷팅
        disagreement_text = _format_disagreements(disagreements)

        # Gemini 반박
        gemini_rebuttal = await _get_gemini_rebuttal(
            gemini_client=gemini_client,
            model=gemini_model,
            thinking_level=thinking_level,
            disagreement_details=disagreement_text,
            your_analysis=json.dumps(gemini_analysis, ensure_ascii=False),
            other_analysis=json.dumps(gpt4o_analysis, ensure_ascii=False),
        )

        # GPT-4o 반박
        gpt4o_rebuttal = await _get_gpt4o_rebuttal(
            openai_client=openai_client,
            model=gpt_model,
            disagreement_details=disagreement_text,
            your_analysis=json.dumps(gpt4o_analysis, ensure_ascii=False),
            other_analysis=json.dumps(gemini_analysis, ensure_ascii=False),
        )

        logger.info("Debate round complete")

        return {
            "gemini_rebuttal": gemini_rebuttal,
            "gpt4o_rebuttal": gpt4o_rebuttal,
            "phase": "synthesizing",
        }

    # ========================================
    # Node 5: 최종 합의 도출
    # ========================================
    async def synthesize_final(state: DebateState) -> dict:
        """토론 결과를 종합하여 최종 분석을 생성합니다."""
        logger.info("Debate: Synthesizing final analysis")

        gemini_rebuttal = state.get("gemini_rebuttal", {})
        gpt4o_rebuttal = state.get("gpt4o_rebuttal", {})
        disagreements = state.get("disagreements", [])

        prompt = create_synthesize_prompt(
            gemini_final=json.dumps(gemini_rebuttal, ensure_ascii=False),
            gpt4o_final=json.dumps(gpt4o_rebuttal, ensure_ascii=False),
            original_disagreements=_format_disagreements(disagreements),
        )

        # Gemini 3 Pro가 최종 합의 도출
        config = genai_types.GenerateContentConfig(
            thinking_config=genai_types.ThinkingConfig(
                thinking_level=thinking_level,
            ),
            temperature=0.3,
        )

        try:
            response = await gemini_client.aio.models.generate_content(
                model=gemini_model,
                contents=prompt,
                config=config,
            )

            response_text = _extract_gemini_text(response)
            final_analysis = _parse_json(response_text)

            logger.info("Final synthesis complete")

            return {
                "final_analysis": final_analysis,
                "consensus_method": "debated",
                "phase": "done",
            }
        except Exception as e:
            logger.error(f"Synthesis failed: {e}", exc_info=True)
            # 실패 시 Gemini 원본 분석 사용
            return {
                "final_analysis": state.get("gemini_analysis", {}),
                "consensus_method": "single",
                "phase": "done",
            }

    # ========================================
    # Node 6: 단순 병합 (불일치 없을 때)
    # ========================================
    def merge_results(state: DebateState) -> dict:
        """일치 시 두 분석을 가중 평균으로 병합합니다."""
        logger.info("Debate: Merging aligned results")

        gemini = state.get("gemini_analysis", {})
        gpt4o = state.get("gpt4o_analysis", {})

        if not gpt4o:
            return {
                "final_analysis": gemini,
                "consensus_method": "single",
                "phase": "done",
            }

        # 점수를 평균, 나머지는 Gemini 것 사용 (추천 답변은 둘 중 더 나은 것)
        merged = json.loads(json.dumps(gemini))  # deep copy
        gemini_qs = gemini.get("questions", [])
        gpt4o_qs = gpt4o.get("questions", [])

        for i in range(min(len(gemini_qs), len(gpt4o_qs))):
            avg_score = round((gemini_qs[i].get("score", 0) + gpt4o_qs[i].get("score", 0)) / 2)
            merged["questions"][i]["score"] = avg_score

            # GPT-4o의 추천 답변이 더 상세하면 사용
            gpt_rec = gpt4o_qs[i].get("recommended_answer")
            gem_rec = gemini_qs[i].get("recommended_answer")
            if gpt_rec and (not gem_rec or len(gpt_rec) > len(gem_rec)):
                merged["questions"][i]["recommended_answer"] = gpt_rec

        # 종합 점수 평균
        g_overall = gemini.get("overall_score", 0)
        o_overall = gpt4o.get("overall_score", 0)
        merged["overall_score"] = round((g_overall + o_overall) / 2)

        return {
            "final_analysis": merged,
            "consensus_method": "merged",
            "phase": "done",
        }

    # ========================================
    # 라우팅 함수
    # ========================================
    def route_after_compare(state: DebateState) -> Literal["debate_round", "merge_results"]:
        """비교 후 라우팅: 불일치 있으면 토론, 없으면 병합."""
        disagreements = state.get("disagreements", [])
        if disagreements:
            return "debate_round"
        return "merge_results"

    # ========================================
    # 그래프 구성
    # ========================================
    workflow.add_node("load_gemini_result", load_gemini_result)
    workflow.add_node("gpt4o_analyze", gpt4o_analyze)
    workflow.add_node("compare_analyses", compare_analyses)
    workflow.add_node("debate_round", debate_round)
    workflow.add_node("synthesize_final", synthesize_final)
    workflow.add_node("merge_results", merge_results)

    # 엣지 정의
    workflow.set_entry_point("load_gemini_result")
    workflow.add_edge("load_gemini_result", "gpt4o_analyze")
    workflow.add_edge("gpt4o_analyze", "compare_analyses")

    # 조건부 분기: 불일치 여부에 따라
    workflow.add_conditional_edges(
        "compare_analyses",
        route_after_compare,
        {
            "debate_round": "debate_round",
            "merge_results": "merge_results",
        },
    )

    workflow.add_edge("debate_round", "synthesize_final")
    workflow.add_edge("synthesize_final", END)
    workflow.add_edge("merge_results", END)

    return workflow.compile()


# ============================================
# 헬퍼 함수들
# ============================================


def _parse_json(text: str) -> dict[str, Any]:
    """JSON 파싱 (코드블록 처리 포함)."""
    cleaned = text.strip()
    if "```json" in cleaned:
        start = cleaned.index("```json") + 7
        end = cleaned.index("```", start)
        cleaned = cleaned[start:end].strip()
    elif "```" in cleaned:
        start = cleaned.index("```") + 3
        end = cleaned.index("```", start)
        cleaned = cleaned[start:end].strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        brace_start = cleaned.find("{")
        brace_end = cleaned.rfind("}") + 1
        if brace_start != -1 and brace_end > brace_start:
            try:
                return json.loads(cleaned[brace_start:brace_end])
            except json.JSONDecodeError:
                pass
        logger.warning(f"Failed to parse JSON: {cleaned[:200]}...")
        return {}


def _extract_gemini_text(response: Any) -> str:
    """Gemini 응답에서 텍스트 추출 (thinking 제외)."""
    if not response.candidates:
        return ""
    candidate = response.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return ""
    texts = []
    for part in candidate.content.parts:
        # thought 파트는 제외하고 text만 추출
        if hasattr(part, "text") and part.text and not (hasattr(part, "thought") and part.thought):
            texts.append(part.text)
    return "".join(texts)


def _format_disagreements(disagreements: list[dict[str, Any]]) -> str:
    """불일치 항목을 프롬프트용 텍스트로 포맷팅."""
    if not disagreements:
        return "(불일치 항목 없음)"

    lines = []
    for d in disagreements:
        lines.append(
            f"### 질문 {d['question_index'] + 1}: {d.get('question', '')}\n"
            f"- Gemini 점수: {d.get('gemini_score', '?')} ({d.get('gemini_verdict', '')})\n"
            f"  근거: {d.get('gemini_reasoning', '')}\n"
            f"- GPT-4o 점수: {d.get('gpt4o_score', '?')} ({d.get('gpt4o_verdict', '')})\n"
            f"  근거: {d.get('gpt4o_reasoning', '')}\n"
            f"- 점수 차이: {d.get('score_diff', 0)}"
        )
    return "\n\n".join(lines)


async def _get_gemini_rebuttal(
    gemini_client: genai.Client,
    model: str,
    thinking_level: str,
    disagreement_details: str,
    your_analysis: str,
    other_analysis: str,
) -> dict[str, Any]:
    """Gemini의 토론 반박을 생성합니다."""
    prompt = create_debate_rebuttal_prompt(
        disagreement_details=disagreement_details,
        your_analysis=your_analysis,
        other_analysis=other_analysis,
    )

    config = genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(thinking_level=thinking_level),
        temperature=0.3,
    )

    try:
        response = await gemini_client.aio.models.generate_content(
            model=model,
            contents=prompt,
            config=config,
        )
        return _parse_json(_extract_gemini_text(response))
    except Exception as e:
        logger.error(f"Gemini rebuttal failed: {e}")
        return {}


async def _get_gpt4o_rebuttal(
    openai_client: AsyncOpenAI,
    model: str,
    disagreement_details: str,
    your_analysis: str,
    other_analysis: str,
) -> dict[str, Any]:
    """GPT-4o의 토론 반박을 생성합니다."""
    prompt = create_debate_rebuttal_prompt(
        disagreement_details=disagreement_details,
        your_analysis=your_analysis,
        other_analysis=other_analysis,
    )

    try:
        response = await openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "당신은 면접 답변 평가 전문가입니다. JSON으로만 응답하세요."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return _parse_json(response.choices[0].message.content or "")
    except Exception as e:
        logger.error(f"GPT-4o rebuttal failed: {e}")
        return {}


# ============================================
# 서비스 클래스
# ============================================


class DebateService:
    """토론 서비스 - LangGraph 그래프를 관리합니다."""

    def __init__(
        self,
        google_api_key: str,
        openai_api_key: str,
        gemini_model: str = "gemini-3-pro-preview",
        gpt_model: str = "gpt-4o",
        thinking_level: str = "HIGH",
    ):
        """Initialize debate service."""
        self._google_api_key = google_api_key
        self._openai_api_key = openai_api_key
        self._gemini_model = gemini_model
        self._gpt_model = gpt_model
        self._thinking_level = thinking_level
        self._graph = create_debate_graph(
            google_api_key=google_api_key,
            openai_api_key=openai_api_key,
            gemini_model=gemini_model,
            gpt_model=gpt_model,
            thinking_level=thinking_level,
        )
        logger.info(
            f"DebateService initialized: gemini={gemini_model}, gpt={gpt_model}"
        )

    async def run_debate(
        self,
        qa_pairs: list[dict[str, Any]],
        gemini_analysis: dict[str, Any],
        resume_text: str = "",
        job_posting_text: str = "",
        interview_type: str = "tech",
    ) -> DebateResult:
        """토론을 실행하고 결과를 반환합니다.

        Args:
            qa_pairs: 면접 질의응답 목록
            gemini_analysis: 1단계 Gemini 분석 결과 (dict)
            resume_text: 이력서 텍스트
            job_posting_text: 채용공고 텍스트
            interview_type: 면접 유형

        Returns:
            DebateResult 토론 결과
        """
        initial_state: DebateState = {
            "qa_pairs": qa_pairs,
            "resume_text": resume_text,
            "job_posting_text": job_posting_text,
            "interview_type": interview_type,
            "gemini_analysis": gemini_analysis,
            "gpt4o_analysis": None,
            "disagreements": [],
            "agreements": [],
            "gemini_rebuttal": None,
            "gpt4o_rebuttal": None,
            "final_analysis": None,
            "consensus_method": "single",
            "phase": "loading",
        }

        # LangGraph 실행
        result = await self._graph.ainvoke(initial_state)

        # 결과 변환
        final_data = result.get("final_analysis", gemini_analysis)
        gpt4o_data = result.get("gpt4o_analysis")

        return DebateResult(
            final_analysis=InterviewAnalysis.from_dict(final_data) if final_data else InterviewAnalysis(),
            gemini_analysis=InterviewAnalysis.from_dict(gemini_analysis),
            gpt4o_analysis=InterviewAnalysis.from_dict(gpt4o_data) if gpt4o_data else None,
            disagreements=result.get("disagreements", []),
            consensus_method=result.get("consensus_method", "single"),
        )
