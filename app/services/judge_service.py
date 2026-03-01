"""
ADR-091: LLM-as-Judge 서비스.

Gemini Judge LLM으로 RAG 파이프라인 응답 품질을 채점하고 Langfuse에 기록한다.

Judge 모델: settings.eval_gemini_model (gemini-3.1-pro-preview)
채점 기준: relevance / accuracy / fluency / completeness (각 1~5점)
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from app.config.settings import get_settings
from app.schemas.judge import CriterionScore, JudgeResult
from app.utils.langfuse_client import LangfuseTraceContext, record_score, trace_llm_call

logger = logging.getLogger(__name__)

# Judge 프롬프트 템플릿 로드
_JUDGE_PROMPT_PATH = (
    Path(__file__).parent.parent / "prompts" / "templates" / "evaluation" / "llm_judge.md"
)


def _load_judge_prompt() -> str:
    """Judge 프롬프트 템플릿 파일 로드."""
    try:
        return _JUDGE_PROMPT_PATH.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.error("Judge 프롬프트 파일 없음: %s", _JUDGE_PROMPT_PATH)
        raise


class JudgeService:
    """ADR-091: LLM-as-Judge — RAG 파이프라인 응답 품질 채점.

    Judge 모델: settings.eval_gemini_model (gemini-3.1-pro-preview).
    google-genai SDK 직접 사용 (InterviewAnalyzer와 동일 패턴).
    채점 결과는 Langfuse trace에 자동 기록된다.

    사용 예::

        judge = JudgeService()
        result = await judge.score(
            question="이 이력서의 강점은?",
            answer="Python과 FastAPI 경험이 풍부합니다.",
            pipeline_stage="rag",
        )
        print(result.overall_score)  # 4.2
    """

    JUDGE_CRITERIA: list[str] = ["relevance", "accuracy", "fluency", "completeness"]

    def __init__(self, api_key: str | None = None) -> None:
        settings = get_settings()
        resolved_key = api_key or settings.google_api_key
        if not resolved_key:
            raise ValueError("GOOGLE_API_KEY 또는 GEMINI_API_KEY가 필요합니다.")
        # 쉼표 구분 다중 키 지원 — 첫 번째 키 사용
        self.api_key = resolved_key.split(",")[0].strip()
        self.model_name = settings.eval_gemini_model
        self.client = genai.Client(api_key=self.api_key)
        self._prompt_template = _load_judge_prompt()
        logger.info("JudgeService initialized: model=%s (ADR-091)", self.model_name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def score(
        self,
        question: str,
        answer: str,
        pipeline_stage: str,
        reference_answer: str | None = None,
        user_id: str | None = None,
    ) -> JudgeResult:
        """Judge LLM으로 응답 품질 채점 후 Langfuse에 점수 기록.

        Args:
            question: 평가 대상 질문
            answer: 평가 대상 AI 응답
            pipeline_stage: "langchain_only" | "rag" | "vllm" | "sagemaker"
            reference_answer: 참조 답변 (없으면 None)
            user_id: Langfuse 사용자 ID 연동용 (선택)

        Returns:
            JudgeResult (기준별 점수 + 종합 점수 + 근거)
        """
        prompt = self._build_prompt(question, answer, reference_answer)

        # Langfuse trace 생성
        trace = trace_llm_call(
            name=f"llm_judge_{pipeline_stage}",
            user_id=user_id,
            metadata={"pipeline_stage": pipeline_stage, "adr": "091"},
        )

        raw_json = await self._call_judge(prompt)
        result = self._parse_result(raw_json, question, answer, pipeline_stage)

        # Langfuse에 점수 기록
        self._record_to_langfuse(trace, result)

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        question: str,
        answer: str,
        reference_answer: str | None,
    ) -> str:
        """Judge 프롬프트 생성."""
        return self._prompt_template.format(
            question=question,
            answer=answer,
            reference_answer=reference_answer or "없음",
        )

    async def _call_judge(self, prompt: str) -> str:
        """Gemini Judge 호출 — thinking 비활성화 (평가 일관성 우선)."""
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="NONE"),
            temperature=0.1,  # 채점 일관성을 위해 낮은 temperature
        )
        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            return self._extract_text(response)
        except Exception as e:
            logger.error("Judge LLM 호출 실패: %s", e)
            raise

    def _extract_text(self, response: Any) -> str:
        """Gemini 응답에서 텍스트 추출 (InterviewAnalyzer 동일 패턴)."""
        if not response.candidates:
            return ""
        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return ""
        texts = []
        for part in candidate.content.parts:
            if (
                hasattr(part, "text")
                and part.text
                and not (hasattr(part, "thought") and part.thought)
            ):
                texts.append(part.text)
        return "".join(texts)

    def _parse_result(
        self,
        raw_json: str,
        question: str,
        answer: str,
        pipeline_stage: str,
    ) -> JudgeResult:
        """Judge LLM 응답 JSON → JudgeResult 변환."""
        try:
            # ```json ... ``` 코드블록 제거
            clean = re.sub(r"```json\s*|\s*```", "", raw_json).strip()
            data: dict[str, Any] = json.loads(clean)
        except (json.JSONDecodeError, ValueError) as e:
            logger.error("Judge 응답 JSON 파싱 실패: %s\n원문: %.200s", e, raw_json)
            # 파싱 실패 시 기본 점수 반환
            return self._fallback_result(question, answer, pipeline_stage)

        scores = []
        for item in data.get("scores", []):
            try:
                scores.append(
                    CriterionScore(
                        name=item["name"],
                        score=float(item["score"]),
                        reasoning=item.get("reasoning", ""),
                    )
                )
            except Exception as parse_err:
                logger.warning("점수 항목 파싱 실패: %s — %s", item, parse_err)

        # 파싱된 점수가 없으면 기본 처리
        if not scores:
            return self._fallback_result(question, answer, pipeline_stage)

        overall = float(data.get("overall_score", self._avg_score(scores)))
        overall = max(1.0, min(5.0, overall))

        return JudgeResult(
            question=question,
            answer=answer,
            scores=scores,
            overall_score=overall,
            overall_reasoning=data.get("overall_reasoning", ""),
            pipeline_stage=pipeline_stage,
            judge_model=self.model_name,
        )

    def _fallback_result(self, question: str, answer: str, pipeline_stage: str) -> JudgeResult:
        """Judge 호출/파싱 실패 시 기본값 반환."""
        fallback_scores = [
            CriterionScore(name=c, score=1.0, reasoning="Judge 호출 실패로 기본값 반환")
            for c in self.JUDGE_CRITERIA
        ]
        return JudgeResult(
            question=question,
            answer=answer,
            scores=fallback_scores,
            overall_score=1.0,
            overall_reasoning="Judge 호출 또는 응답 파싱에 실패했습니다.",
            pipeline_stage=pipeline_stage,
            judge_model=self.model_name,
        )

    @staticmethod
    def _avg_score(scores: list[CriterionScore]) -> float:
        """기준별 점수 평균 계산."""
        if not scores:
            return 1.0
        return round(sum(s.score for s in scores) / len(scores), 2)

    def _record_to_langfuse(
        self,
        trace: "LangfuseTraceContext | None",
        result: JudgeResult,
    ) -> None:
        """채점 결과를 Langfuse trace에 점수로 기록."""
        if trace is None:
            return
        for criterion in result.scores:
            record_score(
                trace=trace,
                name=f"judge_{criterion.name}",
                value=criterion.score,
                comment=criterion.reasoning,
            )
        record_score(
            trace=trace,
            name="judge_overall",
            value=result.overall_score,
            comment=result.overall_reasoning,
        )
        logger.info(
            "ADR-091: Judge 채점 완료 — stage=%s overall=%.1f (relevance=%.1f, accuracy=%.1f)",
            result.pipeline_stage,
            result.overall_score,
            result.score_by_name("relevance") or 0,
            result.score_by_name("accuracy") or 0,
        )
