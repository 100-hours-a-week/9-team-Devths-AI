"""
Interview Analyzer - 1단계 Gemini 3 Pro 면접 답변 분석.

면접 종료 시 호출되어 전체 Q&A를 분석하고 피드백을 생성합니다.
google-genai SDK 직접 사용 (thinking 기능 활용).
"""

import json
import logging
import os
from typing import Any

from google import genai
from google.genai import types

from app.prompts.evaluation import create_analyze_prompt

from .entities import InterviewAnalysis, QuestionAnalysis

logger = logging.getLogger(__name__)


class InterviewAnalyzer:
    """Gemini 3 Pro 기반 면접 답변 분석기.

    면접 종료 후 전체 Q&A를 분석하여:
    - 각 답변의 적절성 평가 (적절/부적절/보완필요)
    - 부적절/보완필요 시 추천 답변 제공
    - 종합 점수 및 피드백 생성
    """

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str = "gemini-3-pro-preview",
        thinking_level: str = "HIGH",
    ):
        """Initialize analyzer.

        Args:
            api_key: Google API key.
            model_name: Gemini model for evaluation.
            thinking_level: Thinking depth (NONE, LOW, MEDIUM, HIGH).
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is required")

        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.thinking_level = thinking_level

        logger.info(
            f"InterviewAnalyzer initialized: model={model_name}, " f"thinking={thinking_level}"
        )

    async def analyze(
        self,
        qa_pairs: list[dict[str, Any]],
        resume_text: str = "",
        job_posting_text: str = "",
    ) -> InterviewAnalysis:
        """면접 Q&A 전체를 분석합니다.

        Args:
            qa_pairs: [{"question": "...", "answer": "...", "category": "..."}, ...]
            resume_text: 이력서 텍스트
            job_posting_text: 채용공고 텍스트

        Returns:
            InterviewAnalysis 분석 결과
        """
        if not qa_pairs:
            logger.warning("No Q&A pairs provided for analysis")
            return InterviewAnalysis(
                overall_feedback="분석할 질의응답이 없습니다.",
                model_used=self.model_name,
            )

        # 프롬프트 생성
        prompt = create_analyze_prompt(
            qa_pairs=qa_pairs,
            resume_text=resume_text,
            job_posting_text=job_posting_text,
        )

        # Gemini 3 Pro 호출 (thinking 기능 활용)
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level=self.thinking_level,
            ),
            temperature=0.3,  # 평가는 일관성이 중요하므로 낮은 temperature
        )

        try:
            response = await self.client.aio.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )

            # 응답 텍스트 추출
            response_text = self._extract_text(response)
            logger.debug(f"Gemini analysis response length: {len(response_text)}")

            # JSON 파싱
            analysis_data = self._parse_json_response(response_text)

            # InterviewAnalysis 객체로 변환
            analysis = self._build_analysis(analysis_data, qa_pairs)
            analysis.model_used = self.model_name

            logger.info(
                f"Interview analysis complete: "
                f"{len(analysis.questions)} questions, "
                f"overall_score={analysis.overall_score}"
            )

            return analysis

        except Exception as e:
            logger.error(f"Interview analysis failed: {e}", exc_info=True)
            # 실패 시 기본 분석 반환
            return InterviewAnalysis(
                overall_feedback=f"분석 중 오류가 발생했습니다: {str(e)}",
                model_used=self.model_name,
            )

    def _extract_text(self, response: Any) -> str:
        """Gemini 응답에서 텍스트 추출 (thinking 파트 제외)."""
        if not response.candidates:
            return ""

        candidate = response.candidates[0]
        if not candidate.content or not candidate.content.parts:
            return ""

        # thinking 파트를 제외하고 텍스트만 추출
        texts = []
        for part in candidate.content.parts:
            # thought 파트는 제외하고 text만 추출
            if (
                hasattr(part, "text")
                and part.text
                and not (hasattr(part, "thought") and part.thought)
            ):
                texts.append(part.text)

        return "".join(texts)

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        """응답 텍스트에서 JSON 파싱.

        코드블록 안의 JSON도 처리합니다.
        """
        # 코드블록 제거
        cleaned = text.strip()
        if "```json" in cleaned:
            start = cleaned.index("```json") + 7
            end = cleaned.index("```", start)
            cleaned = cleaned[start:end].strip()
        elif "```" in cleaned:
            start = cleaned.index("```") + 3
            end = cleaned.index("```", start)
            cleaned = cleaned[start:end].strip()

        # JSON 파싱 시도
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # { } 블록만 추출 시도
            brace_start = cleaned.find("{")
            brace_end = cleaned.rfind("}") + 1
            if brace_start != -1 and brace_end > brace_start:
                try:
                    return json.loads(cleaned[brace_start:brace_end])
                except json.JSONDecodeError:
                    pass

            logger.warning(f"Failed to parse JSON from response: {cleaned[:200]}...")
            return {}

    def _build_analysis(
        self,
        data: dict[str, Any],
        original_qa_pairs: list[dict[str, Any]],
    ) -> InterviewAnalysis:
        """파싱된 데이터를 InterviewAnalysis 객체로 변환."""
        questions = []
        raw_questions = data.get("questions", [])

        for i, q_data in enumerate(raw_questions):
            # 원본 Q&A에서 카테고리 가져오기
            category = ""
            if i < len(original_qa_pairs):
                category = original_qa_pairs[i].get("category", "")

            questions.append(
                QuestionAnalysis(
                    question=q_data.get("question", ""),
                    user_answer=q_data.get("user_answer", ""),
                    verdict=q_data.get("verdict", "보완필요"),
                    score=q_data.get("score", 3),
                    reasoning=q_data.get("reasoning", ""),
                    recommended_answer=q_data.get("recommended_answer"),
                    category=category,
                )
            )

        return InterviewAnalysis(
            questions=questions,
            overall_score=data.get("overall_score", 0),
            overall_feedback=data.get("overall_feedback", ""),
            strengths=data.get("strengths", []),
            improvements=data.get("improvements", []),
        )
