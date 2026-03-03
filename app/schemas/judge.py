"""
ADR-091: LLM-as-Judge 스키마 정의.

RAG 파이프라인 응답 품질 채점을 위한 Pydantic 모델.
"""

from typing import Literal

from pydantic import BaseModel, Field


class CriterionScore(BaseModel):
    """단일 채점 기준 점수."""

    name: Literal["relevance", "accuracy", "fluency", "completeness"]
    score: float = Field(ge=1.0, le=5.0, description="채점 점수 (1.0~5.0)")
    reasoning: str = Field(description="채점 근거 (한국어)")


class JudgeResult(BaseModel):
    """Judge LLM 채점 결과."""

    question: str = Field(description="평가 대상 질문")
    answer: str = Field(description="평가 대상 응답")
    scores: list[CriterionScore] = Field(description="기준별 점수 목록")
    overall_score: float = Field(ge=1.0, le=5.0, description="종합 점수 (1.0~5.0)")
    overall_reasoning: str = Field(description="종합 평가 근거")
    pipeline_stage: str = Field(
        description="평가 대상 파이프라인 단계",
        examples=["langchain_only", "rag", "vllm", "sagemaker"],
    )
    judge_model: str = Field(default="", description="Judge LLM 모델명")

    def score_by_name(self, name: str) -> float | None:
        """기준명으로 점수 조회."""
        for s in self.scores:
            if s.name == name:
                return s.score
        return None


class JudgeRequest(BaseModel):
    """POST /ai/evaluation/llm-judge 요청 모델."""

    question: str = Field(description="평가할 질문")
    answer: str = Field(description="평가할 AI 응답")
    pipeline_stage: str = Field(
        description="파이프라인 단계",
        examples=["langchain_only", "rag", "vllm", "sagemaker"],
    )
    reference_answer: str | None = Field(default=None, description="참조 답변 (선택)")
    user_id: str | None = Field(default=None, description="사용자 ID (Langfuse 연동용)")


class JudgeResponse(BaseModel):
    """POST /ai/evaluation/llm-judge 응답 모델."""

    result: JudgeResult
    langfuse_trace_id: str | None = Field(default=None, description="Langfuse trace ID")
