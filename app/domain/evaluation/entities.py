"""
Evaluation Domain Entities.

Defines data structures for interview analysis and debate evaluation.
"""

from dataclasses import dataclass, field
from typing import Any, TypedDict


# ============================================
# 1단계: Gemini 면접 분석 결과
# ============================================


@dataclass
class QuestionAnalysis:
    """단일 질문에 대한 분석 결과."""

    question: str
    user_answer: str
    verdict: str  # "적절" | "부적절" | "보완필요"
    score: int  # 1-5
    reasoning: str  # 판단 근거
    recommended_answer: str | None = None  # 보완/부적절 시 추천 답변
    category: str = ""  # 질문 카테고리

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "question": self.question,
            "user_answer": self.user_answer,
            "verdict": self.verdict,
            "score": self.score,
            "reasoning": self.reasoning,
            "recommended_answer": self.recommended_answer,
            "category": self.category,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "QuestionAnalysis":
        """Create from dictionary."""
        return cls(
            question=data.get("question", ""),
            user_answer=data.get("user_answer", ""),
            verdict=data.get("verdict", "보완필요"),
            score=data.get("score", 3),
            reasoning=data.get("reasoning", ""),
            recommended_answer=data.get("recommended_answer"),
            category=data.get("category", ""),
        )


@dataclass
class InterviewAnalysis:
    """전체 면접 분석 결과 (1단계 Gemini 분석)."""

    questions: list[QuestionAnalysis] = field(default_factory=list)
    overall_score: int = 0  # 1-5 종합 점수
    overall_feedback: str = ""
    strengths: list[str] = field(default_factory=list)
    improvements: list[str] = field(default_factory=list)
    model_used: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "questions": [q.to_dict() for q in self.questions],
            "overall_score": self.overall_score,
            "overall_feedback": self.overall_feedback,
            "strengths": self.strengths,
            "improvements": self.improvements,
            "model_used": self.model_used,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InterviewAnalysis":
        """Create from dictionary."""
        return cls(
            questions=[
                QuestionAnalysis.from_dict(q) for q in data.get("questions", [])
            ],
            overall_score=data.get("overall_score", 0),
            overall_feedback=data.get("overall_feedback", ""),
            strengths=data.get("strengths", []),
            improvements=data.get("improvements", []),
            model_used=data.get("model_used", ""),
        )


# ============================================
# 2단계: LangGraph 토론 상태
# ============================================


class DebateState(TypedDict):
    """LangGraph 토론 상태."""

    # 입력 데이터
    qa_pairs: list[dict[str, Any]]  # [{"question": ..., "answer": ...}, ...]
    resume_text: str
    job_posting_text: str
    interview_type: str

    # 1단계 Gemini 분석 결과 (이미 완료된 것)
    gemini_analysis: dict[str, Any]

    # 2단계 GPT-4o 분석 결과
    gpt4o_analysis: dict[str, Any] | None

    # 비교 결과
    disagreements: list[dict[str, Any]]  # 불일치 항목들
    agreements: list[dict[str, Any]]  # 일치 항목들

    # 토론 결과
    gemini_rebuttal: dict[str, Any] | None
    gpt4o_rebuttal: dict[str, Any] | None

    # 최종 결과
    final_analysis: dict[str, Any] | None
    consensus_method: str  # "single" | "merged" | "debated"

    # 메타데이터
    phase: str  # "loading" | "gpt4o_analyzing" | "comparing" | "debating" | "synthesizing" | "done"


@dataclass
class DebateResult:
    """토론 최종 결과."""

    final_analysis: InterviewAnalysis
    gemini_analysis: InterviewAnalysis
    gpt4o_analysis: InterviewAnalysis | None = None
    disagreements: list[dict[str, Any]] = field(default_factory=list)
    consensus_method: str = "single"  # "single" | "merged" | "debated"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "final_analysis": self.final_analysis.to_dict(),
            "gemini_analysis": self.gemini_analysis.to_dict(),
            "gpt4o_analysis": self.gpt4o_analysis.to_dict() if self.gpt4o_analysis else None,
            "disagreements": self.disagreements,
            "consensus_method": self.consensus_method,
        }
