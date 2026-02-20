"""
면접 답변 평가 프롬프트 정의
- Gemini 1차 분석
- GPT-4o 독립 분석
- 토론 반박/합의

마크다운 템플릿을 사용하는 하이브리드 방식
"""

from pathlib import Path
from typing import Any

# ============================================================================
# 템플릿 로더
# ============================================================================

TEMPLATE_DIR = Path(__file__).parent / "templates" / "evaluation"


def load_prompt(template_name: str) -> str:
    """마크다운 템플릿 파일에서 프롬프트 로드"""
    template_path = TEMPLATE_DIR / f"{template_name}.md"
    try:
        return template_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Template not found: {template_path}") from None


# ============================================================================
# Q&A 포맷팅 헬퍼
# ============================================================================


def format_qa_history(qa_pairs: list[dict[str, Any]]) -> str:
    """Q&A 이력을 프롬프트용 문자열로 변환.

    Args:
        qa_pairs: [{"question": "...", "answer": "...", "category": "..."}, ...]

    Returns:
        포맷팅된 Q&A 문자열
    """
    if not qa_pairs:
        return "(질의응답 기록 없음)"

    formatted = []
    for i, qa in enumerate(qa_pairs, 1):
        category = qa.get("category", "")
        category_str = f" [{category}]" if category else ""
        formatted.append(
            f"### 질문 {i}{category_str}\n"
            f"**Q**: {qa.get('question', '')}\n\n"
            f"**A**: {qa.get('answer', '')}"
        )

    return "\n\n---\n\n".join(formatted)


# ============================================================================
# 프롬프트 생성 함수
# ============================================================================


def create_analyze_prompt(
    qa_pairs: list[dict[str, Any]],
    resume_text: str = "",
    job_posting_text: str = "",
) -> str:
    """Gemini 1차 분석 프롬프트 생성 (1단계).

    Args:
        qa_pairs: 질의응답 목록
        resume_text: 이력서 텍스트
        job_posting_text: 채용공고 텍스트

    Returns:
        포맷팅된 분석 프롬프트
    """
    template = load_prompt("analyze_interview")
    return template.format(
        resume_text=resume_text or "(이력서 정보 없음)",
        job_posting_text=job_posting_text or "(채용공고 정보 없음)",
        qa_history=format_qa_history(qa_pairs),
    )


def create_gpt4o_analyze_prompt(
    qa_pairs: list[dict[str, Any]],
    resume_text: str = "",
    job_posting_text: str = "",
) -> str:
    """GPT-4o 독립 분석 프롬프트 생성 (2단계).

    Args:
        qa_pairs: 질의응답 목록
        resume_text: 이력서 텍스트
        job_posting_text: 채용공고 텍스트

    Returns:
        포맷팅된 GPT-4o 분석 프롬프트
    """
    template = load_prompt("gpt4o_analyze")
    return template.format(
        resume_text=resume_text or "(이력서 정보 없음)",
        job_posting_text=job_posting_text or "(채용공고 정보 없음)",
        qa_history=format_qa_history(qa_pairs),
    )


def create_debate_rebuttal_prompt(
    disagreement_details: str,
    your_analysis: str,
    other_analysis: str,
) -> str:
    """토론 반박 프롬프트 생성.

    Args:
        disagreement_details: 불일치 항목 설명
        your_analysis: 자신의 분석 (JSON 문자열)
        other_analysis: 상대 분석 (JSON 문자열)

    Returns:
        포맷팅된 토론 프롬프트
    """
    template = load_prompt("debate_rebuttal")
    return template.format(
        disagreement_details=disagreement_details,
        your_analysis=your_analysis,
        other_analysis=other_analysis,
    )


def create_synthesize_prompt(
    gemini_final: str,
    gpt4o_final: str,
    original_disagreements: str,
) -> str:
    """최종 합의 프롬프트 생성.

    Args:
        gemini_final: Gemini 최종 의견 (JSON 문자열)
        gpt4o_final: GPT-4o 최종 의견 (JSON 문자열)
        original_disagreements: 원래 불일치 항목

    Returns:
        포맷팅된 합의 프롬프트
    """
    template = load_prompt("synthesize_final")
    return template.format(
        gemini_final=gemini_final,
        gpt4o_final=gpt4o_final,
        original_disagreements=original_disagreements,
    )
