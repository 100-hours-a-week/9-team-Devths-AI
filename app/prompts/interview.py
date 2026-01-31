"""
면접 관련 프롬프트 정의
- 면접 질문 생성
- 꼬리질문 생성
- 면접 리포트 생성

마크다운 템플릿을 사용하는 하이브리드 방식
참고: 99.꼬리질문 테스트.ipynb
"""

from pathlib import Path

# ============================================================================
# 템플릿 로더
# ============================================================================

TEMPLATE_DIR = Path(__file__).parent / "templates" / "interview"


def load_prompt(template_name: str) -> str:
    """마크다운 템플릿 파일에서 프롬프트 로드"""
    template_path = TEMPLATE_DIR / f"{template_name}.md"
    try:
        return template_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Template not found: {template_path}") from None


# ============================================================================
# 시스템 프롬프트
# ============================================================================


def get_system_general() -> str:
    """일반 시스템 프롬프트"""
    return load_prompt("system_general")


def get_system_analysis() -> str:
    """분석 시스템 프롬프트"""
    return load_prompt("system_analysis")


def get_system_interview() -> str:
    """면접 시스템 프롬프트"""
    return load_prompt("system_interview")


def get_system_followup() -> str:
    """꼬리질문 시스템 프롬프트"""
    return load_prompt("system_followup")


# ============================================================================
# 헬퍼 함수
# ============================================================================


def create_analysis_prompt(resume_text: str, job_posting_text: str) -> str:
    """분석 프롬프트 생성"""
    template = load_prompt("analysis")
    return template.format(resume_text=resume_text, job_posting_text=job_posting_text)


def create_interview_question_prompt(
    resume_text: str, job_posting_text: str, interview_type: str = "technical"
) -> str:
    """면접 질문 생성 프롬프트"""
    interview_type_kr = "기술" if interview_type == "technical" else "인성"
    template = load_prompt("interview_question")
    return template.format(
        resume_text=resume_text, job_posting_text=job_posting_text, interview_type=interview_type_kr
    )


def create_followup_prompt(
    original_question: str, candidate_answer: str, star_analysis: dict | None = None
) -> str:
    """꼬리질문 생성 프롬프트"""
    if star_analysis is None:
        star_analysis = {
            "situation": "unknown",
            "task": "unknown",
            "action": "unknown",
            "result": "unknown",
        }

    template = load_prompt("followup")
    return template.format(
        original_question=original_question,
        candidate_answer=candidate_answer,
        star_situation=star_analysis.get("situation", "unknown"),
        star_task=star_analysis.get("task", "unknown"),
        star_action=star_analysis.get("action", "unknown"),
        star_result=star_analysis.get("result", "unknown"),
    )


def create_interview_report_prompt(
    qa_history: str, resume_text: str = "", job_posting_text: str = ""
) -> str:
    """면접 리포트 생성 프롬프트"""
    template = load_prompt("interview_report")
    return template.format(
        qa_history=qa_history,
        resume_text=resume_text or "(이력서 정보 없음)",
        job_posting_text=job_posting_text or "(채용공고 정보 없음)",
    )


# ============================================================================
# 역호환성을 위한 상수 (선택적)
# ============================================================================

# 기존 코드에서 직접 참조하는 경우를 위해 lazy loading
SYSTEM_GENERAL = get_system_general()
SYSTEM_ANALYSIS = get_system_analysis()
SYSTEM_INTERVIEW = get_system_interview()
SYSTEM_FOLLOWUP = get_system_followup()
ANALYSIS_PROMPT = load_prompt("analysis")
INTERVIEW_QUESTION_PROMPT = load_prompt("interview_question")
FOLLOWUP_PROMPT = load_prompt("followup")
INTERVIEW_REPORT_PROMPT = load_prompt("interview_report")
