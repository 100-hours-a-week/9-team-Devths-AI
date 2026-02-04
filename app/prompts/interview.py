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


def load_question_examples(interview_type: str) -> str:
    """면접 유형에 맞는 질문 예시 로드"""
    if interview_type == "technical":
        return load_prompt("technical_questions")
    else:
        return load_prompt("personality_questions")


def create_interview_question_prompt(
    resume_text: str,
    job_posting_text: str,
    interview_type: str = "technical",
    asked_questions: list[str] | None = None,
) -> str:
    """면접 질문 생성 프롬프트 (asked_questions: 이미 한 질문 목록, 반복 방지용)"""
    interview_type_kr = "기술" if interview_type == "technical" else "인성"
    if asked_questions:
        asked_questions_section = (
            "\n## 이미 한 질문 목록 (아래와 겹치지 않는 새 질문만 생성)\n"
            + "\n".join(f"- {q}" for q in asked_questions)
        )
    else:
        asked_questions_section = ""
    template = load_prompt("interview_question")
    question_examples = load_question_examples(interview_type)
    return template.format(
        resume_text=resume_text,
        job_posting_text=job_posting_text,
        interview_type=interview_type_kr,
        asked_questions_section=asked_questions_section,
        question_examples=question_examples,
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
# 기술 면접 5단계 프롬프트 (신규)
# ============================================================================


def get_system_tech_interview() -> str:
    """기술 면접관 시스템 프롬프트"""
    return load_prompt("system_tech_interview")


def create_tech_interview_init_prompt(
    resume_text: str,
    job_posting_text: str,
    portfolio_text: str = "",
) -> str:
    """
    기술 면접 초기 질문 세트 생성 프롬프트
    - 5개 카테고리별 질문 생성
    - JSON 형식으로 반환
    """
    template = load_prompt("tech_interview_init")
    return template.format(
        resume_text=resume_text or "(이력서 정보 없음)",
        job_posting_text=job_posting_text or "(채용공고 정보 없음)",
        portfolio_text=portfolio_text or "(포트폴리오 정보 없음)",
    )


def create_tech_followup_prompt(
    question_id: int,
    category_name: str,
    original_question: str,
    conversation_history: str,
    last_answer: str,
    current_depth: int,
) -> str:
    """
    기술 면접 꼬리질문 생성 프롬프트
    - 현재 깊이에 따른 질문 전략 적용
    - 최대 3 depths
    """
    template = load_prompt("tech_followup")
    remaining_followups = 3 - current_depth
    return template.format(
        question_id=question_id,
        category_name=category_name,
        original_question=original_question,
        conversation_history=conversation_history,
        last_answer=last_answer,
        current_depth=current_depth,
        remaining_followups=remaining_followups,
    )


def create_tech_next_question_prompt(
    completed_questions: str,
    next_question_id: int,
    next_category_name: str,
    next_question: str,
) -> str:
    """
    다음 주제 질문으로 전환 프롬프트
    - 자연스러운 전환 멘트 생성
    """
    template = load_prompt("tech_next_question")
    return template.format(
        completed_questions=completed_questions,
        next_question_id=next_question_id,
        next_category_name=next_category_name,
        next_question=next_question,
    )


def format_conversation_history(qa_pairs: list[dict]) -> str:
    """
    대화 이력을 프롬프트용 문자열로 포맷팅
    qa_pairs: [{"role": "interviewer|candidate", "content": "..."}, ...]
    """
    if not qa_pairs:
        return "(이전 대화 없음)"

    formatted = []
    for item in qa_pairs:
        role = item.get("role", "unknown")
        content = item.get("content", "")
        if role == "interviewer":
            formatted.append(f"면접관: {content}")
        elif role == "candidate":
            formatted.append(f"지원자: {content}")

    return "\n".join(formatted)


def format_completed_questions(questions: list[dict]) -> str:
    """
    완료된 질문들을 프롬프트용 문자열로 포맷팅
    questions: [{"id": 1, "category_name": "...", "question": "...", "summary": "..."}, ...]
    """
    if not questions:
        return "(완료된 질문 없음)"

    formatted = []
    for q in questions:
        formatted.append(
            f"- [{q.get('id', '?')}] {q.get('category_name', '')}: {q.get('question', '')}"
        )

    return "\n".join(formatted)


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
