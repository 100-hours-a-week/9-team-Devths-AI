"""
면접 관련 프롬프트 정의
- 면접 질문 생성
- 꼬리질문 생성
- 면접 리포트 생성

참고: 99.꼬리질문 테스트.ipynb
"""

from typing import Dict, Optional

# ============================================================================
# 시스템 프롬프트
# ============================================================================

SYSTEM_GENERAL = """당신은 취업 준비를 돕는 친절한 AI 어시스턴트입니다.
이력서 작성, 면접 준비, 취업 관련 질문에 대해 도움을 드립니다.
항상 한국어로 자연스럽게 응답하세요."""

SYSTEM_ANALYSIS = """당신은 채용 전문가입니다.
이력서와 채용공고를 분석하여 구체적이고 실용적인 피드백을 제공합니다.
강점, 약점, 개선점을 명확히 구분하여 설명하세요."""

SYSTEM_INTERVIEW = """당신은 전문 면접관입니다.
지원자의 역량을 공정하게 평가하고 성장을 도울 수 있는 질문을 생성합니다.
JSON 형식으로만 응답하세요."""

SYSTEM_FOLLOWUP = """당신은 면접관입니다. JSON 형식으로만 응답하세요."""


# ============================================================================
# 분석 프롬프트
# ============================================================================

ANALYSIS_PROMPT = """당신은 채용 전문가입니다. 아래 이력서와 채용공고를 분석하여 JSON 형식으로 응답해주세요.

## 이력서
{resume_text}

## 채용공고
{job_posting_text}

## 출력 형식
{{
  "job_posting": {{"company": "회사명", "position": "포지션", "skills": {{"required": [], "preferred": []}}}},
  "resume": {{"total_experience": "경력", "skills": [], "strengths": [], "weaknesses": []}},
  "matching": {{"overall_grade": "A-F", "overall_score": 0, "skill_matching": {{"matched": [], "missing": []}}}}
}}

JSON만 출력하세요.
"""


# ============================================================================
# 면접 질문 생성 프롬프트
# ============================================================================

INTERVIEW_QUESTION_PROMPT = """당신은 전문 면접관입니다. 지원자에게 적합한 면접 질문을 생성하세요.

## 이력서
{resume_text}

## 채용공고
{job_posting_text}

## 면접 유형
{interview_type}

## 규칙
1. 이력서와 채용공고를 참고하여 맞춤형 질문 생성
2. {interview_type} 면접에 적합한 질문
3. 한국어로 자연스러운 질문
4. 구체적이고 답변하기 좋은 질문

## 출력 형식
{{
  "question": "질문 내용",
  "category": "기술/인성/경험",
  "difficulty": "easy/medium/hard",
  "expected_answer_points": ["예상 답변 포인트1", "예상 답변 포인트2"]
}}

JSON만 출력하세요.
"""


# ============================================================================
# 꼬리질문 생성 프롬프트 (STAR 기반)
# ============================================================================

FOLLOWUP_PROMPT = """당신은 전문 면접관입니다. 지원자의 답변을 분석하고 적절한 꼬리질문을 생성하세요.

## 면접 질문
{original_question}

## 지원자 답변
{candidate_answer}

## STAR 분석
- Situation: {star_situation}
- Task: {star_task}
- Action: {star_action}
- Result: {star_result}

## 규칙
1. 부족한 STAR 요소 보완
2. 기술적 세부사항 확인
3. 한국어로 자연스러운 질문

## 출력 형식
{{"followup_type": "star_complement|technical_deep_dive|verification", "question": "질문", "focus_area": "목적", "expected_answer_elements": []}}

JSON만 출력하세요.
"""


# ============================================================================
# 면접 리포트 생성 프롬프트
# ============================================================================

INTERVIEW_REPORT_PROMPT = """당신은 면접 평가 전문가입니다. 면접 내용을 종합 분석하여 리포트를 작성하세요.

## 면접 Q&A
{qa_history}

## 이력서
{resume_text}

## 채용공고
{job_posting_text}

## 규칙
1. 각 답변을 STAR 기준으로 평가
2. 강점과 개선점을 구체적으로 명시
3. 학습 가이드 제공
4. 건설적이고 격려하는 톤 유지

## 출력 형식
{{
  "evaluations": [
    {{
      "question": "질문",
      "answer": "답변",
      "good_points": ["잘한 점1", "잘한 점2"],
      "improvements": ["개선점1", "개선점2"],
      "score": 85
    }}
  ],
  "strength_patterns": ["강점 패턴1", "강점 패턴2"],
  "weakness_patterns": ["약점 패턴1"],
  "learning_guide": ["학습 가이드1", "학습 가이드2"],
  "overall_score": 80,
  "overall_grade": "B+"
}}

JSON만 출력하세요.
"""


# ============================================================================
# 헬퍼 함수
# ============================================================================

def create_analysis_prompt(resume_text: str, job_posting_text: str) -> str:
    """분석 프롬프트 생성"""
    return ANALYSIS_PROMPT.format(
        resume_text=resume_text,
        job_posting_text=job_posting_text
    )


def create_interview_question_prompt(
    resume_text: str,
    job_posting_text: str,
    interview_type: str = "technical"
) -> str:
    """면접 질문 생성 프롬프트"""
    interview_type_kr = "기술" if interview_type == "technical" else "인성"
    return INTERVIEW_QUESTION_PROMPT.format(
        resume_text=resume_text,
        job_posting_text=job_posting_text,
        interview_type=interview_type_kr
    )


def create_followup_prompt(
    original_question: str,
    candidate_answer: str,
    star_analysis: Optional[Dict] = None
) -> str:
    """꼬리질문 생성 프롬프트"""
    if star_analysis is None:
        star_analysis = {
            "situation": "unknown",
            "task": "unknown",
            "action": "unknown",
            "result": "unknown"
        }
    
    return FOLLOWUP_PROMPT.format(
        original_question=original_question,
        candidate_answer=candidate_answer,
        star_situation=star_analysis.get("situation", "unknown"),
        star_task=star_analysis.get("task", "unknown"),
        star_action=star_analysis.get("action", "unknown"),
        star_result=star_analysis.get("result", "unknown")
    )


def create_interview_report_prompt(
    qa_history: str,
    resume_text: str = "",
    job_posting_text: str = ""
) -> str:
    """면접 리포트 생성 프롬프트"""
    return INTERVIEW_REPORT_PROMPT.format(
        qa_history=qa_history,
        resume_text=resume_text or "(이력서 정보 없음)",
        job_posting_text=job_posting_text or "(채용공고 정보 없음)"
    )
