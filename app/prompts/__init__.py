"""
Prompt 모듈
모든 LLM 프롬프트를 중앙에서 관리합니다.
"""

from app.prompts.chat import (
    GENERAL_CHAT_PROMPT,
    RAG_CONTEXT_PROMPT,
    SYSTEM_GENERAL_CHAT,
    SYSTEM_RAG_CHAT,
    create_rag_prompt,
    get_extract_title_prompt,
    get_opening_prompt,
    get_system_prompt,
)
from app.prompts.interview import (
    # 프롬프트 템플릿
    ANALYSIS_PROMPT,
    FOLLOWUP_PROMPT,
    INTERVIEW_QUESTION_PROMPT,
    INTERVIEW_REPORT_PROMPT,
    SYSTEM_ANALYSIS,
    SYSTEM_FOLLOWUP,
    # 시스템 프롬프트
    SYSTEM_GENERAL,
    SYSTEM_INTERVIEW,
    # 헬퍼 함수
    create_analysis_prompt,
    create_followup_prompt,
    create_interview_question_prompt,
    create_interview_report_prompt,
    # 기술 면접 5단계 프롬프트 (신규)
    get_system_tech_interview,
    create_tech_interview_init_prompt,
    create_tech_followup_prompt,
    create_tech_next_question_prompt,
    format_conversation_history,
    format_completed_questions,
)

__all__ = [
    # Interview prompts
    "SYSTEM_GENERAL",
    "SYSTEM_ANALYSIS",
    "SYSTEM_INTERVIEW",
    "SYSTEM_FOLLOWUP",
    "ANALYSIS_PROMPT",
    "INTERVIEW_QUESTION_PROMPT",
    "FOLLOWUP_PROMPT",
    "INTERVIEW_REPORT_PROMPT",
    "create_analysis_prompt",
    "create_interview_question_prompt",
    "create_followup_prompt",
    "create_interview_report_prompt",
    # 기술 면접 5단계 프롬프트 (신규)
    "get_system_tech_interview",
    "create_tech_interview_init_prompt",
    "create_tech_followup_prompt",
    "create_tech_next_question_prompt",
    "format_conversation_history",
    "format_completed_questions",
    # Chat prompts
    "SYSTEM_GENERAL_CHAT",
    "SYSTEM_RAG_CHAT",
    "RAG_CONTEXT_PROMPT",
    "GENERAL_CHAT_PROMPT",
    "create_rag_prompt",
    "get_extract_title_prompt",
    "get_system_prompt",
]
