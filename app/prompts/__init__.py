"""
Prompt 모듈
모든 LLM 프롬프트를 중앙에서 관리합니다.
"""

from app.prompts.interview import (
    # 시스템 프롬프트
    SYSTEM_GENERAL,
    SYSTEM_ANALYSIS,
    SYSTEM_INTERVIEW,
    SYSTEM_FOLLOWUP,
    # 프롬프트 템플릿
    ANALYSIS_PROMPT,
    INTERVIEW_QUESTION_PROMPT,
    FOLLOWUP_PROMPT,
    INTERVIEW_REPORT_PROMPT,
    # 헬퍼 함수
    create_analysis_prompt,
    create_interview_question_prompt,
    create_followup_prompt,
    create_interview_report_prompt,
)

from app.prompts.chat import (
    SYSTEM_GENERAL_CHAT,
    SYSTEM_RAG_CHAT,
    RAG_CONTEXT_PROMPT,
    GENERAL_CHAT_PROMPT,
    create_rag_prompt,
    get_system_prompt,
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
    # Chat prompts
    "SYSTEM_GENERAL_CHAT",
    "SYSTEM_RAG_CHAT",
    "RAG_CONTEXT_PROMPT",
    "GENERAL_CHAT_PROMPT",
    "create_rag_prompt",
    "get_system_prompt",
]
