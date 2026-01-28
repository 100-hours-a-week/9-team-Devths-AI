"""
일반 채팅 및 RAG 관련 프롬프트 정의
마크다운 템플릿을 사용하는 하이브리드 방식
"""

from pathlib import Path


# ============================================================================
# 템플릿 로더
# ============================================================================

TEMPLATE_DIR = Path(__file__).parent / "templates" / "chat"


def load_prompt(template_name: str) -> str:
    """마크다운 템플릿 파일에서 프롬프트 로드"""
    template_path = TEMPLATE_DIR / f"{template_name}.md"
    try:
        return template_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise FileNotFoundError(f"Template not found: {template_path}")


# ============================================================================
# 시스템 프롬프트
# ============================================================================

def get_system_general_chat() -> str:
    """일반 채팅용 시스템 프롬프트"""
    return load_prompt("system_general")


def get_system_rag_chat() -> str:
    """RAG 채팅용 시스템 프롬프트"""
    return load_prompt("system_rag")


# ============================================================================
# 헬퍼 함수
# ============================================================================


def create_rag_prompt(user_message: str, context: str | None = None) -> str:
    """RAG 컨텍스트가 있는 경우의 프롬프트 생성"""
    if context:
        template = load_prompt("rag_context")
        return template.format(context=context, user_message=user_message)
    
    template = load_prompt("general_chat")
    return template.format(user_message=user_message)


def get_system_prompt(use_rag: bool = False) -> str:
    """시스템 프롬프트 반환"""
    return get_system_rag_chat() if use_rag else get_system_general_chat()


# ============================================================================
# 역호환성을 위한 상수 (선택적)
# ============================================================================

# 기존 코드에서 직접 참조하는 경우를 위해 lazy loading
SYSTEM_GENERAL_CHAT = get_system_general_chat()
SYSTEM_RAG_CHAT = get_system_rag_chat()
RAG_CONTEXT_PROMPT = load_prompt("rag_context")
GENERAL_CHAT_PROMPT = load_prompt("general_chat")
