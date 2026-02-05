"""
프롬프트 인젝션 방어 모듈
- API 호출 전 사용자 입력 검사
- 위험 패턴 감지 시 경고 또는 차단
"""

import re
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    """위험 수준"""

    SAFE = "safe"
    WARNING = "warning"  # 경고만 (로깅)
    BLOCK = "block"  # 차단 (응답 거부)


@dataclass
class GuardResult:
    """검사 결과"""

    risk_level: RiskLevel
    matched_patterns: list[str]
    message: str


# ============================================================================
# 위험 패턴 정의
# ============================================================================

# 차단 패턴 (BLOCK) - 명확한 인젝션 시도
BLOCK_PATTERNS: list[tuple[str, str]] = [
    # 시스템 프롬프트 탈취 시도
    (r"시스템\s*프롬프트", "시스템 프롬프트 요청"),
    (r"system\s*prompt", "system prompt request"),
    (r"내부\s*지침", "내부 지침 요청"),
    (r"설정\s*(정보|값|내용)", "설정 정보 요청"),
    (r"지침\s*(보여|알려|출력)", "지침 공개 요청"),
    (r"프롬프트\s*(보여|알려|출력|확인)", "프롬프트 공개 요청"),
    # 역할 변경/탈옥 시도
    (r"이제부터\s*(너는|넌|당신은)", "역할 변경 시도"),
    (r"(DAN|jailbreak)\s*mode", "탈옥 시도"),
    (r"개발자\s*모드", "개발자 모드 요청"),
    (r"제한\s*(해제|풀어|없애)", "제한 해제 시도"),
    # 인젝션 명령어
    (r"ignore\s*(all\s*)?(previous|above)", "ignore instructions"),
    (r"이전\s*(지시|명령|지침)\s*(무시|잊어)", "이전 지시 무시"),
    (r"forget\s*(everything|all|previous)", "forget instructions"),
    (r"disregard\s*(all|previous|above)", "disregard instructions"),
    # 프롬프트 리킹
    (r"repeat\s*(the|your)\s*(prompt|instruction)", "prompt leak attempt"),
    (r"echo\s*(back|your)\s*(instruction|prompt)", "echo instructions"),
    (r"print\s*(your|the)\s*(prompt|system)", "print prompt"),
    (r"(출력|보여줘|알려줘).*?(위|앞|처음).*?(내용|텍스트|글)", "컨텍스트 리킹 시도"),
]

# 경고 패턴 (WARNING) - 의심스러운 요청
WARNING_PATTERNS: list[tuple[str, str]] = [
    # 디버깅/테스트 명목
    (r"디버깅\s*(중|목적|용)", "디버깅 명목"),
    (r"테스트\s*(중|목적|용)", "테스트 명목"),
    (r"개발자\s*(인데|입니다|야)", "개발자 주장"),
    (r"관리자\s*(인데|입니다|야)", "관리자 주장"),
    # 역할 관련
    (r"(역할|페르소나)\s*(바꿔|변경)", "역할 변경 요청"),
    (r"(행동|동작)\s*(규칙|방식)\s*(바꿔|변경)", "규칙 변경 요청"),
    # 인코딩 시도
    (r"base64", "base64 인코딩 언급"),
    (r"\\x[0-9a-f]{2}", "hex 인코딩 감지"),
    (r"&#\d+;", "HTML 엔티티 감지"),
]


# ============================================================================
# 검사 함수
# ============================================================================


def check_prompt_injection(user_input: str) -> GuardResult:
    """
    사용자 입력에서 프롬프트 인젝션 시도 검사

    Args:
        user_input: 사용자 입력 텍스트

    Returns:
        GuardResult: 검사 결과 (risk_level, matched_patterns, message)
    """
    if not user_input:
        return GuardResult(
            risk_level=RiskLevel.SAFE, matched_patterns=[], message="입력 없음"
        )

    # 소문자 변환 (대소문자 무시 검사)
    input_lower = user_input.lower()
    matched_patterns: list[str] = []

    # 차단 패턴 검사
    for pattern, description in BLOCK_PATTERNS:
        if re.search(pattern, input_lower, re.IGNORECASE):
            matched_patterns.append(f"[BLOCK] {description}")

    if matched_patterns:
        return GuardResult(
            risk_level=RiskLevel.BLOCK,
            matched_patterns=matched_patterns,
            message="보안 및 내부 운영 정책에 따라 시스템의 핵심 지침과 설정 정보는 보호되어야 하는 영역이므로, 개발자님의 디버깅 목적이라 하더라도 채팅창을 통한 직접 확인은 제한됩니다.",
        )

    # 경고 패턴 검사
    for pattern, description in WARNING_PATTERNS:
        if re.search(pattern, input_lower, re.IGNORECASE):
            matched_patterns.append(f"[WARNING] {description}")

    if matched_patterns:
        return GuardResult(
            risk_level=RiskLevel.WARNING,
            matched_patterns=matched_patterns,
            message="의심스러운 패턴 감지됨 (로깅만)",
        )

    return GuardResult(
        risk_level=RiskLevel.SAFE, matched_patterns=[], message="안전한 입력"
    )


def should_block_request(user_input: str) -> tuple[bool, str | None]:
    """
    요청 차단 여부 판단 (간단한 인터페이스)

    Args:
        user_input: 사용자 입력

    Returns:
        tuple[bool, str | None]: (차단 여부, 차단 시 응답 메시지)
    """
    result = check_prompt_injection(user_input)

    if result.risk_level == RiskLevel.BLOCK:
        return True, result.message

    return False, None


def get_safe_response() -> str:
    """차단 시 반환할 안전한 응답 메시지"""
    return "보안 및 내부 운영 정책에 따라 시스템의 핵심 지침과 설정 정보는 보호되어야 하는 영역이므로, 개발자님의 디버깅 목적이라 하더라도 채팅창을 통한 직접 확인은 제한됩니다."
