"""
로그 인젝션(Log Injection) 방어를 위한 유틸리티 모듈

CRLF Injection 공격을 방어하기 위해 개행 문자를 제거합니다.
"""

import re


def sanitize_log_input(input_value: str | int | None) -> str:
    """
    로그에 안전하게 기록할 수 있도록 문자열을 sanitize합니다.
    개행 문자(\n, \r)를 공백으로 치환하여 CRLF Injection을 방어합니다.

    Args:
        input_value: sanitize할 값 (str, int, None)

    Returns:
        sanitize된 문자열 (None 입력 시 "None" 반환)
    """
    if input_value is None:
        return "None"

    # 문자열로 변환
    text = str(input_value)

    # 개행 문자 제거 (CRLF Injection 방어)
    sanitized = re.sub(r"[\r\n]", " ", text)

    # 제어 문자 제거 (추가 보안)
    sanitized = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", sanitized)

    return sanitized
