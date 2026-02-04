"""
로그 인젝션(Log Injection) 방어를 위한 유틸리티 모듈

CRLF Injection 공격을 방어하기 위해 개행 문자를 제거합니다.
"""

import logging
import re

# 모듈 레벨 로거
_logger = logging.getLogger(__name__)


def _sanitize(text: str) -> str:
    """
    내부 sanitization 로직.
    개행 문자와 제어 문자를 제거합니다.
    """
    # 개행 문자 제거 (CRLF Injection 방어)
    result = re.sub(r"[\r\n]", " ", text)
    # 제어 문자 제거 (추가 보안)
    result = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", result)
    return result


def sanitize_log_input(input_value: str | int | None) -> str:
    """
    로그에 안전하게 기록할 수 있도록 문자열을 sanitize합니다.
    개행 문자(\\n, \\r)를 공백으로 치환하여 CRLF Injection을 방어합니다.

    Args:
        input_value: sanitize할 값 (str, int, None)

    Returns:
        sanitize된 문자열 (None 입력 시 "None" 반환)
    """
    if input_value is None:
        return "None"
    return _sanitize(str(input_value))


def safe_log(
    logger: logging.Logger,
    level: int,
    msg: str,
    *args: str | int | None,
) -> None:
    """
    안전한 로깅 함수.
    모든 인자를 sanitize한 후 로그를 기록합니다.

    Args:
        logger: 사용할 로거 인스턴스
        level: 로그 레벨 (logging.INFO, logging.ERROR 등)
        msg: 로그 메시지 포맷 문자열 (%s 사용)
        *args: 포맷 문자열에 전달할 인자들 (자동으로 sanitize됨)

    Example:
        safe_log(logger, logging.INFO, "User ID: %s, Room: %s", user_id, room_id)
    """
    # 모든 인자를 sanitize
    sanitized_args = tuple(_sanitize(str(arg)) if arg is not None else "None" for arg in args)
    logger.log(level, msg, *sanitized_args)


def safe_info(logger: logging.Logger, msg: str, *args: str | int | None) -> None:
    """INFO 레벨 안전 로깅"""
    safe_log(logger, logging.INFO, msg, *args)


def safe_error(logger: logging.Logger, msg: str, *args: str | int | None) -> None:
    """ERROR 레벨 안전 로깅"""
    safe_log(logger, logging.ERROR, msg, *args)


def safe_warning(logger: logging.Logger, msg: str, *args: str | int | None) -> None:
    """WARNING 레벨 안전 로깅"""
    safe_log(logger, logging.WARNING, msg, *args)
