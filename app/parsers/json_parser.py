"""
JSON 파서 유틸리티
LLM 응답에서 JSON을 추출하고 파싱하는 함수들을 제공합니다.

참고: 99.꼬리질문 테스트.ipynb
"""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def extract_json_from_text(text: str) -> dict[str, Any] | None:
    """
    텍스트에서 JSON을 추출합니다.

    여러 방법을 시도합니다:
    1. 전체 텍스트가 JSON인 경우
    2. ```json ... ``` 코드 블록 내부
    3. {...} 형태의 JSON 객체

    Args:
        text: JSON이 포함된 텍스트

    Returns:
        파싱된 JSON 딕셔너리 또는 None
    """
    if not text or not text.strip():
        return None

    # 1. 전체 텍스트가 JSON인 경우
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. ```json ... ``` 또는 ``` ... ``` 코드 블록 내부
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if json_match:
        try:
            return json.loads(json_match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    # 3. {...} 형태의 JSON 객체 찾기
    brace_match = re.search(r"\{[\s\S]*\}", text)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except (json.JSONDecodeError, ValueError):
            pass

    return None


def parse_llm_json_response(
    response_text: str, fallback: dict[str, Any] | None = None
) -> dict[str, Any]:
    """
    LLM 응답 텍스트에서 JSON을 파싱합니다.

    Args:
        response_text: LLM 응답 텍스트
        fallback: 파싱 실패 시 반환할 기본값

    Returns:
        파싱된 JSON 딕셔너리 또는 fallback
    """
    parsed = extract_json_from_text(response_text)

    if parsed:
        logger.debug("JSON 파싱 성공")
        return parsed
    else:
        logger.warning(f"JSON 파싱 실패: {response_text[:100]}...")
        if fallback:
            return fallback
        return {}


def safe_json_parse(text: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    안전하게 JSON을 파싱합니다 (예외 처리 포함).

    Args:
        text: 파싱할 텍스트
        default: 파싱 실패 시 기본값

    Returns:
        파싱된 JSON 딕셔너리 또는 default
    """
    try:
        parsed = extract_json_from_text(text)
        return parsed if parsed is not None else (default or {})
    except Exception as e:
        logger.error(f"JSON 파싱 중 오류 발생: {e}")
        return default or {}


def find_json_in_text(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """
    텍스트에서 JSON을 찾고 원본 JSON 문자열도 반환합니다.

    Args:
        text: 검색할 텍스트

    Returns:
        (파싱된 JSON 딕셔너리, 원본 JSON 문자열) 튜플
    """
    if not text:
        return None, None

    # {...} 형태의 JSON 객체 찾기
    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1

    if start_idx != -1 and end_idx > start_idx:
        json_str = text[start_idx:end_idx]
        try:
            parsed = json.loads(json_str)
            return parsed, json_str
        except (json.JSONDecodeError, ValueError):
            pass

    # extract_json_from_text로 재시도
    parsed = extract_json_from_text(text)
    if parsed:
        return parsed, None

    return None, None
