"""
ChromaDB 유틸리티 함수

ChromaDB where 필터 정규화 등 공통 로직을 제공합니다.
"""

from typing import Any


def normalize_chromadb_filter(where: dict[str, Any] | None) -> dict[str, Any] | None:
    """ChromaDB where 필터 정규화.

    - None이거나 빈 딕셔너리면 None 반환
    - user_id는 문자열로 변환 (ChromaDB 메타데이터는 문자열로 저장됨)
    - 빈 문자열 값은 제외

    Args:
        where: ChromaDB where 필터 딕셔너리

    Returns:
        정규화된 필터 딕셔너리 또는 None
    """
    if not where:
        return None

    normalized: dict[str, Any] = {}
    for k, v in where.items():
        if v is None:
            continue
        if k == "user_id":
            v_str = str(v).strip()
            if v_str:
                normalized[k] = v_str
        else:
            normalized[k] = v

    return normalized or None
