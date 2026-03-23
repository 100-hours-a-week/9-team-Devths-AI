"""
ChromaDB 유틸리티 함수

ChromaDB where 필터 정규화 등 공통 로직을 제공합니다.
"""

import importlib
import logging
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


def apply_chromadb_content_type_fix() -> None:
    """
    chromadb 0.5.x _make_request Content-Type 누락 버그 패치 (Issue #3899).

    _make_request에서 json= → orjson bytes 변환 시 Content-Type: application/json이
    누락되는 버그를 Request-level 헤더 주입으로 수정.
    httpx ByteStream이 session-level Content-Type을 덮어써서
    chroma_server_headers 워크어라운드가 무효화되는 문제 해결.

    chromadb 1.0.0에서 공식 수정됨. 해당 버전 업그레이드 시 이 패치는 무해(harmless).
    """
    _patch_logger = logging.getLogger(__name__)

    try:
        from chromadb.api.fastapi import FastAPI as ChromaFastAPI

        if getattr(ChromaFastAPI._make_request, "_content_type_patched", False):
            _patch_logger.debug("ChromaDB Content-Type 패치 이미 적용됨, 스킵")
            return

        orig = ChromaFastAPI._make_request

        def patched_make_request(self, method, path, **kwargs):
            if "json" in kwargs:
                kwargs.setdefault("headers", {})["Content-Type"] = "application/json"
            return orig(self, method, path, **kwargs)

        patched_make_request._content_type_patched = True
        ChromaFastAPI._make_request = patched_make_request
        _patch_logger.info("✅ ChromaDB _make_request Content-Type 패치 완료 (Issue #3899)")

    except Exception as e:
        _patch_logger.warning("ChromaDB Content-Type 패치 실패: %s", e)


def apply_chromadb_query_fix() -> None:
    """
    chromadb 0.4.x 버그 패치.

    Collection.query()에서 where_document=None → {} 변환으로
    "Expected where document to have exactly one operator, got {}" 오류 발생.

    FastAPI._query (HTTP 모드) 및 SegmentAPI._query (로컬 모드)를
    클래스 레벨에서 패치하여 {}를 None으로 변환 후 원본 호출.

    chromadb 업그레이드 시 자동으로 무해(harmless)해지는 구조.
    """
    _patch_logger = logging.getLogger(__name__)
    patched_any = False

    for module_path, class_name in [
        ("chromadb.api.fastapi", "FastAPI"),
        ("chromadb.api.segment", "SegmentAPI"),
    ]:
        try:
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)

            if getattr(cls._query, "_chromadb_query_patched", False):
                continue  # 이미 패치됨

            original_query = cls._query

            def _make_fixed(orig):
                def _fixed_query(self, *args, **kwargs):
                    # chromadb 0.4.x에서 Collection.query()는 _client._query()를
                    # 항상 keyword argument로만 호출하므로 kwargs로 체크.
                    # chromadb 업그레이드 후 positional 방식으로 변경되면
                    # kwargs 체크가 무효화되므로 업그레이드 시 재확인 필요.
                    if kwargs.get("where") == {}:
                        kwargs["where"] = None
                    if kwargs.get("where_document") == {}:
                        kwargs["where_document"] = None
                    return orig(self, *args, **kwargs)

                _fixed_query._chromadb_query_patched = True
                return _fixed_query

            cls._query = _make_fixed(original_query)
            patched_any = True
            _patch_logger.info(
                "✅ ChromaDB %s._query 패치 완료 (where_document={} 버그 수정)",
                class_name,
            )
        except (ImportError, AttributeError) as e:
            _patch_logger.debug("ChromaDB %s 패치 스킵: %s", class_name, e)

    if not patched_any:
        _patch_logger.warning(
            "ChromaDB query 패치 대상 클래스를 찾지 못했습니다 — 버그가 잔존할 수 있음"
        )
