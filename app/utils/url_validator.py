"""
URL 검증 유틸리티 - SSRF (Server-Side Request Forgery) 방어

허용된 도메인/프로토콜만 접근을 허용하여 내부 네트워크 접근을 방지합니다.
"""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

from app.utils.log_sanitizer import sanitize_log_input

logger = logging.getLogger(__name__)

# 허용된 프로토콜
ALLOWED_SCHEMES = {"http", "https", "data"}

# 허용된 도메인 패턴 (S3, CloudFront 등)
ALLOWED_DOMAIN_PATTERNS = [
    ".s3.amazonaws.com",
    ".s3.ap-northeast-2.amazonaws.com",
    ".cloudfront.net",
    "storage.googleapis.com",
    ".blob.core.windows.net",
]

# 명시적으로 허용된 도메인
ALLOWED_DOMAINS = {
    "localhost",  # 로컬 개발용 (프로덕션에서는 제거 권장)
}


def is_private_ip(ip_str: str) -> bool:
    """
    IP 주소가 사설 IP인지 확인합니다.

    Args:
        ip_str: IP 주소 문자열

    Returns:
        사설 IP이면 True
    """
    try:
        ip = ipaddress.ip_address(ip_str)
        return (
            ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_reserved
        )
    except ValueError:
        return False


def is_allowed_host(hostname: str) -> bool:
    """
    호스트가 허용된 도메인인지 확인합니다.

    Args:
        hostname: 호스트명

    Returns:
        허용된 호스트이면 True
    """
    if not hostname:
        return False

    hostname_lower = hostname.lower()

    # 명시적 허용 도메인 체크
    if hostname_lower in ALLOWED_DOMAINS:
        return True

    # 허용된 도메인 패턴 체크
    for pattern in ALLOWED_DOMAIN_PATTERNS:
        if hostname_lower.endswith(pattern):
            return True

    # IP 주소인 경우 사설 IP 차단
    try:
        # DNS 조회하여 실제 IP 확인
        ip_addresses = socket.gethostbyname_ex(hostname)[2]
        for ip in ip_addresses:
            if is_private_ip(ip):
                safe_hostname = sanitize_log_input(hostname)
                logger.warning("[URLValidator] 사설 IP 접근 시도 차단: %s -> %s", safe_hostname, ip)
                return False
        return True
    except socket.gaierror:
        # DNS 조회 실패 시 차단
        safe_hostname = sanitize_log_input(hostname)
        logger.warning("[URLValidator] DNS 조회 실패: %s", safe_hostname)
        return False


def validate_url(url: str) -> tuple[bool, str]:
    """
    URL이 안전한지 검증합니다 (SSRF 방어).

    Args:
        url: 검증할 URL

    Returns:
        (유효 여부, 에러 메시지)
    """
    if not url:
        return False, "URL이 비어있습니다"

    # data: URL은 항상 허용 (base64 인코딩된 데이터)
    if url.startswith("data:"):
        return True, ""

    try:
        parsed = urlparse(url)
    except Exception as e:
        return False, f"URL 파싱 실패: {e}"

    # 프로토콜 검증
    if parsed.scheme not in ALLOWED_SCHEMES:
        return False, f"허용되지 않은 프로토콜: {parsed.scheme}"

    # 호스트 검증
    hostname = parsed.hostname
    if not hostname:
        return False, "호스트명이 없습니다"

    # 허용된 호스트인지 확인
    if not is_allowed_host(hostname):
        return False, f"허용되지 않은 호스트: {hostname}"

    return True, ""


def validate_url_or_raise(url: str) -> None:
    """
    URL을 검증하고, 유효하지 않으면 예외를 발생시킵니다.

    Args:
        url: 검증할 URL

    Raises:
        ValueError: URL이 유효하지 않은 경우
    """
    is_valid, error_msg = validate_url(url)
    if not is_valid:
        raise ValueError(f"SSRF 방어: {error_msg}")
