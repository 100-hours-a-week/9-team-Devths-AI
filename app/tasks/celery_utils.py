"""
Celery Utility Functions — ADR-102 Fallback Support.

Celery 브로커(Redis) 연결 상태를 확인하고,
연결 불가 시 asyncio.create_task()로 fallback할 수 있도록 지원.
"""

import logging
import time

logger = logging.getLogger(__name__)

# 캐시 TTL (초)
_CACHE_TTL = 30
_last_check_time: float = 0
_last_check_result: bool = False


def is_celery_available() -> bool:
    """Celery 브로커(Redis) 연결 상태 확인.

    Redis ping 테스트로 Celery 브로커 연결 가능 여부를 확인합니다.
    결과는 30초간 캐싱되어 매 요청마다 ping하지 않습니다.

    Returns:
        bool: Celery 사용 가능 여부
    """
    global _last_check_time, _last_check_result

    current_time = time.time()

    # 캐시된 결과가 유효하면 반환
    if current_time - _last_check_time < _CACHE_TTL:
        return _last_check_result

    # Redis 연결 테스트
    try:
        from app.config.settings import get_settings

        settings = get_settings()
        broker_url = settings.celery_broker_url

        if not broker_url:
            logger.warning("[CeleryUtils] CELERY_BROKER_URL이 설정되지 않음 → Celery 비활성화")
            _last_check_result = False
            _last_check_time = current_time
            return False

        # Redis 연결 테스트
        import redis

        # redis://host:port/db 형식 파싱
        r = redis.from_url(broker_url, socket_connect_timeout=2, socket_timeout=2)
        r.ping()

        logger.debug("[CeleryUtils] Redis 브로커 연결 성공")
        _last_check_result = True
        _last_check_time = current_time
        return True

    except ImportError:
        logger.warning("[CeleryUtils] redis 패키지 미설치 → Celery 비활성화")
        _last_check_result = False
        _last_check_time = current_time
        return False

    except Exception as e:
        logger.warning("[CeleryUtils] Redis 브로커 연결 실패: %s → asyncio fallback", e)
        _last_check_result = False
        _last_check_time = current_time
        return False


def reset_celery_cache():
    """Celery 연결 상태 캐시 초기화 (테스트용)."""
    global _last_check_time, _last_check_result
    _last_check_time = 0
    _last_check_result = False
