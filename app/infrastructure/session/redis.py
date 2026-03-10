"""
Redis Session Store Implementation.

Implements the BaseSessionStore interface with Redis backend.
Used for production environments with distributed systems.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any

from filelock import FileLock
from filelock import Timeout as FileLockTimeout

from app.utils.log_sanitizer import sanitize_log_input

from .base import BaseSessionStore

logger = logging.getLogger(__name__)

# 서버 재시작에도 세션을 보존하기 위한 파일 기반 폴백 경로
_PERSIST_FILE = os.path.join(os.environ.get("TMPDIR", "/tmp"), "devths_interview_sessions.json")
_PERSIST_TTL = 7200  # 2시간 (초) — 파일 폴백 세션 만료 시간
# 멀티워커(Gunicorn -w 4) 프로세스 간 파일 동시 접근 방지 — threading.Lock()은 프로세스 내부만 보호
_FILE_LOCK = FileLock(f"{_PERSIST_FILE}.lock", timeout=2)


class RedisSessionStore(BaseSessionStore):
    """Redis session store implementation.

    Provides persistent, distributed session storage using Redis.
    Redis 장애 및 서버 재시작에도 세션을 보존하기 위한 2단계 폴백:
      1. in-memory fallback (_fallback dict) — 빠른 접근
      2. 파일 기반 폴백 (_PERSIST_FILE) — 서버 재시작 후에도 복원
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
        key_prefix: str = "session:",
        socket_timeout: float = 5.0,
        socket_connect_timeout: float = 5.0,
    ):
        """Initialize Redis session store.

        Args:
            redis_url: Redis connection URL.
            default_ttl: Default TTL in seconds (1 hour).
            key_prefix: Prefix for all session keys.
            socket_timeout: Socket timeout in seconds.
            socket_connect_timeout: Connection timeout in seconds.
        """
        try:
            import redis.asyncio as redis
        except ImportError as err:
            raise ImportError("redis package is required. Install with: pip install redis") from err

        self._redis = redis.from_url(
            redis_url,
            decode_responses=True,
            socket_timeout=socket_timeout,
            socket_connect_timeout=socket_connect_timeout,
        )
        self._default_ttl = default_ttl
        self._key_prefix = key_prefix
        # 1단계: in-memory 폴백 (빠른 접근, 재시작 시 초기화)
        self._fallback: dict[str, Any] = {}
        # 2단계: 파일 기반 폴백 (재시작 후에도 복원)
        self._load_file_fallback()

        logger.info(f"RedisSessionStore initialized with URL: {redis_url}")

    def _load_file_fallback(self) -> None:
        """서버 시작 시 파일에서 세션 복원 (재시작 후 세션 보존)."""
        try:
            if not os.path.exists(_PERSIST_FILE):
                return
            with _FILE_LOCK, open(_PERSIST_FILE, encoding="utf-8") as f:
                raw: dict[str, Any] = json.load(f)
            now = time.time()
            valid = {
                k: v["data"]
                for k, v in raw.items()
                if isinstance(v, dict) and now - v.get("ts", 0) < _PERSIST_TTL
            }
            self._fallback.update(valid)
            if valid:
                logger.info("📂 파일 폴백에서 세션 %d개 복원 (재시작 후 복구)", len(valid))
        except FileLockTimeout:
            logger.warning("파일 폴백 로드 실패: 락 획득 타임아웃 (워커 경합 가능성)")
        except Exception as e:
            logger.warning("파일 폴백 로드 실패 (무시): %s", e)

    def _save_file_fallback(self, key: str, value: dict[str, Any]) -> None:
        """세션을 파일에 저장 (서버 재시작 대비). 만료 항목 동시 정리."""
        try:
            with _FILE_LOCK:
                try:
                    with open(_PERSIST_FILE, encoding="utf-8") as f:
                        raw: dict[str, Any] = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    raw = {}
                # 저장 시 만료 항목 정리 — 파일 무한 증가 방지
                now = time.time()
                raw = {
                    k: v
                    for k, v in raw.items()
                    if isinstance(v, dict) and now - v.get("ts", 0) < _PERSIST_TTL
                }
                raw[key] = {"data": value, "ts": now}
                with open(_PERSIST_FILE, "w", encoding="utf-8") as f:
                    json.dump(raw, f, ensure_ascii=False)
        except FileLockTimeout:
            logger.warning("파일 폴백 저장 실패: 락 획득 타임아웃 (워커 경합 가능성)")
        except Exception as e:
            logger.warning("파일 폴백 저장 실패 (무시): %s", e)

    def _delete_file_fallback(self, key: str) -> None:
        """파일 폴백에서 세션 삭제."""
        try:
            with _FILE_LOCK:
                try:
                    with open(_PERSIST_FILE, encoding="utf-8") as f:
                        raw: dict[str, Any] = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    return
                if key in raw:
                    del raw[key]
                    with open(_PERSIST_FILE, "w", encoding="utf-8") as f:
                        json.dump(raw, f, ensure_ascii=False)
        except FileLockTimeout:
            logger.warning("파일 폴백 삭제 실패: 락 획득 타임아웃 (워커 경합 가능성)")
        except Exception as e:
            logger.warning("파일 폴백 삭제 실패 (무시): %s", e)

    @property
    def store_name(self) -> str:
        """Get the store name."""
        return "redis"

    def _make_key(self, key: str) -> str:
        """Create full Redis key with prefix.

        Args:
            key: Session key.

        Returns:
            Full Redis key.
        """
        return f"{self._key_prefix}{key}"

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get session data by key.

        Args:
            key: Session key.

        Returns:
            Session data dict if found, None otherwise.
            Redis 장애 시 in-memory → 파일 폴백 순서로 복원 시도.
        """
        safe_key = sanitize_log_input(key)
        try:
            logger.debug("Redis GET: %s", sanitize_log_input(self._make_key(key)))
            data = await self._redis.get(self._make_key(key))
            if data is None:
                logger.debug("Redis GET: %s not found", safe_key)
                # Redis에 없으면 메모리 폴백 체크
                fallback = self._fallback.get(key)
                if fallback:
                    logger.warning("Redis miss — 메모리 폴백에서 세션 복원: %s", safe_key)
                return fallback
            result = json.loads(data)
            logger.debug("Redis GET: %s found, size=%d", safe_key, len(data))
            self._fallback[key] = result  # 폴백 동기화
            return result
        except Exception as e:
            logger.error("Redis GET error for %s: %s", safe_key, type(e).__name__)
            # 1단계: 메모리 폴백
            fallback = self._fallback.get(key)
            if fallback:
                logger.warning("Redis 장애 — 메모리 폴백으로 세션 복원: %s", safe_key)
                return fallback
            # 2단계: 파일 폴백 (서버 재시작 후 메모리가 비어 있을 때) — 별도 스레드로 비블로킹
            await asyncio.to_thread(self._load_file_fallback)
            fallback = self._fallback.get(key)
            if fallback:
                logger.warning("Redis 장애 — 파일 폴백으로 세션 복원 (재시작 후): %s", safe_key)
            return fallback

    async def set(
        self,
        key: str,
        value: dict[str, Any],
        ttl: int | None = None,
    ) -> None:
        """Set session data.

        Args:
            key: Session key.
            value: Session data to store.
            ttl: Time-to-live in seconds (None for default).
        """
        # 폴백 먼저 저장 (항상 성공 — Redis 장애 및 재시작에도 세션 보존)
        self._fallback[key] = value
        # 파일 폴백 저장 — 이벤트 루프 비블로킹 (스레드 풀에서 실행)
        asyncio.get_running_loop().run_in_executor(None, self._save_file_fallback, key, value)
        try:
            ttl_seconds = ttl if ttl is not None else self._default_ttl
            data = json.dumps(value)

            if ttl_seconds > 0:
                await self._redis.setex(
                    self._make_key(key),
                    ttl_seconds,
                    data,
                )
            else:
                await self._redis.set(self._make_key(key), data)

            logger.debug(f"Set session {key} with TTL {ttl_seconds}s")
        except Exception as e:
            logger.error(f"Error setting session {key}: {e} (메모리 폴백 사용 중)")

    async def delete(self, key: str) -> bool:
        """Delete session data.

        Args:
            key: Session key.

        Returns:
            True if deleted, False if not found.
        """
        self._fallback.pop(key, None)  # 메모리 폴백에서 삭제
        # 파일 폴백 삭제 — 이벤트 루프 비블로킹 (스레드 풀에서 실행)
        asyncio.get_running_loop().run_in_executor(None, self._delete_file_fallback, key)
        try:
            result = await self._redis.delete(self._make_key(key))
            if result > 0:
                logger.debug(f"Deleted session {key}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting session {key}: {e}")
            return False

    async def exists(self, key: str) -> bool:
        """Check if a session exists.

        Args:
            key: Session key.

        Returns:
            True if exists, False otherwise.
        """
        try:
            return await self._redis.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.error(f"Error checking session existence {key}: {e}")
            return False

    async def update(
        self,
        key: str,
        value: dict[str, Any],
        ttl: int | None = None,
    ) -> bool:
        """Update existing session data.

        Args:
            key: Session key.
            value: New session data (merged with existing).
            ttl: New TTL (None to keep existing).

        Returns:
            True if updated, False if not found.
        """
        try:
            redis_key = self._make_key(key)

            # Get existing data
            existing = await self._redis.get(redis_key)
            if existing is None:
                return False

            # Merge data
            existing_data = json.loads(existing)
            existing_data.update(value)

            # Get remaining TTL if not provided
            if ttl is None:
                remaining_ttl = await self._redis.ttl(redis_key)
                ttl = remaining_ttl if remaining_ttl > 0 else self._default_ttl

            # Save updated data
            if ttl > 0:
                await self._redis.setex(redis_key, ttl, json.dumps(existing_data))
            else:
                await self._redis.set(redis_key, json.dumps(existing_data))

            logger.debug(f"Updated session {key}")
            return True
        except Exception as e:
            logger.error(f"Error updating session {key}: {e}")
            return False

    async def list_keys(self, pattern: str = "*") -> list[str]:
        """List session keys matching a pattern.

        Args:
            pattern: Pattern to match (supports * wildcard).

        Returns:
            List of matching keys (without prefix).
        """
        try:
            full_pattern = f"{self._key_prefix}{pattern}"
            keys = []

            async for key in self._redis.scan_iter(match=full_pattern):
                # Remove prefix from key
                key_without_prefix = key[len(self._key_prefix) :]
                keys.append(key_without_prefix)

            return keys
        except Exception as e:
            logger.error(f"Error listing session keys: {e}")
            return []

    async def clear_all(self) -> int:
        """Clear all sessions.

        Returns:
            Number of sessions cleared.
        """
        try:
            pattern = f"{self._key_prefix}*"
            count = 0

            async for key in self._redis.scan_iter(match=pattern):
                await self._redis.delete(key)
                count += 1

            logger.info(f"Cleared {count} sessions")
            return count
        except Exception as e:
            logger.error(f"Error clearing sessions: {e}")
            return 0

    async def close(self) -> None:
        """Close Redis connection."""
        await self._redis.close()
