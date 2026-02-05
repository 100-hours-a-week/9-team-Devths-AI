"""
Redis Session Store Implementation.

Implements the BaseSessionStore interface with Redis backend.
Used for production environments with distributed systems.
"""

import json
import logging
from typing import Any

from .base import BaseSessionStore

logger = logging.getLogger(__name__)


class RedisSessionStore(BaseSessionStore):
    """Redis session store implementation.

    Provides persistent, distributed session storage using Redis.
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379/0",
        default_ttl: int = 3600,
        key_prefix: str = "session:",
    ):
        """Initialize Redis session store.

        Args:
            redis_url: Redis connection URL.
            default_ttl: Default TTL in seconds (1 hour).
            key_prefix: Prefix for all session keys.
        """
        try:
            import redis.asyncio as redis
        except ImportError:
            raise ImportError("redis package is required. Install with: pip install redis")

        self._redis = redis.from_url(redis_url, decode_responses=True)
        self._default_ttl = default_ttl
        self._key_prefix = key_prefix

        logger.info(f"RedisSessionStore initialized with URL: {redis_url}")

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
        """
        try:
            data = await self._redis.get(self._make_key(key))
            if data is None:
                return None
            return json.loads(data)
        except Exception as e:
            logger.error(f"Error getting session {key}: {e}")
            return None

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
            logger.error(f"Error setting session {key}: {e}")
            raise

    async def delete(self, key: str) -> bool:
        """Delete session data.

        Args:
            key: Session key.

        Returns:
            True if deleted, False if not found.
        """
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
                key_without_prefix = key[len(self._key_prefix):]
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
