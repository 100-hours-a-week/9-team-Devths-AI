"""
In-Memory Session Store Implementation.

Implements the BaseSessionStore interface with in-memory storage.
Used for development and testing environments.
"""

import fnmatch
import logging
from datetime import datetime, timedelta
from typing import Any

from .base import BaseSessionStore

logger = logging.getLogger(__name__)


class InMemorySessionStore(BaseSessionStore):
    """In-memory session store implementation.

    Note: This store is not persistent and will lose all data on restart.
    Use Redis for production environments.
    """

    def __init__(self, default_ttl: int = 3600):
        """Initialize in-memory session store.

        Args:
            default_ttl: Default TTL in seconds (1 hour).
        """
        self._sessions: dict[str, dict[str, Any]] = {}
        self._expirations: dict[str, datetime] = {}
        self._default_ttl = default_ttl

        logger.info("InMemorySessionStore initialized")

    @property
    def store_name(self) -> str:
        """Get the store name."""
        return "memory"

    def _is_expired(self, key: str) -> bool:
        """Check if a session has expired.

        Args:
            key: Session key.

        Returns:
            True if expired, False otherwise.
        """
        if key not in self._expirations:
            return False
        return datetime.now() > self._expirations[key]

    def _cleanup_expired(self, key: str) -> None:
        """Remove expired session.

        Args:
            key: Session key.
        """
        if self._is_expired(key):
            self._sessions.pop(key, None)
            self._expirations.pop(key, None)

    async def get(self, key: str) -> dict[str, Any] | None:
        """Get session data by key.

        Args:
            key: Session key.

        Returns:
            Session data dict if found, None otherwise.
        """
        self._cleanup_expired(key)

        if key not in self._sessions:
            return None

        return self._sessions[key].copy()

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
        self._sessions[key] = value.copy()

        ttl_seconds = ttl if ttl is not None else self._default_ttl
        if ttl_seconds > 0:
            self._expirations[key] = datetime.now() + timedelta(seconds=ttl_seconds)
        elif key in self._expirations:
            del self._expirations[key]

        logger.debug(f"Set session {key} with TTL {ttl_seconds}s")

    async def delete(self, key: str) -> bool:
        """Delete session data.

        Args:
            key: Session key.

        Returns:
            True if deleted, False if not found.
        """
        if key not in self._sessions:
            return False

        del self._sessions[key]
        self._expirations.pop(key, None)
        logger.debug(f"Deleted session {key}")
        return True

    async def exists(self, key: str) -> bool:
        """Check if a session exists.

        Args:
            key: Session key.

        Returns:
            True if exists, False otherwise.
        """
        self._cleanup_expired(key)
        return key in self._sessions

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
        self._cleanup_expired(key)

        if key not in self._sessions:
            return False

        # Merge with existing data
        self._sessions[key].update(value)

        # Update TTL if provided
        if ttl is not None:
            if ttl > 0:
                self._expirations[key] = datetime.now() + timedelta(seconds=ttl)
            elif key in self._expirations:
                del self._expirations[key]

        logger.debug(f"Updated session {key}")
        return True

    async def list_keys(self, pattern: str = "*") -> list[str]:
        """List session keys matching a pattern.

        Args:
            pattern: Pattern to match (supports * wildcard).

        Returns:
            List of matching keys.
        """
        # Cleanup expired sessions first
        expired_keys = [k for k in self._sessions if self._is_expired(k)]
        for key in expired_keys:
            self._cleanup_expired(key)

        # Filter by pattern
        if pattern == "*":
            return list(self._sessions.keys())

        return [k for k in self._sessions.keys() if fnmatch.fnmatch(k, pattern)]

    async def clear_all(self) -> int:
        """Clear all sessions.

        Returns:
            Number of sessions cleared.
        """
        count = len(self._sessions)
        self._sessions.clear()
        self._expirations.clear()
        logger.info(f"Cleared {count} sessions")
        return count
