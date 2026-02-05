"""
Abstract Base Class for Session Stores.

Defines the interface for all session store implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class SessionData:
    """Session data structure."""

    session_id: str
    data: dict[str, Any]
    created_at: datetime
    updated_at: datetime | None = None
    expires_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "session_id": self.session_id,
            "data": self.data,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionData":
        """Create from dictionary."""
        return cls(
            session_id=data["session_id"],
            data=data["data"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else None
            ),
            expires_at=(
                datetime.fromisoformat(data["expires_at"])
                if data.get("expires_at")
                else None
            ),
            metadata=data.get("metadata", {}),
        )


class BaseSessionStore(ABC):
    """Abstract base class for session stores.

    All session store implementations must inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    async def get(self, key: str) -> dict[str, Any] | None:
        """Get session data by key.

        Args:
            key: Session key.

        Returns:
            Session data dict if found, None otherwise.
        """
        pass

    @abstractmethod
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
            ttl: Time-to-live in seconds (None for no expiration).
        """
        pass

    @abstractmethod
    async def delete(self, key: str) -> bool:
        """Delete session data.

        Args:
            key: Session key.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    async def exists(self, key: str) -> bool:
        """Check if a session exists.

        Args:
            key: Session key.

        Returns:
            True if exists, False otherwise.
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> list[str]:
        """List session keys matching a pattern.

        Args:
            pattern: Pattern to match (supports * wildcard).

        Returns:
            List of matching keys.
        """
        pass

    @abstractmethod
    async def clear_all(self) -> int:
        """Clear all sessions.

        Returns:
            Number of sessions cleared.
        """
        pass

    async def health_check(self) -> bool:
        """Check if the store is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        try:
            test_key = "_health_check_"
            await self.set(test_key, {"test": True}, ttl=10)
            result = await self.get(test_key)
            await self.delete(test_key)
            return result is not None
        except Exception:
            return False

    @property
    @abstractmethod
    def store_name(self) -> str:
        """Get the store name."""
        pass
