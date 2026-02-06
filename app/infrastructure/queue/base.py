"""
Abstract Base Class for Task Queues.

Defines the interface for all task queue implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    """Task status values."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TaskData:
    """Task data structure."""

    task_id: str
    task_type: str
    status: TaskStatus
    created_at: datetime
    updated_at: datetime | None = None
    progress: int = 0
    message: str | None = None
    request: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "progress": self.progress,
            "message": self.message,
            "request": self.request,
            "result": self.result,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskData":
        """Create from dictionary."""
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            status=TaskStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=(
                datetime.fromisoformat(data["updated_at"])
                if data.get("updated_at")
                else None
            ),
            progress=data.get("progress", 0),
            message=data.get("message"),
            request=data.get("request", {}),
            result=data.get("result"),
            error=data.get("error"),
        )


class BaseTaskQueue(ABC):
    """Abstract base class for task queues.

    All task queue implementations must inherit from this class
    and implement the required methods.
    """

    @abstractmethod
    async def enqueue(
        self,
        task_id: str,
        task_type: str,
        task_func: Callable,
        request_data: dict[str, Any],
        **kwargs: Any,
    ) -> str:
        """Enqueue a task for processing.

        Args:
            task_id: Unique task identifier.
            task_type: Type of task (e.g., 'text_extract', 'masking').
            task_func: Async function to execute.
            request_data: Request data to pass to the function.
            **kwargs: Additional parameters for the task.

        Returns:
            The task ID.
        """
        pass

    @abstractmethod
    async def get_status(self, task_id: str) -> TaskData | None:
        """Get the status of a task.

        Args:
            task_id: Task ID to check.

        Returns:
            TaskData if found, None otherwise.
        """
        pass

    @abstractmethod
    async def update_status(
        self,
        task_id: str,
        status: TaskStatus,
        progress: int | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
    ) -> bool:
        """Update the status of a task.

        Args:
            task_id: Task ID to update.
            status: New task status.
            progress: Progress percentage (0-100).
            message: Status message.
            result: Task result (if completed).
            error: Error details (if failed).

        Returns:
            True if updated, False if not found.
        """
        pass

    @abstractmethod
    async def delete(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: Task ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        pass

    @abstractmethod
    async def list_tasks(
        self,
        task_type: str | None = None,
        status: TaskStatus | None = None,
        limit: int = 100,
    ) -> list[TaskData]:
        """List tasks with optional filters.

        Args:
            task_type: Filter by task type.
            status: Filter by status.
            limit: Maximum number of tasks to return.

        Returns:
            List of TaskData objects.
        """
        pass

    @abstractmethod
    async def cleanup_old_tasks(
        self,
        max_age_hours: int = 24,
    ) -> int:
        """Clean up old completed/failed tasks.

        Args:
            max_age_hours: Maximum age in hours for tasks to keep.

        Returns:
            Number of tasks deleted.
        """
        pass

    async def health_check(self) -> bool:
        """Check if the queue is healthy.

        Returns:
            True if healthy, False otherwise.
        """
        return True

    @property
    @abstractmethod
    def queue_name(self) -> str:
        """Get the queue name."""
        pass
