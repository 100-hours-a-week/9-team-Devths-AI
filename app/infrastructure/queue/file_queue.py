"""
File-based Task Queue Implementation.

Implements the BaseTaskQueue interface with file-based storage.
Used for development and testing environments.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

from .base import BaseTaskQueue, TaskData, TaskStatus

logger = logging.getLogger(__name__)


class FileTaskQueue(BaseTaskQueue):
    """File-based task queue implementation.

    Note: This queue is not suitable for production.
    Use Celery with Redis for production environments.
    """

    def __init__(
        self,
        storage_dir: str = "/tmp/ai_tasks",
    ):
        """Initialize file-based task queue.

        Args:
            storage_dir: Directory to store task files.
        """
        self._storage_dir = Path(storage_dir)
        self._storage_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"FileTaskQueue initialized at {storage_dir}")

    @property
    def queue_name(self) -> str:
        """Get the queue name."""
        return "file"

    def _get_task_path(self, task_id: str) -> Path:
        """Get path to task file.

        Args:
            task_id: Task ID.

        Returns:
            Path to task file.
        """
        return self._storage_dir / f"{task_id}.json"

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
            task_type: Type of task.
            task_func: Async function to execute.
            request_data: Request data to pass to the function.
            **kwargs: Additional parameters.

        Returns:
            The task ID.
        """
        # Create initial task data
        task_data = TaskData(
            task_id=task_id,
            task_type=task_type,
            status=TaskStatus.PROCESSING,
            created_at=datetime.now(),
            progress=0,
            message="작업을 시작합니다...",
            request=request_data,
        )

        # Save initial state
        await self._save_task(task_data)

        # Run task in background
        asyncio.create_task(self._run_task(task_id, task_func, request_data, kwargs))

        return task_id

    async def _run_task(
        self,
        task_id: str,
        task_func: Callable,
        request_data: dict[str, Any],
        kwargs: dict[str, Any],
    ) -> None:
        """Run task function and update status.

        Args:
            task_id: Task ID.
            task_func: Function to execute.
            request_data: Request data.
            kwargs: Additional arguments.
        """
        try:
            # Execute task function
            result = await task_func(request_data, **kwargs)

            # Update with success
            await self.update_status(
                task_id=task_id,
                status=TaskStatus.COMPLETED,
                progress=100,
                message="작업이 완료되었습니다.",
                result=result,
            )
        except Exception as e:
            logger.error(f"Task {task_id} failed: {e}")
            await self.update_status(
                task_id=task_id,
                status=TaskStatus.FAILED,
                message="작업이 실패했습니다.",
                error={"message": str(e), "type": type(e).__name__},
            )

    async def _save_task(self, task_data: TaskData) -> None:
        """Save task data to file.

        Args:
            task_data: Task data to save.
        """
        path = self._get_task_path(task_data.task_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(task_data.to_dict(), f, ensure_ascii=False, indent=2)

    async def _load_task(self, task_id: str) -> TaskData | None:
        """Load task data from file.

        Args:
            task_id: Task ID.

        Returns:
            TaskData if found, None otherwise.
        """
        path = self._get_task_path(task_id)
        if not path.exists():
            return None

        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            return TaskData.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading task {task_id}: {e}")
            return None

    async def get_status(self, task_id: str) -> TaskData | None:
        """Get the status of a task.

        Args:
            task_id: Task ID to check.

        Returns:
            TaskData if found, None otherwise.
        """
        return await self._load_task(task_id)

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
            progress: Progress percentage.
            message: Status message.
            result: Task result.
            error: Error details.

        Returns:
            True if updated, False if not found.
        """
        task_data = await self._load_task(task_id)
        if task_data is None:
            return False

        task_data.status = status
        task_data.updated_at = datetime.now()

        if progress is not None:
            task_data.progress = progress
        if message is not None:
            task_data.message = message
        if result is not None:
            task_data.result = result
        if error is not None:
            task_data.error = error

        await self._save_task(task_data)
        return True

    async def delete(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: Task ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        path = self._get_task_path(task_id)
        if not path.exists():
            return False

        path.unlink()
        logger.debug(f"Deleted task {task_id}")
        return True

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
            limit: Maximum number of tasks.

        Returns:
            List of TaskData objects.
        """
        tasks = []

        for path in self._storage_dir.glob("*.json"):
            task_id = path.stem
            task_data = await self._load_task(task_id)

            if task_data is None:
                continue

            # Apply filters
            if task_type and task_data.task_type != task_type:
                continue
            if status and task_data.status != status:
                continue

            tasks.append(task_data)

            if len(tasks) >= limit:
                break

        # Sort by creation time (newest first)
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks

    async def cleanup_old_tasks(
        self,
        max_age_hours: int = 24,
    ) -> int:
        """Clean up old completed/failed tasks.

        Args:
            max_age_hours: Maximum age in hours.

        Returns:
            Number of tasks deleted.
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        deleted = 0

        for path in self._storage_dir.glob("*.json"):
            task_id = path.stem
            task_data = await self._load_task(task_id)

            if task_data is None:
                continue

            # Only delete completed/failed tasks older than cutoff
            if task_data.status in (TaskStatus.COMPLETED, TaskStatus.FAILED):
                if task_data.created_at < cutoff:
                    await self.delete(task_id)
                    deleted += 1

        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old tasks")

        return deleted
