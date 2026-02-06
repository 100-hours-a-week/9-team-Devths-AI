"""
Celery Task Queue Implementation.

Implements the BaseTaskQueue interface with Celery backend.
Used for production environments with distributed task processing.
"""

import json
import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from typing import Any

from .base import BaseTaskQueue, TaskData, TaskStatus

logger = logging.getLogger(__name__)


class CeleryTaskQueue(BaseTaskQueue):
    """Celery task queue implementation.

    Provides distributed task processing using Celery with Redis backend.
    """

    def __init__(
        self,
        broker_url: str = "redis://localhost:6379/1",
        backend_url: str = "redis://localhost:6379/2",
        task_prefix: str = "ai_task:",
    ):
        """Initialize Celery task queue.

        Args:
            broker_url: Celery broker URL (Redis).
            backend_url: Celery result backend URL (Redis).
            task_prefix: Prefix for task keys in Redis.
        """
        try:
            import redis.asyncio as redis
            from celery import Celery
        except ImportError as err:
            raise ImportError(
                "celery and redis packages are required. " "Install with: pip install celery redis"
            ) from err

        self._celery = Celery(
            "ai_tasks",
            broker=broker_url,
            backend=backend_url,
        )

        # Configure Celery
        self._celery.conf.update(
            task_serializer="json",
            accept_content=["json"],
            result_serializer="json",
            timezone="Asia/Seoul",
            enable_utc=True,
            task_track_started=True,
            result_extended=True,
        )

        # Redis client for task metadata
        self._redis = redis.from_url(backend_url, decode_responses=True)
        self._task_prefix = task_prefix

        logger.info(f"CeleryTaskQueue initialized with broker: {broker_url}")

    @property
    def queue_name(self) -> str:
        """Get the queue name."""
        return "celery"

    def _make_key(self, task_id: str) -> str:
        """Create Redis key for task metadata.

        Args:
            task_id: Task ID.

        Returns:
            Full Redis key.
        """
        return f"{self._task_prefix}{task_id}"

    async def enqueue(
        self,
        task_id: str,
        task_type: str,
        task_func: Callable,  # noqa: ARG002
        request_data: dict[str, Any],
        **kwargs: Any,  # noqa: ARG002
    ) -> str:
        """Enqueue a task for processing.

        Note: In Celery, we store metadata in Redis and use Celery's
        task tracking for the actual execution status.

        Args:
            task_id: Unique task identifier.
            task_type: Type of task.
            task_func: Async function to execute (must be a Celery task).
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

        # Store metadata in Redis
        await self._redis.setex(
            self._make_key(task_id),
            86400,  # 24 hour TTL
            json.dumps(task_data.to_dict()),
        )

        # Note: In real implementation, task_func should be a Celery task
        # decorated with @celery.task. For now, we just store the metadata.
        # The actual task execution would be triggered by:
        # task_func.apply_async(args=[request_data], task_id=task_id, **kwargs)

        logger.info(f"Enqueued task {task_id} of type {task_type}")
        return task_id

    async def get_status(self, task_id: str) -> TaskData | None:
        """Get the status of a task.

        Args:
            task_id: Task ID to check.

        Returns:
            TaskData if found, None otherwise.
        """
        try:
            data = await self._redis.get(self._make_key(task_id))
            if data is None:
                return None
            return TaskData.from_dict(json.loads(data))
        except Exception as e:
            logger.error(f"Error getting task status {task_id}: {e}")
            return None

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
        try:
            key = self._make_key(task_id)
            data = await self._redis.get(key)

            if data is None:
                return False

            task_data = TaskData.from_dict(json.loads(data))
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

            # Get remaining TTL
            ttl = await self._redis.ttl(key)
            if ttl <= 0:
                ttl = 86400

            await self._redis.setex(key, ttl, json.dumps(task_data.to_dict()))
            return True

        except Exception as e:
            logger.error(f"Error updating task status {task_id}: {e}")
            return False

    async def delete(self, task_id: str) -> bool:
        """Delete a task.

        Args:
            task_id: Task ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        try:
            result = await self._redis.delete(self._make_key(task_id))
            if result > 0:
                logger.debug(f"Deleted task {task_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting task {task_id}: {e}")
            return False

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
        try:
            tasks = []
            pattern = f"{self._task_prefix}*"

            async for key in self._redis.scan_iter(match=pattern):
                if len(tasks) >= limit:
                    break

                data = await self._redis.get(key)
                if data is None:
                    continue

                task_data = TaskData.from_dict(json.loads(data))

                # Apply filters
                if task_type and task_data.task_type != task_type:
                    continue
                if status and task_data.status != status:
                    continue

                tasks.append(task_data)

            # Sort by creation time (newest first)
            tasks.sort(key=lambda t: t.created_at, reverse=True)
            return tasks[:limit]

        except Exception as e:
            logger.error(f"Error listing tasks: {e}")
            return []

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
        try:
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            deleted = 0
            pattern = f"{self._task_prefix}*"

            async for key in self._redis.scan_iter(match=pattern):
                data = await self._redis.get(key)
                if data is None:
                    continue

                task_data = TaskData.from_dict(json.loads(data))

                # Only delete completed/failed tasks older than cutoff
                if (
                    task_data.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)
                    and task_data.created_at < cutoff
                ):
                    await self._redis.delete(key)
                    deleted += 1

            if deleted > 0:
                logger.info(f"Cleaned up {deleted} old tasks")

            return deleted

        except Exception as e:
            logger.error(f"Error cleaning up tasks: {e}")
            return 0

    async def close(self) -> None:
        """Close connections."""
        await self._redis.close()
