"""
Task 저장소 — FileTaskStore (파일 기반) + RedisTaskStore (Redis 기반)

모든 비동기 작업(text_extract, masking 등)을 통합 관리.
- FileTaskStore: 순수 로컬 환경(Redis 없음) fallback용
- RedisTaskStore: Redis가 있는 모든 환경(dev/nonprod/prod) 기본 사용 (ADR-130)
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class FileTaskStore:
    """통합 파일 기반 Task 저장소"""

    def __init__(self, storage_dir: str = "/tmp/ai_tasks"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def _get_task_file(self, task_id: str) -> Path:
        """Task ID에 해당하는 파일 경로"""
        return self.storage_dir / f"{task_id}.json"

    def save(self, task_id: str, data: dict[str, Any]) -> None:
        """Task 저장"""
        # datetime 객체를 문자열로 변환
        serializable_data = data.copy()
        if "created_at" in serializable_data and isinstance(
            serializable_data["created_at"], datetime
        ):
            serializable_data["created_at"] = serializable_data["created_at"].isoformat()

        task_file = self._get_task_file(task_id)
        with open(task_file, "w", encoding="utf-8") as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)

    def get(self, task_id: str) -> dict[str, Any] | None:
        """Task 조회"""
        task_file = self._get_task_file(task_id)
        if not task_file.exists():
            return None

        with open(task_file, encoding="utf-8") as f:
            data = json.load(f)

        # ISO 형식 문자열을 datetime으로 변환
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])

        return data

    def exists(self, task_id: str) -> bool:
        """Task 존재 여부"""
        return self._get_task_file(task_id).exists()

    def delete(self, task_id: str) -> None:
        """Task 삭제"""
        task_file = self._get_task_file(task_id)
        if task_file.exists():
            task_file.unlink()

    def list_all(self) -> list[str]:
        """모든 Task ID 목록"""
        return [f.stem for f in self.storage_dir.glob("*.json")]

    def cleanup_old_tasks(self, max_age_hours: int = 24) -> int:
        """오래된 Task 정리"""
        from datetime import timedelta

        now = datetime.now()
        deleted_count = 0

        for task_file in self.storage_dir.glob("*.json"):
            try:
                with open(task_file) as f:
                    data = json.load(f)

                created_at = datetime.fromisoformat(data.get("created_at", now.isoformat()))
                if now - created_at > timedelta(hours=max_age_hours):
                    task_file.unlink()
                    deleted_count += 1
            except Exception:
                # 손상된 파일은 삭제
                task_file.unlink()
                deleted_count += 1

        return deleted_count


class RedisTaskStore:
    """Redis 기반 Task 저장소 — FileTaskStore 드롭인 교체 (ADR-130).

    FileTaskStore와 동일한 save() / get() / delete() / exists() 인터페이스를 제공하여
    호출 측 코드(text_extract.py, masking.py, task.py) 변경 없이 교체 가능.

    - Redis DB 0 사용, "legacy_task:" prefix로 세션 키와 네임스페이스 분리
    - TTL 24시간(86400초) 자동 만료 → /tmp 파일 누적 문제 해소
    - k8s 분리 Deployment 환경에서 FastAPI·Celery Worker 간 상태 공유 지원
    """

    def __init__(self, redis_url: str, ttl: int = 86400, prefix: str = "legacy_task:"):
        import redis as _redis

        self._redis = _redis.from_url(redis_url, decode_responses=True)
        self._ttl = ttl
        self._prefix = prefix

    def _key(self, task_id: str) -> str:
        return f"{self._prefix}{task_id}"

    def save(self, task_id: str, data: dict[str, Any]) -> None:
        """Task 저장 (TTL 포함)"""
        serializable = data.copy()
        if "created_at" in serializable and isinstance(serializable["created_at"], datetime):
            serializable["created_at"] = serializable["created_at"].isoformat()
        self._redis.setex(
            self._key(task_id),
            self._ttl,
            json.dumps(serializable, ensure_ascii=False),
        )

    def get(self, task_id: str) -> dict[str, Any] | None:
        """Task 조회"""
        raw = self._redis.get(self._key(task_id))
        if raw is None:
            return None
        data = json.loads(raw)
        if "created_at" in data and isinstance(data["created_at"], str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return data

    def exists(self, task_id: str) -> bool:
        """Task 존재 여부"""
        return bool(self._redis.exists(self._key(task_id)))

    def delete(self, task_id: str) -> None:
        """Task 삭제"""
        self._redis.delete(self._key(task_id))


# 전역 인스턴스
_task_store = None


def get_task_store() -> FileTaskStore:
    """Task Store 싱글톤 (FileTaskStore — 레거시 호환용)"""
    global _task_store
    if _task_store is None:
        _task_store = FileTaskStore()
    return _task_store
