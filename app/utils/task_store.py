"""
간단한 파일 기반 Task 저장소

개발 환경에서 uvicorn --reload 모드에서도 task가 유지되도록 함
프로덕션에서는 Redis 등으로 교체 필요
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class FileTaskStore:
    """파일 기반 Task 저장소"""

    def __init__(self, storage_dir: str = "/tmp/masking_tasks"):
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


# 전역 인스턴스
_task_store = None


def get_task_store() -> FileTaskStore:
    """Task Store 싱글톤"""
    global _task_store
    if _task_store is None:
        _task_store = FileTaskStore()
    return _task_store
