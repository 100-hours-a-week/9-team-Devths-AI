"""
Task Storage - 파일 기반 작업 저장소

uvicorn reload 모드에서도 작동하도록 파일 시스템 사용
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# 작업 저장 디렉토리
TASKS_DIR = Path("/tmp/masking_tasks")
TASKS_DIR.mkdir(exist_ok=True)


def _get_task_file(task_id: str) -> Path:
    """작업 ID에 대한 파일 경로 반환"""
    return TASKS_DIR / f"{task_id}.json"


def save_task(task_id: str, task_data: Dict[str, Any]) -> None:
    """작업 데이터 저장"""
    try:
        # datetime 객체를 문자열로 변환
        task_data_copy = task_data.copy()
        if "created_at" in task_data_copy and isinstance(task_data_copy["created_at"], datetime):
            task_data_copy["created_at"] = task_data_copy["created_at"].isoformat()

        file_path = _get_task_file(task_id)
        with open(file_path, 'w') as f:
            json.dump(task_data_copy, f)
        logger.debug(f"Saved task {task_id} to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save task {task_id}: {e}")


def load_task(task_id: str) -> Optional[Dict[str, Any]]:
    """작업 데이터 로드"""
    try:
        file_path = _get_task_file(task_id)
        if not file_path.exists():
            logger.debug(f"Task {task_id} not found at {file_path}")
            return None

        with open(file_path, 'r') as f:
            task_data = json.load(f)

        # created_at을 datetime 객체로 변환
        if "created_at" in task_data and isinstance(task_data["created_at"], str):
            task_data["created_at"] = datetime.fromisoformat(task_data["created_at"])

        logger.debug(f"Loaded task {task_id} from {file_path}")
        return task_data
    except Exception as e:
        logger.error(f"Failed to load task {task_id}: {e}")
        return None


def delete_task(task_id: str) -> None:
    """작업 데이터 삭제"""
    try:
        file_path = _get_task_file(task_id)
        if file_path.exists():
            file_path.unlink()
            logger.debug(f"Deleted task {task_id}")
    except Exception as e:
        logger.error(f"Failed to delete task {task_id}: {e}")


def list_all_tasks() -> list[str]:
    """모든 작업 ID 목록 반환"""
    try:
        return [f.stem for f in TASKS_DIR.glob("*.json")]
    except Exception as e:
        logger.error(f"Failed to list tasks: {e}")
        return []
