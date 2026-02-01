"""
면접 데이터셋 서비스
UICHEOL-HWANG/InterView_Datasets 관리 및 제공
"""

import json
import random
from pathlib import Path
from typing import Any

from app.core.config import settings


class InterviewDatasetService:
    """면접 질문/답변 데이터셋 서비스"""

    def __init__(self):
        self.data_dir = Path(settings.BASE_DIR) / "data"
        self.train_data = []
        self.valid_data = []
        self._load_datasets()

    def _load_datasets(self):
        """데이터셋 로드"""
        train_file = self.data_dir / "interview_dataset_train.json"
        valid_file = self.data_dir / "interview_dataset_valid.json"

        if train_file.exists():
            with open(train_file, "r", encoding="utf-8") as f:
                self.train_data = json.load(f)

        if valid_file.exists():
            with open(valid_file, "r", encoding="utf-8") as f:
                self.valid_data = json.load(f)

    def get_questions_by_occupation(
        self, occupation: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """직무별 면접 질문 가져오기

        Args:
            occupation: 직무 (ICT, 경영/회계/사무, 금융/보험, etc.)
            limit: 최대 질문 개수

        Returns:
            면접 질문/답변 리스트
        """
        data = self.train_data + self.valid_data

        if occupation:
            data = [item for item in data if item.get("occupation") == occupation]

        # 랜덤 샘플링
        if len(data) > limit:
            data = random.sample(data, limit)

        return data

    def get_questions_by_experience(
        self, experience: str | None = None, limit: int = 10
    ) -> list[dict[str, Any]]:
        """경력별 면접 질문 가져오기

        Args:
            experience: 경력 (NEW, CAREER)
            limit: 최대 질문 개수

        Returns:
            면접 질문/답변 리스트
        """
        data = self.train_data + self.valid_data

        if experience:
            data = [item for item in data if item.get("experience") == experience]

        # 랜덤 샘플링
        if len(data) > limit:
            data = random.sample(data, limit)

        return data

    def get_random_questions(self, limit: int = 10) -> list[dict[str, Any]]:
        """랜덤 면접 질문 가져오기

        Args:
            limit: 최대 질문 개수

        Returns:
            면접 질문/답변 리스트
        """
        data = self.train_data + self.valid_data
        return random.sample(data, min(limit, len(data)))

    def get_statistics(self) -> dict[str, Any]:
        """데이터셋 통계 정보

        Returns:
            통계 정보 딕셔너리
        """
        all_data = self.train_data + self.valid_data

        occupations = {}
        experiences = {}

        for item in all_data:
            occ = item.get("occupation", "unknown")
            exp = item.get("experience", "unknown")

            occupations[occ] = occupations.get(occ, 0) + 1
            experiences[exp] = experiences.get(exp, 0) + 1

        return {
            "total": len(all_data),
            "train": len(self.train_data),
            "valid": len(self.valid_data),
            "occupations": occupations,
            "experiences": experiences,
        }
