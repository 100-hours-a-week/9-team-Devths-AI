from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict
import sys
import os

# FASTAPI_Project_model을 sys.path에 추가
current_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models.sentiment import SentimentPrediction, get_default_model


class SentimentAnalysisService:
    """단일 모델에 대한 추론 로직을 감싸는 서비스 계층."""

    def __init__(self):
        self._model = get_default_model()

    def predict(self, text: str) -> Dict[str, Any]:
        prediction: SentimentPrediction = self._model.predict(text)
        payload = asdict(prediction)
        payload["top_tokens"] = [
            {"token": token, "impact": impact}
            for token, impact in prediction.top_tokens
        ]
        return payload


service = SentimentAnalysisService()


def get_sentiment_service() -> SentimentAnalysisService:
    return service


