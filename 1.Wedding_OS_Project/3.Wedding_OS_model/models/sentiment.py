from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple


TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


@dataclass(frozen=True)
class SentimentExample:
    text: str
    label: str


@dataclass(frozen=True)
class SentimentPrediction:
    label: str
    confidence: float
    probabilities: Dict[str, float]
    top_tokens: List[Tuple[str, float]]


class NaiveBayesSentimentModel:
    """
    아주 작은 데이터셋으로 학습한 나이브 베이즈 감성 분류기.

    - 메모리 상주형 모델이라서 FastAPI 인스턴스 시작 시 곧바로 로딩 가능
    - Laplace smoothing 적용
    """

    def __init__(self, dataset: Sequence[SentimentExample]):
        if not dataset:
            raise ValueError("dataset must not be empty")
        self._classes = sorted({example.label for example in dataset})
        if len(self._classes) < 2:
            raise ValueError("dataset must contain at least two labels")

        self._token_counts = {label: {} for label in self._classes}
        self._doc_counts = {label: 0 for label in self._classes}
        self._total_tokens = {label: 0 for label in self._classes}

        for example in dataset:
            tokens = self._tokenize(example.text)
            self._doc_counts[example.label] += 1
            for token in tokens:
                self._token_counts[example.label][token] = (
                    self._token_counts[example.label].get(token, 0) + 1
                )
                self._total_tokens[example.label] += 1

        self._doc_total = sum(self._doc_counts.values())
        self._priors = {
            label: self._doc_counts[label] / self._doc_total for label in self._classes
        }

        vocab = set()
        for label in self._classes:
            vocab.update(self._token_counts[label].keys())
        self._vocab = sorted(vocab)
        self._vocab_size = len(self._vocab)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return [token.lower() for token in TOKEN_PATTERN.findall(text)]

    def _log_likelihood(self, label: str, token: str) -> float:
        token_count = self._token_counts[label].get(token, 0)
        return math.log(
            (token_count + 1) / (self._total_tokens[label] + self._vocab_size)
        )

    def predict(self, text: str) -> SentimentPrediction:
        if not text or not text.strip():
            raise ValueError("text must not be empty")

        tokens = self._tokenize(text)
        if not tokens:
            raise ValueError("text must contain alphabetic characters")

        log_scores = {}
        token_impacts: Dict[str, float] = {}

        for label in self._classes:
            log_score = math.log(self._priors[label])
            for token in tokens:
                ll = self._log_likelihood(label, token)
                log_score += ll
                token_impacts[token] = token_impacts.get(token, 0.0) + ll
            log_scores[label] = log_score

        # softmax for probabilities
        max_log = max(log_scores.values())
        exps = {label: math.exp(score - max_log) for label, score in log_scores.items()}
        total = sum(exps.values())
        probabilities = {label: value / total for label, value in exps.items()}

        label = max(probabilities, key=probabilities.get)
        confidence = probabilities[label]

        top_tokens = sorted(
            ((token, impact) for token, impact in token_impacts.items()),
            key=lambda item: abs(item[1]),
            reverse=True,
        )[:5]

        return SentimentPrediction(
            label=label,
            confidence=confidence,
            probabilities=probabilities,
            top_tokens=top_tokens,
        )


DEFAULT_DATASET: Tuple[SentimentExample, ...] = (
    SentimentExample("I absolutely love this product, it works great", "positive"),
    SentimentExample("Fantastic quality and excellent design", "positive"),
    SentimentExample("The service was friendly and I felt valued", "positive"),
    SentimentExample("Horrible experience, I want a refund", "negative"),
    SentimentExample("This is the worst purchase I have ever made", "negative"),
    SentimentExample("Customer support ignored my problem", "negative"),
    SentimentExample("Pretty good overall, satisfied with the result", "positive"),
    SentimentExample("It broke after two days of normal use", "negative"),
    SentimentExample("Absolutely delighted with my new phone", "positive"),
    SentimentExample("Disappointed and frustrated by the delay", "negative"),
)


class SentimentModelRegistry:
    """단일톤 registry: 향후 다중 모델 지원 고려."""

    def __init__(self):
        self._model: NaiveBayesSentimentModel | None = None

    def load(self) -> NaiveBayesSentimentModel:
        if self._model is None:
            self._model = NaiveBayesSentimentModel(DEFAULT_DATASET)
        return self._model


registry = SentimentModelRegistry()


def get_default_model() -> NaiveBayesSentimentModel:
    return registry.load()


