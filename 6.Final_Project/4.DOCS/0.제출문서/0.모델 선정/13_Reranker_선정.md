# Reranker 선정

> **작성 상태**: ✅ 완료
> **최종 업데이트**: 2026-01-25
> **구현 시점**: V3 이후

---

## 📋 목차

- [1. 개요](#1-개요)
- [2. Reranker란](#2-reranker란)
- [3. 후보 모델 비교](#3-후보-모델-비교)
- [4. 구현 계획](#4-구현-계획)

---

## 1. 개요

### 목적
VectorDB 검색 결과를 재정렬하여 검색 정확도 향상

### 구현 시점
**V3 이후** (현재는 임베딩 검색만 사용)

### 예상 효과
검색 정확도 +5~10% 향상

---

## 2. Reranker란

### 2단계 검색 전략

```
[1단계: Embedding 검색]
사용자 쿼리 → 임베딩 → VectorDB 검색 → Top-20 후보

[2단계: Reranking]
Top-20 후보 → Reranker 모델 → 점수 재계산 → Top-5 최종 결과

장점:
1. 1단계: 빠른 검색 (벡터 유사도)
2. 2단계: 정밀한 관련성 평가 (Cross-Encoder)
```

### 왜 필요한가?

```
임베딩 검색의 한계:
- Bi-Encoder 방식 (쿼리와 문서를 각각 인코딩)
- 의미적 유사도만 측정
- 세부적인 관련성 판단 어려움

Reranker 장점:
- Cross-Encoder 방식 (쿼리-문서 쌍을 함께 인코딩)
- 더 정밀한 관련성 점수
- 질문-답변 맥락 이해 우수
```

---

## 3. 후보 모델 비교

### 3.1 BGE-reranker-v2-m3 ✅ 예상 선정

**개요**:
- BAAI에서 개발한 오픈소스 Reranker
- 다국어 지원 (한국어 포함)

**장점**:
- ✅ 무료 (오픈소스)
- ✅ 한국어 성능 양호
- ✅ 다양한 크기 (small, base, large)

**단점**:
- ❌ 모델 로딩 필요
- ❌ GPU 권장 (추론 속도)

**성능**:
- 검색 정확도 향상: +5~10%
- 추론 속도: ~50ms/쿼리 (GPU)

**비용**: $0 (서버 비용만)

---

### 3.2 Cohere Rerank API

**장점**:
- ✅ 높은 정확도
- ✅ 즉시 사용 가능 (API)
- ✅ 인프라 관리 불필요

**단점**:
- ❌ API 비용 ($1/1K 요청)
- ❌ 인터넷 연결 필수

**비용**: $1/1K 요청

---

### 3.3 비교 표

| 항목 | BGE-reranker | Cohere Rerank |
|------|-------------|---------------|
| **비용** | 무료 | $1/1K 요청 |
| **정확도** | 양호 | 우수 |
| **한국어** | 양호 | 우수 |
| **인프라** | GPU 필요 | API |
| **추론 속도** | 50ms (GPU) | 100ms (API) |

---

## 4. 구현 계획

### V3 구현 예정

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

class RerankerService:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            'BAAI/bge-reranker-v2-m3'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            'BAAI/bge-reranker-v2-m3'
        )

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list:
        """문서 재정렬"""
        pairs = [[query, doc] for doc in documents]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        scores = self.model(**inputs).logits.squeeze(-1)

        # 점수 기준 정렬
        sorted_indices = scores.argsort(descending=True)[:top_k]
        return [documents[i] for i in sorted_indices]
```

### RAG 파이프라인 통합

```python
async def rag_with_reranker(self, user_id: str, query: str):
    # 1단계: 임베딩 검색 (Top-20)
    candidates = await self.vectordb.search(
        query=query,
        n_results=20,
        where={"user_id": user_id}
    )

    # 2단계: Reranking (Top-5)
    reranked = self.reranker.rerank(
        query=query,
        documents=[c['content'] for c in candidates],
        top_k=5
    )

    # 3단계: LLM 응답 생성
    context = "\n\n".join(reranked)
    return await self.llm.generate(context + "\n\n" + query)
```

### 비용 분석 (V3)

```
BGE-reranker 선택 시:
- 서버 비용: GPU 서버 일부 사용
- 추가 비용: 거의 없음

Cohere Rerank 선택 시:
- 월 5,000 쿼리 × $0.001 = $5/월
```

---

## 참고 자료

- [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Cohere Rerank](https://cohere.com/rerank)
- [Two-Stage Retrieval Paper](https://arxiv.org/abs/1901.04085)
