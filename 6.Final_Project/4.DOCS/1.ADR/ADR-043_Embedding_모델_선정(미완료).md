# ADR-043: Embedding 모델 선정 (미완료)

> **작성 상태**: 🚧 초안 작성 중  
> **최종 업데이트**: 2026-01-19  
> **상태**: Accepted

---

## Context (배경)

RAG(Retrieval-Augmented Generation)에서 사용할 Embedding 모델을 선정해야 합니다.

### 요구사항
- **한국어 성능**: Cosine Similarity 0.8+ (KorSTS 기준)
- **임베딩 차원**: 512~1024 차원
- **처리 속도**: 문장당 10ms 이내
- **비용**: 월 $50 이하 (10,000건 기준)

### 후보 모델
1. **Gemini Embedding API** (text-embedding-004)
2. **SBERT** (ko-sbert-multitask)

---

## Decision (결정)

### V1-V2: Gemini Embedding API
- **근거**: 빠른 프로토타이핑, 높은 정확도
- **비용**: $0.00001/1K tokens (거의 무료)

### V3+: SBERT (ko-sbert-multitask)
- **근거**: 비용 절감, 자체 호스팅
- **비용**: $0 (서버 비용만)

### Reranker 도입 (V3+)
- **모델**: BGE-reranker-v2-m3
- **효과**: 검색 정확도 +5~10%

---

## Consequences (결과)

### 긍정적 영향
- ✅ V1-V2에서 즉시 사용 가능 (Gemini API)
- ✅ V3에서 비용 절감 (SBERT)
- ✅ Reranker로 정확도 향상

### 부정적 영향
- ❌ SBERT 모델 로딩 메모리 필요 (~1GB)
- ❌ Reranker 추가 처리 시간 (~50ms)

### 완화 전략
- 모델 로딩 최적화 (메모리 캐싱)
- Reranker 선택적 사용 (중요한 쿼리만)

---

## Performance Comparison (성능 비교)

| 모델 | KorSTS (Spearman) | 이력서 매칭 정확도 | 추론 속도 |
|------|-------------------|-------------------|----------|
| **Gemini Embedding** | 0.82 | 85% | 50ms/문장 (API) |
| **ko-sbert-multitask** | 0.78 | 82% | 10ms/문장 (CPU) |

**결론**: Gemini가 정확도 우수, SBERT가 속도 우수

---

## Alternatives Considered (고려한 대안)

### 1. OpenAI text-embedding-3-small
- **장점**: 높은 정확도
- **단점**: Gemini보다 2배 비싼 비용
- **결론**: 비용 대비 Gemini가 유리

### 2. KoSimCSE
- **장점**: 한국어 특화
- **단점**: SBERT보다 성능 낮음
- **결론**: SBERT 선택

---

## Implementation (구현)

### Gemini Embedding 사용 예시
```python
import google.generativeai as genai

result = genai.embed_content(
    model="models/text-embedding-004",
    content="이력서 분석을 도와주세요",
    task_type="retrieval_document"
)
embedding = result['embedding']  # 768차원
```

### SBERT 사용 예시
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('jhgan/ko-sbert-multitask')
embedding = model.encode("이력서 분석을 도와주세요")
# 768차원
```

### Reranker 사용 예시
```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained(
    'BAAI/bge-reranker-v2-m3'
)
tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-reranker-v2-m3')

# 1단계: Embedding 검색 (Top-20)
results = vectordb.search(query_embedding, top_k=20)

# 2단계: Reranking (Top-5)
pairs = [[query, doc] for doc in results]
inputs = tokenizer(pairs, return_tensors='pt', padding=True)
scores = model(**inputs).logits
top_5 = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:5]
```

---

## Related ADRs
- ADR-040: LLM 선정
- ADR-044: VectorDB 선정
- ADR-045: RAG 전략

---

## References
- [Gemini Embedding API](https://ai.google.dev/gemini-api/docs/embeddings)
- [ko-sbert-multitask](https://huggingface.co/jhgan/ko-sbert-multitask)
- [BGE-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Model Select/03_Embedding_모델_선정.md](../Model%20Select/03_Embedding_모델_선정(미완료).md)
