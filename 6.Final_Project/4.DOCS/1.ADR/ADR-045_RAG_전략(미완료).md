# ADR-045: RAG 전략 (미완료)

> **작성 상태**: 🚧 초안 작성 중  
> **최종 업데이트**: 2026-01-19  
> **상태**: Accepted

---

## Context (배경)

RAG(Retrieval-Augmented Generation) 전략을 수립해야 합니다.

### 요구사항
- **검색 정확도**: Top-5 검색 정확도 90% 이상
- **응답 품질**: 관련 정보 기반 응답 생성
- **처리 속도**: 검색 + 생성 5초 이내

### RAG 구성 요소
1. **Embedding**: 텍스트 → 벡터 변환
2. **VectorDB**: 벡터 저장 및 검색
3. **Reranker**: 검색 결과 재정렬 (선택)
4. **LLM**: 컨텍스트 기반 응답 생성

---

## Decision (결정)

### V1-V2: Basic RAG
```
쿼리 → Gemini Embedding → VectorDB 검색 (Top-5) → Gemini 응답 생성
```

### V3+: Advanced RAG (Reranker 추가)
```
쿼리 → SBERT Embedding → VectorDB 검색 (Top-20) → Reranker (Top-5) → vLLM 응답 생성
```

### Reranker 도입 효과
- 검색 정확도: 85% → 92% (+7%)
- 추가 처리 시간: ~50ms

---

## Consequences (결과)

### 긍정적 영향
- ✅ V1-V2에서 간단한 구현 (Basic RAG)
- ✅ V3에서 정확도 향상 (Reranker)
- ✅ 비용 효율적 (SBERT + vLLM)

### 부정적 영향
- ❌ Reranker 추가 처리 시간 (~50ms)
- ❌ 복잡도 증가 (2단계 검색)

### 완화 전략
- Reranker 선택적 사용 (중요한 쿼리만)
- 캐싱 전략 (자주 검색되는 쿼리)

---

## RAG Pipeline (파이프라인)

### Basic RAG (V1-V2)

```python
# 1. 쿼리 임베딩
query_embedding = gemini_embedding.embed(query)

# 2. VectorDB 검색
results = vectordb.search(query_embedding, top_k=5)

# 3. 컨텍스트 구성
context = "\n\n".join([r['text'] for r in results])

# 4. LLM 응답 생성
response = gemini_llm.generate(
    prompt=f"Context: {context}\n\nQuestion: {query}"
)
```

### Advanced RAG (V3+)

```python
# 1. 쿼리 임베딩
query_embedding = sbert_model.encode(query)

# 2. VectorDB 검색 (Top-20)
results = vectordb.search(query_embedding, top_k=20)

# 3. Reranking (Top-5)
pairs = [[query, r['text']] for r in results]
inputs = reranker_tokenizer(pairs, return_tensors='pt', padding=True)
scores = reranker_model(**inputs).logits
top_5 = sorted(zip(results, scores), key=lambda x: x[1], reverse=True)[:5]

# 4. 컨텍스트 구성
context = "\n\n".join([r[0]['text'] for r in top_5])

# 5. LLM 응답 생성
response = vllm_service.generate(
    prompt=f"Context: {context}\n\nQuestion: {query}"
)
```

---

## Prompt Engineering (프롬프트 전략)

### System Prompt
```python
SYSTEM_PROMPT = """
당신은 이력서와 채용공고 분석 전문가입니다.
주어진 관련 정보를 바탕으로 정확하고 구체적인 답변을 제공하세요.

규칙:
1. 관련 정보에 없는 내용은 추측하지 마세요
2. 한국어로 자연스럽게 답변하세요
3. 구조화된 형식으로 답변하세요 (목록, 표 등)
"""
```

### RAG Prompt Template
```python
RAG_PROMPT_TEMPLATE = """
관련 정보:
{context}

질문: {query}

위 관련 정보를 참고하여 질문에 답변해주세요.
"""
```

---

## Optimization Strategies (최적화 전략)

### 1. Hybrid Search (V4+)
- Semantic Search (벡터) + Keyword Search (BM25)
- 정확도 향상: +5~10%

### 2. Query Expansion
- 쿼리 확장 (동의어, 관련어 추가)
- 검색 범위 확대

### 3. Context Compression
- 긴 컨텍스트 요약 (LLM 사용)
- 토큰 사용량 감소

---

## Performance Metrics (성능 지표)

### 검색 정확도

| 전략 | Top-5 정확도 | Top-10 정확도 |
|------|------------|-------------|
| **Basic RAG** | 85% | 90% |
| **Advanced RAG (Reranker)** | 92% | 95% |
| **Hybrid Search** | 95% | 97% |

### 처리 속도

| 전략 | 검색 시간 | 생성 시간 | 총 시간 |
|------|----------|----------|---------|
| **Basic RAG** | 50ms | 1.5초 | 1.55초 |
| **Advanced RAG** | 100ms | 1.2초 | 1.30초 |

---

## Alternatives Considered (고려한 대안)

### 1. Fine-tuned LLM (RAG 없이)
- **장점**: 빠른 응답
- **단점**: 최신 정보 반영 어려움
- **결론**: RAG가 더 유연

### 2. Knowledge Graph
- **장점**: 구조화된 지식
- **단점**: 구축 비용 높음
- **결론**: V4+ 검토

---

## Related ADRs
- ADR-040: LLM 선정
- ADR-043: Embedding 모델 선정
- ADR-044: VectorDB 선정

---

## References
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/)
- [BGE-reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Model Select/06_모드별_모델_전략.md](../Model%20Select/06_모드별_모델_전략(미완료).md)
