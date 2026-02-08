# ADR-044: VectorDB 선정 (미완료)

> **작성 상태**: 🚧 초안 작성 중  
> **최종 업데이트**: 2026-01-19  
> **상태**: Accepted

---

## Context (배경)

RAG에서 사용할 VectorDB를 선정해야 합니다.

### 요구사항

| 요구사항 | V1-V2 | V3 | V4+ |
|---------|-------|----|----|
| **데이터 규모** | ~1K 벡터 | ~10K 벡터 | ~100K+ 벡터 |
| **검색 속도** | 100ms 이내 | 50ms 이내 | 10ms 이내 |
| **비용** | 무료 | $50/월 이하 | $200/월 이하 |

### 후보 VectorDB
1. **ChromaDB** (오픈소스, 로컬)
2. **Milvus** (오픈소스, 대규모)
3. **Pinecone** (관리형, SaaS)

---

## Decision (결정)

### V1-V3: ChromaDB
- **근거**: 소규모 데이터, 간단한 설치, 무료
- **비용**: $0.40/월 (스토리지만)

### V4+: Milvus (검토 중)
- **근거**: 대규모 데이터, 확장성, 프로덕션 기능
- **비용**: $154/월 (자체 호스팅)

### 마이그레이션 조건
- 데이터 규모: 50K+ 벡터
- 검색 속도: 200ms 초과
- 동시 사용자: 100+ 명

---

## Consequences (결과)

### 긍정적 영향
- ✅ V1-V3에서 빠른 개발 (ChromaDB)
- ✅ 비용 최소화 (무료)
- ✅ 마이그레이션 용이성 (Export/Import)

### 부정적 영향
- ❌ ChromaDB 대규모 데이터 제한 (~100K 벡터)
- ❌ 수평 확장 불가
- ❌ V4 마이그레이션 비용

### 완화 전략
- 마이그레이션 스크립트 사전 작성
- Milvus 로컬 테스트 환경 구축
- 성능 모니터링 (검색 속도, 메모리 사용량)

---

## Performance Comparison (성능 비교)

| VectorDB | 10K 벡터 | 100K 벡터 | 1M 벡터 | 메모리 사용량 |
|----------|---------|----------|---------|-------------|
| **ChromaDB** | 50ms | 200ms | 2000ms | 500MB |
| **Milvus** | 10ms | 30ms | 100ms | 2GB |
| **Pinecone** | 20ms | 40ms | 80ms | N/A (클라우드) |

**결론**: 소규모(10K)는 ChromaDB 충분, 대규모(100K+)는 Milvus 필요

---

## Migration Strategy (마이그레이션 전략)

### ChromaDB → Milvus

#### 1단계: 데이터 Export
```python
import chromadb
import json

client = chromadb.Client()
collection = client.get_collection("resumes")

# 모든 데이터 가져오기
all_data = collection.get(include=["embeddings", "documents", "metadatas"])

# JSON으로 저장
with open("chromadb_export.json", "w") as f:
    json.dump(all_data, f)
```

#### 2단계: 데이터 Import
```python
from pymilvus import connections, Collection
import json

# Milvus 연결
connections.connect("default", host="localhost", port="19530")
collection = Collection("resumes")

# JSON 로드
with open("chromadb_export.json", "r") as f:
    data = json.load(f)

# Milvus에 삽입
ids = list(range(len(data["embeddings"])))
collection.insert([ids, data["embeddings"], data["documents"]])
```

---

## Alternatives Considered (고려한 대안)

### 1. Pinecone
- **장점**: 완전 관리형, 자동 스케일링
- **단점**: 비용 높음 ($70/월 시작)
- **결론**: 비용 대비 Milvus가 유리

### 2. Qdrant
- **장점**: 오픈소스, 고성능
- **단점**: 커뮤니티 작음
- **결론**: Milvus가 더 성숙

---

## Related ADRs
- ADR-043: Embedding 모델 선정
- ADR-045: RAG 전략

---

## References
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Milvus Documentation](https://milvus.io/docs)
- [Model Select/05_VectorDB_선정.md](../Model%20Select/05_VectorDB_선정(미완료).md)
