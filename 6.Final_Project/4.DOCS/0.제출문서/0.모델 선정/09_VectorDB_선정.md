# VectorDB 선정

> **작성 상태**: ✅ 완료
> **최종 업데이트**: 2026-02-08

**관련 문서** (같은 제출문서 폴더)
- **\[AI] 09_VectorDB_설계.md** — ChromaDB Collection 구조, A안(컬렉션 단일화 + 메타데이터), 상세 스키마·쿼리·생명주기
- **\[AI] 09_VectorDB_Collection_스키마.md** — Collection별 스키마·메타데이터·검색 예시·RDB 관계

---

## 📋 목차

- [1. 개요](#1-개요)
- [2. 요구사항](#2-요구사항)
- [3. 후보 비교](#3-후보-비교)
- [4. 강사님 피드백 요약](#4-강사님-피드백-요약)
- [5. 최종 선정](#5-최종-선정)
- [6. 컬렉션 설계](#6-컬렉션-설계)
- [7. 운영 고려사항](#7-운영-고려사항)
- [8. 확장 계획](#8-확장-계획)
- [9. 구현 참조](#9-구현-참조)

---

## 1. 개요

### 목적
RAG(Retrieval-Augmented Generation) 기반 서비스를 위한 VectorDB 선정

### 사용 사례
- 이력서/포트폴리오 임베딩 저장
- 채용공고 임베딩 저장
- 유사도 기반 문서 검색
- 사용자별 메타데이터 필터링

---

## 2. 요구사항

| 요구사항 | 목표 | 우선순위 |
|---------|------|---------|
| **빠른 프로토타이핑** | 6주 부트캠프 기간 내 완료 | 🔴 높음 |
| **설치 간편성** | pip install로 설치 가능 | 🔴 높음 |
| **메타데이터 필터링** | user_id, document_type 필터 | 🔴 높음 |
| **Python 친화성** | LangChain 통합 용이 | 🟡 중간 |
| **확장성** | 향후 확장 가능성 | 🟢 낮음 (V1) |

### 예상 데이터 규모 (V1)

```
사용자당 데이터:
- 이력서: 1~2개
- 채용공고: ~10개
- 면접 피드백: ~20개

예상 사용자 100명 기준:
- 총 벡터 수: ~3,000개
- 벡터 차원: 768 (Gemini gemini-embedding-001)
- 총 용량: ~10MB (소규모)
```

---

## 3. 후보 비교

### 3.1 ChromaDB ✅ V1 선정

**개요**:
- 오픈소스 임베딩 데이터베이스
- Python 네이티브, LangChain 통합 우수

**장점**:
- ✅ `pip install chromadb`로 간단 설치
- ✅ 메타데이터 필터링 내장 지원
- ✅ 로컬 파일 시스템 저장 (서버 불필요)
- ✅ LangChain과 자연스러운 통합
- ✅ 6주 개발 기간에 적합

**단점**:
- ❌ 대규모 트래픽 시 성능 보장 불확실
- ❌ 분산/확장 미지원
- ❌ 프로덕션급 기능 제한적

**적합 규모**: ~100K 벡터

---

### 3.2 FAISS-GPU

**개요**:
- Facebook AI Research에서 개발한 고성능 유사도 검색 라이브러리
- GPU 가속으로 대규모 검색에 최적화

**장점**:
- ✅ GPU 가속으로 검색 속도 매우 빠름 (100만 벡터에서 <10ms)
- ✅ 대규모 벡터 처리 가능 (~1B 벡터)
- ✅ 실험/벤치마크에 유용

**단점**:
- ❌ **메타데이터 저장 불가** → user_id, document_type 별도 DB 관리 필요
- ❌ GPU 필요
- ❌ 운영 복잡도 중간

**적합 규모**: ~1B 벡터

**현업 의견**: 실서빙 이전 실험 시 자주 사용, 프로덕션에서는 메타데이터 필터링 필요한 RAG에 부적합

---

### 3.3 Milvus

**개요**:
- 분산 벡터 데이터베이스
- 프로덕션급 확장성 제공

**장점**:
- ✅ 분산 시스템 지원
- ✅ SQL-like 메타데이터 필터링
- ✅ 프로덕션급 안정성
- ✅ GPU 가속 선택적

**단점**:
- ❌ **Docker + etcd + MinIO 구성 필요** (복잡)
- ❌ 6주 프로젝트에 과한 복잡도
- ❌ 운영 복잡도 높음

**적합 규모**: ~10B+ 벡터

---

### 3.4 pgvector (PostgreSQL Extension) 🔮 V3 검토

**개요**:
- PostgreSQL의 벡터 검색 확장
- 기존 RDB에 벡터 기능 추가

**장점**:
- ✅ **기존 PostgreSQL 인프라 활용** (리스크 적음)
- ✅ SQL과 벡터 검색 통합
- ✅ 트랜잭션, 백업 등 RDB 기능 그대로 사용
- ✅ **현업에서 많이 사용** (강사님 피드백)

**단점**:
- ❌ PostgreSQL 설정 필요
- ❌ 전용 VectorDB 대비 성능 제한

**현업 의견**: "PostgreSQL, MongoDB의 extension으로 VectorDB 쓰는 회사가 많다. 기존에 쓰던 DB 유지하면서 확장하면 리스크가 적기 때문"

---

### 3.5 비교 표

| 항목 | ChromaDB | FAISS-GPU | Milvus | pgvector |
|------|----------|-----------|--------|----------|
| **용도** | 프로토타입/소규모 | 대규모/고성능 | 분산/프로덕션 | RDB 통합 |
| **스케일** | ~100K 벡터 | ~1B 벡터 | ~10B+ 벡터 | ~10M 벡터 |
| **GPU 가속** | ❌ | ✅ | ✅ (선택) | ❌ |
| **메타데이터 필터링** | ✅ 내장 | ❌ 별도 관리 | ✅ SQL-like | ✅ SQL |
| **분산 시스템** | ❌ | ❌ | ✅ | ❌ (RDB 레플리카) |
| **설치 복잡도** | 낮음 | 중간 | 높음 | 중간 |
| **운영 복잡도** | 낮음 | 중간 | 높음 | 중간 |
| **현업 선호도** | 프로토타입 | 실험용 | 대규모 | 높음 |

---

## 4. 강사님 피드백 요약

### 핵심 피드백

> "ChromaDB를 사용해도 좋고 다른 걸 도입해도 좋다. 다만 **명분이 있어야 하고, 어떤 문제가 일어나서 왜 다른 걸로 전환했는지가 중요**하다."

> "**PostgreSQL, MongoDB의 extension으로 VectorDB 쓰는 회사가 많다**. 회사는 VectorDB를 새로 도입하는 것보다 기존에 쓰던 PostgreSQL을 유지한 채 익스텐션을 활용하면 리스크가 적기 때문."

> "단기적인 건 **Supabase** 사용 권장. PostgreSQL 기반이어서 확장성 좋음."

> "PostgreSQL 환경 구성을 해봐도 좋다. 다만 **너무 큰 리소스 투자는 안하되** 면접에서 질문이 왔을 때 알고 있다고 어필만 해도 좋음."

### 추가 Q&A

| 질문 | 답변 |
|------|------|
| VectorDB 임베딩 주기 (크론, Celery)? | **배치는 사용자 트래픽이 없을 때** 하는 것이 안정적 |
| 중복 데이터 막으려면? | "중복의 정의"가 필요. 유사도만으로 중복 체크 가능한지 확인 필요 |
| 컨테이너 환경에서 운영? | 가볍게 시작한다면 컨테이너 환경으로 시작할 수 있음 |

---

## 5. 최종 선정

### V1: ChromaDB

| 항목 | 내용 |
|------|------|
| **VectorDB** | ChromaDB 0.4.24 |
| **선정 근거** | 6주 개발 기간, 소규모 데이터, 빠른 프로토타이핑 |
| **저장 방식** | 로컬 파일 시스템 |
| **임베딩 모델** | **Gemini `gemini-embedding-001`** (768차원, 최종 선정) |

### 선정 명분 (ADR 기록용)

```markdown
## 의사결정: VectorDB로 ChromaDB 선정

### 상황 (Context)
- 6주 부트캠프 프로젝트로 빠른 프로토타이핑 필요
- 예상 사용자 100명 이하, 총 벡터 ~3,000개 (소규모)
- RAG 기반 메타데이터 필터링 필수 (user_id, document_type)

### 결정 (Decision)
V1에서 ChromaDB 선택

### 근거 (Rationale)
1. `pip install chromadb`로 즉시 사용 가능
2. 메타데이터 필터링 내장 지원
3. LangChain 통합 용이
4. 소규모 데이터에 적합 (100K 벡터 이하)

### 한계 (Limitations)
- 대규모 트래픽 시 성능 미검증
- 분산/확장 미지원

### 전환 조건 (When to Reconsider)
- 사용자 1,000명 이상 시 성능 벤치마크 필요
- 분산 환경 필요 시 pgvector 또는 Milvus 검토
```

---

## 6. 컬렉션 설계

> 상세 스키마·메타데이터·쿼리 예시는 **[AI] 09_VectorDB_설계.md**, **[AI] 09_VectorDB_Collection_스키마.md** 참고.

### 컬렉션 구조 (A안: 단일 컬렉션 + 메타데이터)

면접(인성/기술)은 **별도 컬렉션 없이** `interview_feedback` 한 컬렉션에 저장하고, 메타데이터 `interview_type`으로 구분한다.

```
VectorDB (ChromaDB)
├── resumes            # 이력서 + 포트폴리오 임베딩
├── job_postings       # 채용공고 임베딩
├── portfolios         # 포트폴리오 임베딩 (선택적)
├── analysis_results   # 분석 결과 임베딩 ("이전 피드백" RAG)
├── interview_feedback # 면접 Q&A + 평가 (interview_type: personality | technical)
└── chat_context       # 중요 대화 컨텍스트
```

### 메타데이터 스키마 (공통)

```python
{
    "user_id": "user_123",         # 사용자 ID (필터링 필수)
    "file_id": 42,                 # 원본 파일 ID
    "created_at": "2026-01-25",    # 생성 일시
    "collection_type": "resume"    # 문서 유형
}
```

### 면접 컬렉션 필터 (A안)

```python
# 기술면접만 검색
where={"user_id": user_id, "interview_type": "technical"}

# 인성면접만 검색
where={"user_id": user_id, "interview_type": "personality"}

# 인성+기술 통합 검색
where={"user_id": user_id}
```

### 쿼리 예시

```python
# user_id로 필터링하여 이력서 검색
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5,
    where={"user_id": "user_123"}
)

# 여러 조건 필터링
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=10,
    where={
        "$and": [
            {"user_id": "user_123"},
            {"collection_type": "resume"}
        ]
    }
)
```

---

## 7. 운영 고려사항

### 7.1 임베딩 주기

| 방식 | 설명 | 권장 상황 |
|------|------|----------|
| **실시간** | 문서 업로드 즉시 임베딩 | V1 (트래픽 적음) |
| **배치** | 새벽 시간대 일괄 처리 | 트래픽 많을 때 |

**현재 선택**: 실시간 (V1 트래픽 적음)
**향후 고려**: 배치 처리 (Celery + 크론)

### 7.2 중복 데이터 처리

```python
# 방법 1: file_id 기준 덮어쓰기
def upsert_document(file_id: str, embedding: list, metadata: dict):
    # 기존 문서 삭제
    collection.delete(where={"file_id": file_id})
    # 새 문서 추가
    collection.add(
        ids=[file_id],
        embeddings=[embedding],
        metadatas=[metadata]
    )

# 방법 2: 유사도 기반 중복 체크 (선택적)
def check_duplicate(embedding: list, threshold: float = 0.95):
    results = collection.query(
        query_embeddings=[embedding],
        n_results=1
    )
    if results['distances'][0][0] > threshold:
        return True  # 중복
    return False
```

### 7.3 컨테이너 배포

```yaml
# docker-compose.yml (V1 로컬 저장)
version: '3.8'
services:
  model-api:
    build: .
    volumes:
      - ./chroma_data:/app/chroma_data  # 영속성
    environment:
      - CHROMA_PERSIST_DIRECTORY=/app/chroma_data
```

---

## 8. 확장 계획

### Phase 1: V1 (현재)
- **VectorDB**: ChromaDB
- **규모**: 사용자 100명 이하
- **저장**: 로컬 파일 시스템

### Phase 2: V3 (여유 시 검토)
- **VectorDB**: pgvector (PostgreSQL Extension)
- **이유**: 기존 PostgreSQL 인프라 활용, 현업에서 많이 사용
- **대안**: MongoDB Atlas Vector Search

```python
# pgvector 사용 예시 (V3)
from pgvector.sqlalchemy import Vector

class Document(Base):
    __tablename__ = 'documents'

    id = Column(Integer, primary_key=True)
    user_id = Column(String)
    content = Column(Text)
    embedding = Column(Vector(768))  # 768차원 벡터

# 유사도 검색
results = session.query(Document).order_by(
    Document.embedding.cosine_distance(query_embedding)
).filter(Document.user_id == user_id).limit(5).all()
```

### Phase 3: 엔터프라이즈 (미래)
- **VectorDB**: Milvus 또는 Pinecone
- **조건**: 분산 환경, 고가용성 필요 시
- **인프라**: Kubernetes 기반 배포

### 마이그레이션 전략

```
ChromaDB → pgvector 마이그레이션:

1. pgvector 테이블 생성
2. ChromaDB에서 모든 벡터 + 메타데이터 추출
3. pgvector 테이블에 일괄 삽입
4. API 코드에서 VectorDB 서비스 교체
5. 테스트 후 ChromaDB 데이터 삭제
```

---

## 9. 구현 참조

### 코드 파일
- `3.model/app/services/vectordb_service.py` - VectorDB 서비스

### 현재 구현

```python
import chromadb
from chromadb.config import Settings

class VectorDBService:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path="./chroma_data",
            settings=Settings(anonymized_telemetry=False)
        )
        self.collections = {
            "resumes": self.client.get_or_create_collection("resumes"),
            "job_postings": self.client.get_or_create_collection("job_postings"),
            "portfolios": self.client.get_or_create_collection("portfolios")
        }

    def add_document(
        self,
        collection_name: str,
        doc_id: str,
        embedding: list,
        content: str,
        metadata: dict
    ):
        collection = self.collections[collection_name]
        collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[metadata]
        )

    def query(
        self,
        collection_name: str,
        query_embedding: list,
        n_results: int = 5,
        where: dict = None
    ):
        collection = self.collections[collection_name]
        return collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where
        )
```

---

## 참고 자료

### 내부 문서
- **[AI] 09_VectorDB_설계.md** — 설계 원칙(A안), 5개 Collection 상세 스키마, 임베딩(gemini-embedding-001), 활용 시나리오별 쿼리, 데이터 생명주기
- **[AI] 09_VectorDB_Collection_스키마.md** — Collection별 필드·메타데이터 스키마, 검색 쿼리 예시, RDB와의 관계

### 외부
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Milvus Documentation](https://milvus.io/docs)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Supabase Vector](https://supabase.com/docs/guides/ai)
