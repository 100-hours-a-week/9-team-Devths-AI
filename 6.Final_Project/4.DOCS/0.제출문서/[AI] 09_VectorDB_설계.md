# VectorDB 설계 (통합 문서)

> ChromaDB 선정·Collection 설계·스키마·운영·확장을 **하나의 문서**로 통합했다.  
> (기존: 09_VectorDB_선정.md, 09_VectorDB_설계.md, 09_VectorDB_Collection_스키마.md)

**최종 업데이트**: 2026-02-08

---

## 📚 목차

- [Part 1. VectorDB 선정](#part-1-vectordb-선정)
- [Part 2. 설계 원칙 ](#part-2-설계-원칙-a안)
- [Part 3. Collection 구조 및 상세 스키마](#part-3-collection-구조-및-상세-스키마)
- [Part 4. 임베딩 모델](#part-4-임베딩-모델)
- [Part 5. 활용 시나리오별 쿼리](#part-5-활용-시나리오별-쿼리)
- [Part 6. 데이터 생명주기](#part-6-데이터-생명주기)
- [Part 7. 운영 고려사항](#part-7-운영-고려사항)
- [Part 8. 확장 계획](#part-8-확장-계획)
- [Part 9. 구현 참조 및 RDB 관계](#part-9-구현-참조-및-rdb-관계)
- [참고 자료](#참고-자료)

---

# Part 1. VectorDB 선정

## 1.1 개요

### 목적
RAG(Retrieval-Augmented Generation) 기반 서비스를 위한 VectorDB 선정

### 사용 사례
- 이력서/포트폴리오 임베딩 저장
- 채용공고 임베딩 저장
- 유사도 기반 문서 검색
- 사용자별 메타데이터 필터링

---

## 1.2 요구사항

| 요구사항 | 목표 | 우선순위 |
|---------|------|---------|
| **빠른 프로토타이핑** | 6주 부트캠프 기간 내 완료 | 🔴 높음 |
| **설치 간편성** | pip install로 설치 가능 | 🔴 높음 |
| **메타데이터 필터링** | user_id, document_type 필터 | 🔴 높음 |
| **Python 친화성** | LangChain 통합 용이 | 🟡 중간 |
| **확장성** | 향후 확장 가능성 | 🟢 낮음 (V1) |

### 예상 데이터 규모 (V1)

```
사용자당: 이력서 1~2개, 채용공고 ~10개, 면접 피드백 ~20개
예상 사용자 100명 기준: 총 벡터 ~3,000개, 벡터 차원 768 (Gemini gemini-embedding-001), 총 용량 ~10MB
```

---

## 1.3 후보 비교

### ChromaDB ✅ V1 선정

- **개요:** 오픈소스 임베딩 DB, Python 네이티브, LangChain 통합 우수  
- **장점:** `pip install chromadb`, 메타데이터 필터 내장, 로컬 파일 저장, 6주 개발에 적합  
- **단점:** 대규모 트래픽 미검증, 분산/확장 미지원  
- **적합 규모:** ~100K 벡터

### FAISS-GPU

- **단점:** 메타데이터 저장 불가, GPU 필요 → RAG 메타데이터 필터에 부적합

### Milvus

- **단점:** Docker+etcd+MinIO 등 구성 복잡, 6주 프로젝트에 과한 복잡도

### pgvector (PostgreSQL Extension) 🔮 V3 검토

- **장점:** 기존 PostgreSQL 활용, 현업 다수 사용 (강사님 피드백)  
- **단점:** PostgreSQL 설정 필요, 전용 VectorDB 대비 성능 제한

### 비교 표

| 항목 | ChromaDB | FAISS-GPU | Milvus | pgvector |
|------|----------|-----------|--------|----------|
| **용도** | 프로토타입/소규모 | 대규모/고성능 | 분산/프로덕션 | RDB 통합 |
| **스케일** | ~100K 벡터 | ~1B | ~10B+ | ~10M |
| **메타데이터 필터링** | ✅ 내장 | ❌ | ✅ | ✅ |
| **설치/운영 복잡도** | 낮음 | 중간 | 높음 | 중간 |

---

## 1.4 강사님 피드백 요약

- "ChromaDB 사용해도 좋다. **명분이 있어야 하고, 어떤 문제로 왜 전환했는지가 중요**."
- "**PostgreSQL/MongoDB extension으로 VectorDB 쓰는 회사가 많다.** 기존 DB 유지하며 확장하면 리스크 적음."
- "단기엔 **Supabase** 권장. PostgreSQL 기반으로 확장성 좋음."
- 임베딩 배치: **트래픽 없을 때** 진행 권장. 컨테이너 환경으로 가볍게 시작 가능.

---

## 1.5 최종 선정 (V1: ChromaDB)

| 항목 | 내용 |
|------|------|
| **VectorDB** | ChromaDB 0.4.24 |
| **선정 근거** | 6주 개발, 소규모 데이터, 빠른 프로토타이핑 |
| **저장 방식** | 로컬 파일 시스템 |
| **임베딩 모델** | **Gemini `gemini-embedding-001`** (768차원) |

### ADR (의사결정 기록)

- **상황:** 6주 부트캠프, 사용자 100명 이하, 벡터 ~3,000개, 메타데이터 필터 필수  
- **결정:** V1 ChromaDB 선택  
- **근거:** pip 설치 즉시 가능, 메타데이터 필터 내장, LangChain 통합, 100K 벡터 이하 적합  
- **한계:** 대규모 트래픽 미검증, 분산 미지원  
- **전환 조건:** 사용자 1,000명 이상 시 벤치마크, 분산 필요 시 pgvector/Milvus 검토  

---

# Part 2. 설계 원칙 

## 2.1 컬렉션 단일화 + 메타데이터 필터

면접(인성/기술)을 **별도 컬렉션으로 나누지 않고**, **`interview_feedback` 한 컬렉션 + 메타데이터 `interview_type`** 으로 구분한다.

| 구분 | A안 (채택) | B안 (미채택) |
|------|------------|-------------|
| **구조** | 컬렉션 1개 + `interview_type` | 컬렉션 2개 (인성/기술 분리) |
| **검색 (기술만)** | `where={..., "interview_type": "technical"}` | 기술 전용 컬렉션만 조회 |
| **검색 (통합)** | `where={"user_id": user_id}` | 두 컬렉션 조회 후 병합 |

## 2.2 A안 선택 이유

1. 인성/기술 모두 "면접 Q&A·피드백"으로 **같은 임베딩 공간** 사용이 자연스러움  
2. 타입별/통합 검색 모두 **where 필터**로 처리 가능  
3. 컬렉션 수를 늘리지 않아 **스키마·코드 단순**  
4. 추후 면접 타입 추가 시 **메타데이터 값만 추가**하면 됨  

---

# Part 3. Collection 구조 및 상세 스키마

## 3.1 VectorDB vs RDB 역할

| 저장소 | 저장 데이터 | 용도 |
|--------|------------|------|
| **RDB (PostgreSQL)** | 사용자, 채팅방, 면접 세션, Q&A 원본 | CRUD, 트랜잭션 |
| **VectorDB (ChromaDB)** | 텍스트 임베딩, 메타데이터 | 유사도 검색 (RAG) |

## 3.2 사용 기술

| 항목 | 선택 |
|------|------|
| **VectorDB** | ChromaDB |
| **Embedding Model** | Gemini gemini-embedding-001 |
| **Embedding Dimension** | 768 |
| **Distance Metric** | Cosine Similarity |
| **청킹** | 512 tokens, 50 tokens overlap (이력서/채용공고) |

## 3.3 Collection 구조도

```
VectorDB (ChromaDB)
├── resumes            # 이력서 + 포트폴리오
├── job_postings       # 채용공고
├── portfolios         # 포트폴리오 (선택)
├── analysis_results   # 분석 결과 ("이전 피드백" RAG)
├── interview_feedback # 면접 Q&A + 평가 (interview_type: personality | technical)
└── chat_context       # 중요 대화 컨텍스트
```

### Collection 요약

| # | Collection | 저장 데이터 | 활용 시점 |
|---|------------|------------|----------|
| 1 | `resumes` | 이력서 + 포트폴리오 | 분석, 면접 질문 생성, 일반 대화 RAG |
| 2 | `job_postings` | 채용공고 | 분석, 면접 질문 생성 |
| 3 | `analysis_results` | 분석/매칭 결과 | "이전 피드백 뭐였지?" RAG |
| 4 | `interview_feedback` | 면접 Q&A + 평가 | 약점 기반 질문 생성 (A안: interview_type 필터) |
| 5 | `chat_context` | 중요 대화 | 맥락 유지 대화 |

---

## 3.4 상세 스키마

### 3.4.1 `resumes` (이력서 + 포트폴리오)

**메타데이터:** `user_id`, `document_id`, `document_type`("resume"|"portfolio"), `file_name`, `chunk_index`, `total_chunks`, `created_at`, `updated_at`

```python
collection.add(
    ids=["resume_123_chunk_0", ...],
    embeddings=[[...], ...],  # 768차원
    documents=["이력서 청크 1...", ...],
    metadatas=[{"user_id": "user_456", "document_id": "resume_123", "document_type": "resume", ...}]
)
```

### 3.4.2 `job_postings`

**메타데이터:** `user_id`, `posting_id`, `company_name`, `position`, `job_type`, `deadline`, `chunk_index`, `total_chunks`, `created_at`

### 3.4.3 `analysis_results`

**메타데이터:** `user_id`, `room_id`, `resume_id`, `posting_id`, `analysis_type`, `score`, `grade`, `created_at`  
**document:** 이력서 분석·매칭 결과 전체 텍스트

### 3.4.4 `interview_feedback` (A안)

**메타데이터:** `user_id`, `room_id`, `session_id`, **`interview_type`: "technical" | "personality"**, `total_score`, `grade`, `question_count`, `ended_by`, `weakness_keywords`, `created_at`

**쿼리 예:**  
- 기술만: `where={"user_id": user_id, "interview_type": "technical"}`  
- 인성만: `where={"user_id": user_id, "interview_type": "personality"}`  
- 통합: `where={"user_id": user_id}`  

### 3.4.5 `chat_context`

**메타데이터:** `user_id`, `room_id`, `context_type`, `importance`, `created_at`, `expires_at`

---

# Part 4. 임베딩 모델

**최종 선정:** **Gemini `gemini-embedding-001`** (768차원). 상세 비교·선정 근거는 `0.모델 선정/07_임베딩_모델_선정.md` 참고.

---

# Part 5. 활용 시나리오별 쿼리

## 5.1 이력서 분석 시 RAG

```python
query_vector = embeddings.embed_query(resume_text)
results = analysis_collection.query(
    query_embeddings=[query_vector],
    n_results=3,
    where={"user_id": user_id}
)
# LLM 프롬프트에 results['documents'] 참조로 제공
```

## 5.2 면접 질문 생성 시 RAG (A안)

```python
# 이력서 + 이전 면접 피드백 검색 (interview_type으로 기술/인성 구분)
resume_results = resume_collection.query(..., where={"user_id": user_id})
feedback_results = interview_collection.query(
    ...,
    where={"user_id": user_id, "interview_type": interview_type}  # "technical" | "personality"
)
# 통합 검색 시 where={"user_id": user_id} 만 사용
```

## 5.3 일반 대화 RAG

```python
# chat_context + resumes 검색 후 LLM 프롬프트 구성
context_results = chat_context_collection.query(..., where={"user_id": user_id, "room_id": room_id})
resume_results = resume_collection.query(..., where={"user_id": user_id})
```

## 5.4 Hybrid Search (BM25 + Vector, 선택)

```python
# EnsembleRetriever: BM25 30% + Vector 70%, Top 20 → Reranker로 Top 3 정제
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)
```

---

# Part 6. 데이터 생명주기

## 6.1 저장 시점

| Collection | 저장 시점 | API |
|------------|----------|-----|
| `resumes` | 파일 업로드 후 임베딩 | `/ai/file/embed` |
| `job_postings` | 파일 업로드 후 임베딩 | `/ai/file/embed` |
| `analysis_results` | 분석 완료 후 | `/ai/analyze` 내부 |
| `interview_feedback` | 면접 종료 후 리포트 생성 | `/ai/interview/report` 내부 |
| `chat_context` | 중요 대화 감지 시 | `/ai/chat` 내부 |

## 6.2 삭제/업데이트 정책

| Collection | 삭제 조건 | 업데이트 |
|------------|----------|----------|
| `resumes` / `job_postings` | 파일 삭제 시 | 재업로드 시 기존 삭제 후 재저장 |
| `analysis_results` | 채팅방 삭제 시 | 재분석 시 새로 추가 |
| `interview_feedback` | 채팅방 삭제 시 | 불변(이력 유지) |
| `chat_context` | 만료 시간 도래 | 중요도 변경 시 |

## 6.3 만료 정책 (chat_context)

```python
# 정기 정리: expires_at < now 인 문서 삭제
expired = chat_context_collection.get(where={"expires_at": {"$lt": now}})
if expired['ids']:
    chat_context_collection.delete(ids=expired['ids'])
```

---

# Part 7. 운영 고려사항

## 7.1 임베딩 주기

| 방식 | 설명 | 권장 |
|------|------|------|
| **실시간** | 문서 업로드 즉시 임베딩 | V1 (트래픽 적음) |
| **배치** | 새벽 시간대 일괄 처리 | 트래픽 많을 때 (Celery + 크론) |

**현재:** 실시간

## 7.2 중복 데이터

- **file_id 기준 덮어쓰기:** 기존 문서 삭제 후 동일 file_id로 재추가  
- **유사도 기반 중복 체크(선택):** threshold 0.95 등으로 유사 문서 판별

## 7.3 컨테이너 배포

```yaml
# docker-compose: 영속성 볼륨
volumes:
  - ./chroma_data:/app/chroma_data
environment:
  - CHROMA_PERSIST_DIRECTORY=/app/chroma_data
```

---

# Part 8. 확장 계획

- **Phase 1 (V1):** ChromaDB, 로컬 파일, 사용자 100명 이하  
- **Phase 2 (V3 검토):** pgvector(PostgreSQL Extension) 또는 MongoDB Atlas Vector Search — 기존 DB 활용, 현업 다수 사용  
- **Phase 3 (미래):** Milvus/Pinecone — 분산·고가용성 필요 시  

### ChromaDB → pgvector 마이그레이션

1. pgvector 테이블 생성  
2. ChromaDB에서 벡터·메타데이터 추출  
3. pgvector에 일괄 삽입  
4. API에서 VectorDB 서비스 교체 후 테스트  
5. ChromaDB 데이터 삭제  

---

# Part 9. 구현 참조 및 RDB 관계

## 9.1 구현 참조

- **코드:** `3.model/app/services/vectordb_service.py`, `3.model/app/infrastructure/vectordb/chroma.py`  
- **설정:** `chroma_persist_dir`, `chroma_collection_resume` 등 (`app/config/settings.py`)

## 9.2 RDB와의 관계

- VectorDB에는 **FK 제약 없음**. 메타데이터에 RDB ID(`document_id`, `posting_id`, `session_id` 등)를 저장해 **애플리케이션 레벨**에서 관계 관리.  
- RDB: 사용자, 채팅방, 면접 세션, Q&A 원본 보관. VectorDB: 해당 문서의 임베딩·메타데이터만 보관.

| Collection | RDB 연동 키 (메타데이터) |
|------------|--------------------------|
| `resumes` | `document_id` ↔ RDB resumes.id |
| `job_postings` | `posting_id` ↔ RDB job_postings.id |
| `analysis_results` | `room_id`, `resume_id`, `posting_id` |
| `interview_feedback` | `session_id` ↔ RDB interview_sessions.id |
| `chat_context` | `room_id` ↔ RDB chat_rooms.id |

---

# 참고 자료

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [pgvector GitHub](https://github.com/pgvector/pgvector)
- [Milvus Documentation](https://milvus.io/docs)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Supabase Vector](https://supabase.com/docs/guides/ai)

---

*이 문서는 기존 `09_VectorDB_선정.md`, `09_VectorDB_설계.md`, `09_VectorDB_Collection_스키마.md` 내용을 하나로 통합한 최종 버전이다.*
