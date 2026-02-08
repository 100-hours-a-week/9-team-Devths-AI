# VectorDB Collection 스키마 설계서

> **프로젝트:** Devths AI 취업 도우미  
> **VectorDB:** ChromaDB  
> **Embedding Model:** Gemini text-embedding-004 (768차원)  
> **작성일:** 2026-01-13

---

## 📚 목차

- [1. 개요](#1-개요)
- [2. Collection 전체 구조](#2-collection-전체-구조)
- [3. Collection 상세 스키마](#3-collection-상세-스키마)
- [4. 검색 쿼리 예시](#4-검색-쿼리-예시)
- [5. RDB와의 관계](#5-rdb와의-관계)

---

## 1. 개요

### 1.1. VectorDB vs RDB 역할 분담

| 저장소 | 저장 데이터 | 용도 | 관계 |
|--------|------------|------|------|
| **RDB (PostgreSQL)** | 사용자 정보, 채팅방, 면접 세션, Q&A 원본 | CRUD, 트랜잭션 | ✅ FK로 관계 설정 |
| **VectorDB (ChromaDB)** | 텍스트 임베딩, 메타데이터 | 유사도 검색 (RAG) | ❌ 관계 없음 (독립적) |

**핵심 차이:**
- ✅ RDB: ERD 설계 필요, JOIN 가능
- ✅ VectorDB: Collection 스키마 문서, JOIN 불가

---

### 1.2. 기술 스택

| 항목 | 값 |
|------|-----|
| **VectorDB** | ChromaDB |
| **Embedding Model** | Gemini text-embedding-004 |
| **Embedding Dimension** | 768 |
| **Distance Metric** | Cosine Similarity |
| **청킹 전략** | 512 tokens, 50 tokens overlap |

---

## 2. Collection 전체 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                   VectorDB Collection 구조                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Collection 1: resumes                                 │    │
│  │  - 이력서/포트폴리오 임베딩                             │    │
│  │  - 활용: 분석, 면접 질문 생성, 일반 대화 RAG           │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Collection 2: job_postings                            │    │
│  │  - 채용공고 임베딩                                      │    │
│  │  - 활용: 매칭 분석, 면접 질문 생성                      │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Collection 3: analysis_results                        │    │
│  │  - 분석 결과 임베딩                                     │    │
│  │  - 활용: "이전 피드백 뭐였지?" RAG                      │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Collection 4: interview_feedback                      │    │
│  │  - 면접 Q&A + 평가 임베딩                               │    │
│  │  - 활용: 약점 기반 질문 생성                            │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  Collection 5: chat_context                            │    │
│  │  - 중요 대화 컨텍스트 임베딩                            │    │
│  │  - 활용: 맥락 유지한 대화                               │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Collection 상세 스키마

### 3.1. Collection: `resumes`

**용도:** 이력서/포트폴리오 임베딩 저장

**스키마:**

| 필드 | 타입 | 필수 | 설명 | 예시 |
|------|------|:----:|------|------|
| **id** | string | ✅ | 문서 ID | `resume_123_chunk_0` |
| **embedding** | vector[768] | ✅ | Gemini Embedding 벡터 | `[0.1, 0.2, ..., 0.768]` |
| **document** | string | ✅ | 청크 텍스트 원본 | `"이력서 텍스트..."` |
| **metadata** | object | ✅ | 메타데이터 | 아래 참조 |

**metadata 스키마:**

| 필드 | 타입 | 필수 | 설명 | 예시 |
|------|------|:----:|------|------|
| `user_id` | string | ✅ | 사용자 ID | `user_456` |
| `document_id` | string | ✅ | 이력서 ID (RDB FK) | `resume_123` |
| `document_type` | string | ✅ | 문서 유형 | `resume` 또는 `portfolio` |
| `file_name` | string | ✅ | 원본 파일명 | `홍길동_이력서.pdf` |
| `chunk_index` | int | ✅ | 청크 인덱스 | `0` |
| `total_chunks` | int | ✅ | 전체 청크 수 | `5` |
| `created_at` | datetime | ✅ | 생성 시간 | `2026-01-13T10:00:00Z` |
| `updated_at` | datetime | ✅ | 수정 시간 | `2026-01-13T10:00:00Z` |

**생성 예시:**

```python
from chromadb import Client

client = Client()
collection = client.create_collection(
    name="resumes",
    metadata={"hnsw:space": "cosine"}
)

# 문서 추가
collection.add(
    ids=["resume_123_chunk_0", "resume_123_chunk_1"],
    embeddings=[
        [0.1, 0.2, ..., 0.768],  # 768차원
        [0.3, 0.4, ..., 0.768]
    ],
    documents=[
        "이력서 텍스트 청크 1...",
        "이력서 텍스트 청크 2..."
    ],
    metadatas=[
        {
            "user_id": "user_456",
            "document_id": "resume_123",
            "document_type": "resume",
            "file_name": "홍길동_이력서.pdf",
            "chunk_index": 0,
            "total_chunks": 5,
            "created_at": "2026-01-13T10:00:00Z",
            "updated_at": "2026-01-13T10:00:00Z"
        },
        {
            "user_id": "user_456",
            "document_id": "resume_123",
            "document_type": "resume",
            "file_name": "홍길동_이력서.pdf",
            "chunk_index": 1,
            "total_chunks": 5,
            "created_at": "2026-01-13T10:00:00Z",
            "updated_at": "2026-01-13T10:00:00Z"
        }
    ]
)
```

---

### 3.2. Collection: `job_postings`

**용도:** 채용공고 임베딩 저장

**스키마:**

| 필드 | 타입 | 필수 | 설명 | 예시 |
|------|------|:----:|------|------|
| **id** | string | ✅ | 문서 ID | `posting_789_chunk_0` |
| **embedding** | vector[768] | ✅ | 임베딩 벡터 | `[0.1, 0.2, ..., 0.768]` |
| **document** | string | ✅ | 청크 텍스트 원본 | `"채용공고 텍스트..."` |
| **metadata** | object | ✅ | 메타데이터 | 아래 참조 |

**metadata 스키마:**

| 필드 | 타입 | 필수 | 설명 | 예시 |
|------|------|:----:|------|------|
| `user_id` | string | ✅ | 사용자 ID | `user_456` |
| `posting_id` | string | ✅ | 채용공고 ID (RDB FK) | `posting_789` |
| `company_name` | string | ✅ | 회사명 | `카카오` |
| `position` | string | ✅ | 직무 | `백엔드 개발자` |
| `job_type` | string | ✅ | 채용 유형 | `신입`, `경력`, `인턴` |
| `deadline` | date | ⚠️ | 서류 마감일 | `2026-01-15` |
| `chunk_index` | int | ✅ | 청크 인덱스 | `0` |
| `total_chunks` | int | ✅ | 전체 청크 수 | `3` |
| `created_at` | datetime | ✅ | 생성 시간 | `2026-01-13T10:00:00Z` |

**생성 예시:**

```python
collection = client.create_collection(
    name="job_postings",
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    ids=["posting_789_chunk_0"],
    embeddings=[[0.1, 0.2, ..., 0.768]],
    documents=["채용공고 텍스트 청크 1..."],
    metadatas=[{
        "user_id": "user_456",
        "posting_id": "posting_789",
        "company_name": "카카오",
        "position": "백엔드 개발자",
        "job_type": "신입",
        "deadline": "2026-01-15",
        "chunk_index": 0,
        "total_chunks": 3,
        "created_at": "2026-01-13T10:00:00Z"
    }]
)
```

---

### 3.3. Collection: `analysis_results`

**용도:** 이력서 분석/매칭 결과 임베딩 저장

**스키마:**

| 필드 | 타입 | 필수 | 설명 | 예시 |
|------|------|:----:|------|------|
| **id** | string | ✅ | 분석 결과 ID | `analysis_001` |
| **embedding** | vector[768] | ✅ | 분석 결과 임베딩 | `[0.1, 0.2, ..., 0.768]` |
| **document** | string | ✅ | 분석 결과 전체 텍스트 | 아래 예시 참조 |
| **metadata** | object | ✅ | 메타데이터 | 아래 참조 |

**document 예시:**

```
이력서 분석 결과:
- 강점: 3년 프론트엔드 경험, React/TypeScript 숙련
- 약점: 클라우드 경험 부족
- 제안: AWS 자격증 취득 권장

매칭도: 85점 (A등급)
- 보유 스킬: React, TypeScript
- 부족 스킬: GraphQL, Next.js
```

**metadata 스키마:**

| 필드 | 타입 | 필수 | 설명 | 예시 |
|------|------|:----:|------|------|
| `user_id` | string | ✅ | 사용자 ID | `user_456` |
| `room_id` | string | ✅ | 채팅방 ID | `room_001` |
| `resume_id` | string | ✅ | 이력서 ID (RDB FK) | `resume_123` |
| `posting_id` | string | ⚠️ | 채용공고 ID (RDB FK) | `posting_789` |
| `analysis_type` | string | ✅ | 분석 유형 | `resume_only`, `matching_only`, `full` |
| `score` | int | ⚠️ | 매칭 점수 (0-100) | `85` |
| `grade` | string | ⚠️ | 등급 | `S`, `A`, `B`, `C`, `D` |
| `created_at` | datetime | ✅ | 생성 시간 | `2026-01-13T10:00:00Z` |

**생성 예시:**

```python
collection = client.create_collection(
    name="analysis_results",
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    ids=["analysis_001"],
    embeddings=[[0.1, 0.2, ..., 0.768]],
    documents=["""
        이력서 분석 결과:
        - 강점: 3년 프론트엔드 경험, React/TypeScript 숙련
        - 약점: 클라우드 경험 부족
        - 제안: AWS 자격증 취득 권장
        
        매칭도: 85점 (A등급)
        - 보유 스킬: React, TypeScript
        - 부족 스킬: GraphQL, Next.js
    """],
    metadatas=[{
        "user_id": "user_456",
        "room_id": "room_001",
        "resume_id": "resume_123",
        "posting_id": "posting_789",
        "analysis_type": "full",
        "score": 85,
        "grade": "A",
        "created_at": "2026-01-13T10:00:00Z"
    }]
)
```

---

### 3.4. Collection: `interview_feedback`

**용도:** 면접 Q&A + 평가 임베딩 저장

**스키마:**

| 필드 | 타입 | 필수 | 설명 | 예시 |
|------|------|:----:|------|------|
| **id** | string | ✅ | 세션 ID 또는 QA ID | `interview_session_abc123` |
| **embedding** | vector[768] | ✅ | Q&A/피드백 임베딩 | `[0.1, 0.2, ..., 0.768]` |
| **document** | string | ✅ | Q&A + 평가 텍스트 | 아래 예시 참조 |
| **metadata** | object | ✅ | 메타데이터 | 아래 참조 |

**document 예시 (세션 전체):**

```
면접 유형: 기술 면접
총점: 78점 (B+)

Q1: React Virtual DOM이란?
A1: 실제 DOM과 비교해서...
평가: 80점 - 개념 이해 우수, Reconciliation 설명 추가 필요

Q2: diffing 알고리즘 동작 방식?
A2: 이전 트리와 새 트리를...
평가: 75점 - 기본 원리 이해, 시간 복잡도 설명 부족

강점 패턴: 기술 개념 이해도 높음
약점 패턴: 심화 개념 설명 부족, 답변 길이 짧음
학습 가이드: React Fiber, Concurrent Mode 학습
```

**metadata 스키마 (세션):**

| 필드 | 타입 | 필수 | 설명 | 예시 |
|------|------|:----:|------|------|
| `user_id` | string | ✅ | 사용자 ID | `user_456` |
| `room_id` | string | ✅ | 채팅방 ID | `room_001` |
| `session_id` | string | ✅ | 면접 세션 ID (RDB FK) | `session_abc123` |
| `interview_type` | string | ✅ | 면접 유형 | `technical`, `personality` |
| `total_score` | int | ✅ | 총점 (0-100) | `78` |
| `grade` | string | ✅ | 등급 | `B+` |
| `question_count` | int | ✅ | 질문 수 | `5` |
| `ended_by` | string | ✅ | 종료 방식 | `auto`, `manual` |
| `weakness_keywords` | list[string] | ✅ | 약점 키워드 | `["심화 개념", "답변 구조화"]` |
| `created_at` | datetime | ✅ | 생성 시간 | `2026-01-13T10:00:00Z` |

**생성 예시:**

```python
collection = client.create_collection(
    name="interview_feedback",
    metadata={"hnsw:space": "cosine"}
)

# 세션 전체 저장
collection.add(
    ids=["interview_session_abc123"],
    embeddings=[[0.1, 0.2, ..., 0.768]],
    documents=["""
        면접 유형: 기술 면접
        총점: 78점 (B+)
        
        Q1: React Virtual DOM이란?
        A1: 실제 DOM과 비교해서...
        평가: 80점 - 개념 이해 우수
        
        강점 패턴: 기술 개념 이해도 높음
        약점 패턴: 심화 개념 설명 부족
    """],
    metadatas=[{
        "user_id": "user_456",
        "room_id": "room_001",
        "session_id": "session_abc123",
        "interview_type": "technical",
        "total_score": 78,
        "grade": "B+",
        "question_count": 5,
        "ended_by": "auto",
        "weakness_keywords": ["심화 개념", "답변 구조화"],
        "created_at": "2026-01-13T10:00:00Z"
    }]
)
```

---

### 3.5. Collection: `chat_context`

**용도:** 중요 대화 컨텍스트 임베딩 저장

**스키마:**

| 필드 | 타입 | 필수 | 설명 | 예시 |
|------|------|:----:|------|------|
| **id** | string | ✅ | 컨텍스트 ID | `context_001` |
| **embedding** | vector[768] | ✅ | 컨텍스트 임베딩 | `[0.1, 0.2, ..., 0.768]` |
| **document** | string | ✅ | 컨텍스트 요약 텍스트 | 아래 예시 참조 |
| **metadata** | object | ✅ | 메타데이터 | 아래 참조 |

**document 예시:**

```
사용자가 카카오 백엔드 포지션에 관심 있음.
React 3년 경력 보유, 클라우드 경험 부족.
AWS 자격증 취득 예정.
```

**metadata 스키마:**

| 필드 | 타입 | 필수 | 설명 | 예시 |
|------|------|:----:|------|------|
| `user_id` | string | ✅ | 사용자 ID | `user_456` |
| `room_id` | string | ✅ | 채팅방 ID | `room_001` |
| `context_type` | string | ✅ | 컨텍스트 유형 | `preference`, `advice`, `goal` |
| `importance` | string | ✅ | 중요도 | `high`, `medium`, `low` |
| `created_at` | datetime | ✅ | 생성 시간 | `2026-01-13T10:00:00Z` |
| `expires_at` | datetime | ✅ | 만료 시간 | `2026-04-13T10:00:00Z` (3개월 후) |

**생성 예시:**

```python
collection = client.create_collection(
    name="chat_context",
    metadata={"hnsw:space": "cosine"}
)

collection.add(
    ids=["context_001"],
    embeddings=[[0.1, 0.2, ..., 0.768]],
    documents=[
        "사용자가 카카오 백엔드 포지션에 관심 있음. "
        "React 3년 경력 보유, 클라우드 경험 부족. "
        "AWS 자격증 취득 예정."
    ],
    metadatas=[{
        "user_id": "user_456",
        "room_id": "room_001",
        "context_type": "preference",
        "importance": "high",
        "created_at": "2026-01-13T10:00:00Z",
        "expires_at": "2026-04-13T10:00:00Z"
    }]
)
```

---

## 4. 검색 쿼리 예시

### 4.1. 이력서 기반 RAG 검색

```python
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 임베딩 모델
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004"
)

# 사용자 질문
user_message = "내 이력서 분석해줘"

# 1. 쿼리 임베딩
query_vector = embeddings.embed_query(user_message)

# 2. VectorDB 검색
results = resumes_collection.query(
    query_embeddings=[query_vector],
    n_results=5,
    where={"user_id": "user_456"}
)

# 3. 결과 활용
context = "\n".join(results['documents'][0])
print(f"검색된 컨텍스트: {context}")
```

---

### 4.2. 면접 약점 기반 질문 생성

```python
# 1. 이전 면접 피드백 검색
feedback_results = interview_feedback_collection.query(
    query_embeddings=[query_vector],
    n_results=2,
    where={
        "user_id": "user_456",
        "interview_type": "technical"
    }
)

# 2. 이력서 검색
resume_results = resumes_collection.query(
    query_embeddings=[query_vector],
    n_results=3,
    where={"user_id": "user_456"}
)

# 3. LLM 프롬프트 구성
prompt = f"""
이력서 관련 내용:
{resume_results['documents']}

이전 면접에서의 약점:
{feedback_results['documents']}

위 약점을 보완할 수 있는 기술 면접 질문을 생성해주세요.
"""
```

---

### 4.3. Hybrid Search (BM25 + Vector)

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

# BM25 Retriever
bm25_retriever = BM25Retriever.from_documents(
    documents=vectordb.get_documents("resumes"),
    k=20
)

# Vector Retriever
vector_retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 20,
        "filter": {"user_id": "user_456"}
    }
)

# Ensemble Retriever (BM25 30% + Vector 70%)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.3, 0.7]
)

# 검색
docs = ensemble_retriever.get_relevant_documents("React 경험")
```

---

## 5. RDB와의 관계

### 5.1. VectorDB ↔ RDB 연동

```
┌─────────────────────────────────────────────────────────┐
│  RDB (PostgreSQL)                                       │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌─────────┐       ┌──────────┐       ┌────────────┐   │
│  │ users   │1─────N│ resumes  │       │ chat_rooms │   │
│  │─────────│       │──────────│       │────────────│   │
│  │ id (PK) │       │ id (PK)  │       │ id (PK)    │   │
│  │ email   │       │ user_id  │       │ user_id    │   │
│  └─────────┘       │ file_url │       └────────────┘   │
│                    └──────────┘                         │
│                         │                               │
│                         │ (참조)                        │
│                         ▼                               │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  VectorDB (ChromaDB)                                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌────────────────────────────────────────────────┐    │
│  │  resumes Collection                            │    │
│  │  ┌──────────────────────────────────────────┐  │    │
│  │  │  metadata.document_id = "resume_123"     │  │    │
│  │  │  (RDB resumes.id 참조)                   │  │    │
│  │  └──────────────────────────────────────────┘  │    │
│  └────────────────────────────────────────────────┘    │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

**핵심:**
- ✅ VectorDB의 `metadata.document_id`가 RDB의 `resumes.id`를 참조
- ✅ 하지만 FK 제약 조건은 없음 (독립적)
- ✅ 애플리케이션 레벨에서 관계 관리

---

### 5.2. 데이터 생명주기

| Collection | 저장 시점 | 삭제 시점 | RDB 연동 |
|------------|----------|----------|----------|
| `resumes` | 파일 업로드 후 임베딩 완료 | 파일 삭제 시 | `resumes.id` |
| `job_postings` | 파일 업로드 후 임베딩 완료 | 파일 삭제 시 | `job_postings.id` |
| `analysis_results` | 분석 완료 후 | 채팅방 삭제 시 | `chat_rooms.id` |
| `interview_feedback` | 면접 종료 후 리포트 생성 시 | 채팅방 삭제 시 | `interview_sessions.id` |
| `chat_context` | 중요 대화 감지 시 | 만료 시간 도래 시 | `chat_rooms.id` |

---

## 🎊 최종 요약

### Collection 목록

| # | Collection | 문서 수 (예상) | 용도 |
|---|------------|---------------|------|
| 1 | `resumes` | ~1,000 | 이력서/포트폴리오 RAG |
| 2 | `job_postings` | ~500 | 채용공고 RAG |
| 3 | `analysis_results` | ~2,000 | 이전 피드백 RAG |
| 4 | `interview_feedback` | ~1,000 | 약점 기반 질문 생성 |
| 5 | `chat_context` | ~500 | 맥락 유지 대화 |

### 핵심 특징

- ✅ **Embedding Model:** Gemini text-embedding-004 (768차원)
- ✅ **Distance Metric:** Cosine Similarity
- ✅ **Hybrid Search:** BM25 (30%) + Vector (70%)
- ✅ **Reranker:** Cohere Rerank (Top 20 → Top 3)
- ✅ **RDB 연동:** metadata에 RDB ID 저장 (FK 아님)

**이것으로 VectorDB Collection 스키마 설계가 완료되었습니다!** 🎉
