# RAG 전략

> **작성 상태**: ✅ 완료
> **최종 업데이트**: 2026-01-25

---

## 📋 목차

- [1. 개요](#1-개요)
- [2. 요구사항](#2-요구사항)
- [3. RAG 아키텍처](#3-rag-아키텍처)
- [4. VectorDB 구성](#4-vectordb-구성)
- [5. 검색 전략](#5-검색-전략)
- [6. 컨텍스트 관리](#6-컨텍스트-관리)
- [7. 구현 참조](#7-구현-참조)

---

## 1. 개요

### 목적
Retrieval-Augmented Generation을 통해 개인화된 응답 생성

### RAG란?
```
사용자 질문 → 관련 문서 검색 → 컨텍스트와 함께 LLM에 전달 → 응답 생성

일반 LLM: "취업 준비 어떻게 해?" → 일반적인 조언
RAG LLM: "취업 준비 어떻게 해?" + [이력서] + [채용공고] → 개인화된 조언
```

---

## 2. 요구사항

| 요구사항 | 목표 | 우선순위 |
|---------|------|---------|
| **검색 정확도** | 관련 문서 Top-5에 포함 | 높음 |
| **응답 속도** | 검색 포함 3초 이내 | 높음 |
| **컨텍스트 품질** | 관련성 높은 정보만 포함 | 높음 |

---

## 3. RAG 아키텍처

### 전체 흐름

```
┌─────────────────────────────────────────────────────┐
│                   사용자 질문                         │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               질문 임베딩 (Gemini Embedding)          │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                VectorDB 검색 (ChromaDB)              │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐             │
│  │ resumes │  │job_posts│  │portfolios│            │
│  └─────────┘  └─────────┘  └─────────┘             │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              컨텍스트 구성 (최대 4000자)              │
│  "[출처: 이력서]\n{내용}\n[출처: 채용공고]\n{내용}"    │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│               LLM 응답 생성 (Gemini Flash)           │
│  프롬프트 = 시스템 메시지 + 컨텍스트 + 질문           │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                   스트리밍 응답                       │
└─────────────────────────────────────────────────────┘
```

---

## 4. VectorDB 구성

### 사용 기술: ChromaDB

| 항목 | 내용 |
|------|------|
| **VectorDB** | ChromaDB 0.4.24 |
| **저장 방식** | 로컬 파일 시스템 |
| **임베딩** | Gemini Embedding (768차원) |

### 컬렉션 구조

| 컬렉션 | 용도 | 필터 |
|--------|------|------|
| **resumes** | 이력서 텍스트 | user_id |
| **job_postings** | 채용공고 텍스트 | user_id |
| **portfolios** | 포트폴리오 텍스트 | user_id |

### 메타데이터 스키마

```python
{
    "user_id": "user_123",
    "created_at": "2026-01-25T10:00:00",
    "file_id": 42,
    "collection_type": "resume"
}
```

---

## 5. 검색 전략

### 5.1 일반 대화 모드

```python
# 빠른 응답을 위해 1개만 검색
results = vectordb.query(
    collection="resumes",
    query_embedding=query_embedding,
    n_results=1,
    where={"user_id": user_id}
)
```

### 5.2 분석 모드

```python
# 정확도를 위해 모든 관련 문서 검색
results = vectordb.get_all_documents_by_user(
    user_id=user_id,
    collections=["resumes", "job_postings", "portfolios"]
)
```

### 5.3 검색 최적화

| 모드 | n_results | 근거 |
|------|-----------|------|
| **일반 대화** | 1 | 속도 우선 |
| **분석** | 전체 | 정확도 우선 |
| **면접** | 전체 | 모든 정보 필요 |

---

## 6. 컨텍스트 관리

### 컨텍스트 길이 제한

```python
MAX_CONTEXT_LENGTH = 4000  # 약 1000 토큰

def truncate_context(context: str, max_length: int = 4000) -> str:
    """컨텍스트 길이 제한"""
    if len(context) <= max_length:
        return context
    return context[:max_length] + "\n...(truncated)"
```

### 컨텍스트 구성 예시

```
[출처: 이력서]
3년 경력의 백엔드 개발자입니다.
주요 기술: Python, FastAPI, PostgreSQL, Docker
프로젝트: DAU 100만 서비스 API 개발 경험

[출처: 채용공고]
회사: 카카오
포지션: 백엔드 개발자 (경력 3-5년)
필수 기술: Python, FastAPI, Kubernetes
우대 기술: AWS, Kafka

질문: 내가 이 회사에 지원하면 어떨까?
```

### 프롬프트 템플릿

```python
RAG_PROMPT = """
당신은 취업/진로 상담 전문가입니다.
사용자의 이력서와 채용공고 정보를 참고하여 개인화된 조언을 제공하세요.

## 참고 정보
{context}

## 사용자 질문
{user_message}

## 응답 가이드
- 구체적이고 실행 가능한 조언을 제공하세요
- 이력서와 채용공고를 비교 분석하세요
- 긍정적인 톤을 유지하되 현실적인 피드백을 주세요
"""
```

---

## 7. 구현 참조

### 코드 파일
- `3.model/app/services/rag_service.py` - RAG 파이프라인
- `3.model/app/services/vectordb_service.py` - VectorDB 관리

### 주요 메서드

```python
class RAGService:
    async def retrieve_context(
        self,
        user_id: str,
        query: str,
        n_results: int = 1
    ) -> str:
        """쿼리 기반 컨텍스트 검색"""

    async def retrieve_all_documents(
        self,
        user_id: str
    ) -> str:
        """사용자의 모든 문서 검색"""

    async def chat_with_rag(
        self,
        user_id: str,
        message: str,
        model: str = "gemini"
    ) -> AsyncGenerator[str, None]:
        """RAG 기반 채팅 (스트리밍)"""

    async def analyze_resume_and_posting(
        self,
        user_id: str
    ) -> dict:
        """이력서-채용공고 분석"""
```

### API 사용 예시

```python
# 요청
{
    "room_id": "room_001",
    "user_id": "user_123",
    "message": "내 이력서로 이 회사에 지원하면 어떨까?",
    "model": "gemini",
    "context": {"mode": "general"}
}

# 내부 처리
# 1. user_123의 이력서/채용공고 검색
# 2. 컨텍스트 구성
# 3. Gemini Flash에 전달
# 4. 스트리밍 응답 반환
```

---

## 참고 자료

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [RAG Paper](https://arxiv.org/abs/2005.11401)
- [LangChain RAG Guide](https://python.langchain.com/docs/use_cases/question_answering/)
