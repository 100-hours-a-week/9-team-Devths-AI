# LangChain / LangGraph 비교

> **작성 상태**: ✅ 완료
> **최종 업데이트**: 2026-01-25

---

## 📋 목차

- [1. 개요](#1-개요)
- [2. 프레임워크 비교](#2-프레임워크-비교)
- [3. 현재 선택](#3-현재-선택)
- [4. 향후 계획](#4-향후-계획)

---

## 1. 개요

### 목적
LLM 애플리케이션 개발을 위한 프레임워크 평가

### 평가 대상
- LangChain: LLM 애플리케이션 개발 프레임워크
- LangGraph: 상태 기반 에이전트 워크플로우 프레임워크
- 직접 구현: 프레임워크 없이 직접 API 호출

---

## 2. 프레임워크 비교

### 2.1 LangChain

**개요**:
- LLM 애플리케이션 개발을 위한 종합 프레임워크
- 프롬프트, 체인, 에이전트, 메모리 등 제공

**장점**:
- ✅ 풍부한 기능 (프롬프트 템플릿, 체인, RAG)
- ✅ 다양한 LLM 통합 (OpenAI, Gemini, Anthropic)
- ✅ 커뮤니티 활발
- ✅ 문서화 잘 되어 있음

**단점**:
- ❌ 학습 곡선 존재
- ❌ 추상화 레이어로 인한 디버깅 어려움
- ❌ 버전 업데이트가 빠름 (Breaking Changes)
- ❌ 오버헤드 존재

**적합한 경우**:
- 빠른 프로토타이핑
- 다양한 LLM 전환이 필요한 경우
- 복잡한 체인 구성

---

### 2.2 LangGraph

**개요**:
- LangChain 팀이 만든 상태 기반 워크플로우 프레임워크
- 그래프 구조로 복잡한 에이전트 로직 구현

**장점**:
- ✅ 상태 관리 우수 (State Machine)
- ✅ 복잡한 워크플로우 시각화
- ✅ 조건부 분기 쉽게 구현
- ✅ 병렬 실행 지원

**단점**:
- ❌ 학습 곡선 높음
- ❌ 간단한 작업에는 오버킬
- ❌ LangChain 의존성

**적합한 경우**:
- 복잡한 멀티스텝 에이전트
- 조건부 분기가 많은 워크플로우
- 상태 추적이 중요한 경우

---

### 2.3 직접 구현 ✅ 현재 선택

**장점**:
- ✅ 완전한 제어권
- ✅ 프레임워크 오버헤드 없음
- ✅ 디버깅 용이
- ✅ 의존성 최소화
- ✅ 성능 최적화 가능

**단점**:
- ❌ 모든 기능 직접 구현
- ❌ LLM 전환 시 코드 수정 필요
- ❌ 보일러플레이트 코드

**적합한 경우**:
- 간단한 파이프라인
- 성능이 중요한 경우
- 특정 LLM에 최적화된 구현

---

## 3. 현재 선택

### 선택: 직접 구현 (프레임워크 미사용)

| 항목 | 내용 |
|------|------|
| **선택** | Gemini SDK 직접 사용 |
| **근거** | 간단한 파이프라인, 오버헤드 최소화 |
| **장점** | 빠른 응답, 쉬운 디버깅 |

### 현재 구조

```python
# 직접 구현 방식
import google.generativeai as genai

class LLMService:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    async def generate_response(self, prompt: str) -> str:
        response = await self.model.generate_content_async(prompt)
        return response.text

class RAGService:
    def __init__(self, llm_service, vectordb_service):
        self.llm = llm_service
        self.vectordb = vectordb_service

    async def chat(self, user_id: str, message: str):
        # 1. VectorDB 검색
        context = self.vectordb.search(user_id, message)
        # 2. 프롬프트 구성
        prompt = f"{context}\n\n질문: {message}"
        # 3. LLM 응답
        return await self.llm.generate_response(prompt)
```

### 비교: LangChain 사용 시

```python
# LangChain 사용 시
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
vectorstore = Chroma(...)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
)
response = qa_chain.invoke({"query": message})
```

---

## 4. 향후 계획

### V3+ 도입 검토 시점

**LangGraph 도입 고려 시점**:
- 복잡한 멀티스텝 워크플로우 필요 시
- Routing 로직이 복잡해질 때
- 상태 추적/롤백이 필요할 때

**예상 사용 케이스**:
```
면접 모드 워크플로우:
1. 질문 생성 → 2. 답변 수집 → 3. 평가 → 4. 피드백
   ↑                                      │
   └──────── 추가 질문 필요 시 ────────────┘
```

### 도입 기준

| 기준 | 현재 | 도입 시점 |
|------|------|----------|
| **워크플로우 복잡도** | 단순 (선형) | 복잡 (분기, 루프) |
| **상태 관리** | 불필요 | 필수 |
| **개발 속도** | 충분 | 병목 |

---

## 참고 자료

- [LangChain Documentation](https://python.langchain.com/docs)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Gemini API Python SDK](https://ai.google.dev/gemini-api/docs/quickstart?lang=python)
