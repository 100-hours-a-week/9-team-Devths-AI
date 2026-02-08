# Routing 전략

> **작성 상태**: ✅ 완료
> **최종 업데이트**: 2026-01-25
> **구현 시점**: 나중에 (V2 이후)

---

## 📋 목차

- [1. 개요](#1-개요)
- [2. 요구사항](#2-요구사항)
- [3. Routing 개념](#3-routing-개념)
- [4. 분류 카테고리](#4-분류-카테고리)
- [5. 구현 계획](#5-구현-계획)
- [6. 예상 아키텍처](#6-예상-아키텍처)

---

## 1. 개요

### 목적
LLM이 사용자 입력을 분석하여 적절한 서비스 모드로 자동 분기

### 구현 시점
**나중에 구현** (V1 이후, V2에서 검토)

### 현재 상태
V1에서는 **클라이언트에서 모드를 명시적으로 지정**

---

## 2. 요구사항

| 요구사항 | 목표 | 우선순위 |
|---------|------|---------|
| **분류 정확도** | 90% 이상 | 높음 |
| **응답 속도** | 추가 지연 1초 이내 | 높음 |
| **비용** | 최소화 (경량 모델 사용) | 중간 |

---

## 3. Routing 개념

### 3.1 현재 방식 (V1)

```
클라이언트 → 모드 명시 → 해당 모드 LLM 호출
                │
                └── mode: "general" | "analysis" | "interview_question"
```

### 3.2 목표 방식 (V2+)

```
사용자 입력 → Router LLM → 카테고리 분류 → 해당 모드 LLM 호출
     │              │
     │              └── "이건 면접을 보면 좋겠어"
     │                  → interview 모드로 자동 전환
     │
     └── "이력서 분석해줘"
         → analysis 모드로 자동 전환
```

### 3.3 핵심 시나리오

**매칭도 분석 → 자동 면접 제안**:
```
1. 사용자: "내 이력서와 채용공고 매칭도 분석해줘"
2. LLM 분석: "매칭도 85%, 면접 준비 권장"
3. Router: "면접 모드로 전환할까요?" (자동 제안)
4. 사용자: "좋아"
5. 시스템: 자동으로 면접 모드 시작
```

---

## 4. 분류 카테고리

| 카테고리 | 설명 | 키워드 예시 |
|----------|------|------------|
| **interview** | 면접 연습, 질문 생성, 답변 평가 | "면접", "질문", "연습" |
| **analysis** | 이력서/채용공고 분석, 매칭도 | "분석", "매칭", "평가" |
| **chat** | 일반 대화, 취업/진로 상담 | "상담", "어떻게", "추천" |
| **ocr** | 문서 텍스트 추출 | "텍스트 추출", "PDF" |

---

## 5. 구현 계획

### 5.1 Phase 1: 키워드 기반 분기 (V1.5)

```python
def simple_route(message: str) -> str:
    """간단한 키워드 기반 분기"""
    keywords = {
        "interview": ["면접", "질문", "답변 연습", "모의 면접"],
        "analysis": ["분석", "매칭", "평가해", "강점", "약점"],
        "ocr": ["텍스트 추출", "PDF", "이미지에서"]
    }

    for category, words in keywords.items():
        if any(word in message for word in words):
            return category

    return "chat"  # 기본값
```

### 5.2 Phase 2: LLM 기반 의도 분류 (V2)

```python
ROUTER_PROMPT = """
사용자 입력을 분류하세요.

## 카테고리
- interview: 면접 연습, 질문 생성, 답변 연습
- analysis: 이력서/채용공고 분석, 매칭도 계산
- chat: 일반 대화, 취업/진로 상담
- ocr: 문서 텍스트 추출

## 사용자 입력
{user_message}

## 응답 (카테고리만)
"""

async def llm_route(message: str) -> str:
    """LLM 기반 의도 분류"""
    response = await gemini_flash.generate(
        ROUTER_PROMPT.format(user_message=message),
        max_tokens=10
    )
    return response.strip().lower()
```

### 5.3 Phase 3: 멀티스텝 의도 파악 (V3+)

```python
async def smart_route(message: str, context: dict) -> dict:
    """문맥 기반 스마트 라우팅"""

    # 이전 대화 분석
    if context.get("last_mode") == "analysis":
        if "면접" in message or "준비" in message:
            return {
                "mode": "interview",
                "auto_suggest": True,
                "reason": "분석 결과 면접 준비가 필요해 보입니다"
            }

    # LLM 기반 분류
    category = await llm_route(message)

    return {
        "mode": category,
        "auto_suggest": False,
        "reason": None
    }
```

---

## 6. 예상 아키텍처

### 6.1 시스템 구조

```
┌─────────────────────────────────────────────────────┐
│                    API Gateway                       │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│                  Router Service                      │
│  ┌─────────────────────────────────────────────┐   │
│  │  1. 키워드 매칭 (빠름)                        │   │
│  │  2. LLM 분류 (정확함)                        │   │
│  │  3. 문맥 분석 (스마트)                       │   │
│  └─────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
    ┌──────────┐  ┌──────────┐  ┌──────────┐
    │ Interview │  │ Analysis │  │   Chat   │
    │  Service  │  │  Service │  │  Service │
    └──────────┘  └──────────┘  └──────────┘
```

### 6.2 비용 최적화

```
Router 비용 최소화 전략:
1. 키워드 매칭 먼저 시도 (무료)
2. 불확실한 경우만 LLM 호출
3. 경량 모델 사용 (Gemini Flash)
4. 캐싱: 동일 패턴 입력 캐시

예상 추가 비용:
- 월 5,000건 중 30%만 LLM 호출
- 1,500건 × $0.0001 = $0.15/월
```

---

## 참고 자료

- [LangChain - Router Chains](https://python.langchain.com/docs/how_to/routing/)
- [Semantic Router](https://github.com/aurelio-labs/semantic-router)
