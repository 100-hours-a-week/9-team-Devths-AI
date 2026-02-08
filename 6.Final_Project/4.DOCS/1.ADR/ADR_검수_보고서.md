# ADR 전체 문서 검수 보고서

## 📊 검수 개요

**검수 일시**: 2026-01-13  
**검수 대상**: ADR 001-027 (6개 파일)  
**검수자**: AI팀

---

## ✅ 검수 결과 요약

| 파일 | ADR 범위 | 상태 | 비고 |
|------|---------|------|------|
| `[AI] 00_ADR_ 001-005.md` | ADR-001 ~ ADR-005 | ✅ 완료 | SSE, JSON, ChromaDB, Polling, 면접 저장 |
| `[AI] 01_ADR_ 006-010.md` | ADR-006 ~ ADR-010 | ✅ 완료 | 인프라, K8s, OCR+임베딩, Redis, OCR 서비스 |
| `[AI] 02_ADR_ 011-015.md` | ADR-011 ~ ADR-015 | ✅ 완료 | 서버리스, 캘린더, Wiki, LLM, 배치 (Kubeflow 제외 추가) |
| `[AI] 03_ADR_ 016-020.md` | ADR-016 ~ ADR-020 | ✅ 완료 | LangSmith, FastAPI 분리, Kafka, LangChain, Reranker |
| `[AI] 04_ADR_ 021-025.md` | ADR-021 ~ ADR-025 | ✅ 완료 | Embedding 통일, n8n, Question Pool, VectorDB 필터링, RDB 동기화 |
| `[AI] 05_ADR_ 026-030.md` | ADR-026 ~ ADR-029 | ✅ 완료 | Gunicorn+Uvicorn, AI 페르소나, API 통합, LangChain |

---

## 📋 ADR 전체 목록 (001-029)

### **ADR 001-005** (기본 아키텍처)
- ✅ **ADR-001**: LLM 스트리밍 (SSE vs WebSocket) → **SSE 선택**
- ✅ **ADR-002**: 분석 결과 저장 (JSON vs Text) → **JSON + 하이브리드**
- ✅ **ADR-003**: VectorDB 선택 (ChromaDB vs Pinecone) → **ChromaDB**
- ✅ **ADR-004**: 비동기 처리 (Polling vs Callback) → **Polling**
- ✅ **ADR-005**: 면접 Q&A 저장 → **개별 저장**

### **ADR 006-010** (인프라 & 서비스)
- ✅ **ADR-006**: 인프라 진화 전략 → **EC2 → Docker → K8s**
- ✅ **ADR-007**: K8s 도입 시기 → **V3**
- ✅ **ADR-008**: OCR + 임베딩 통합 → **AI Server 내부**
- ✅ **ADR-009**: 채팅 컨텍스트 관리 → **Redis + LangChain Memory**
- ✅ **ADR-010**: OCR 서비스 선택 → **Gemini Vision (V1)**

### **ADR 011-015** (고급 기능)
- ✅ **ADR-011**: 서버리스 API → **V3 Lambda 검토**
- ✅ **ADR-012**: 캘린더 차별화 → **V2 준비 가이드**
- ✅ **ADR-013**: Wiki 운영 → **GitHub Wiki + 블로그**
- ✅ **ADR-014**: LLM 모델 선정 → **Gemini 3 Flash + GPT-5 mini Fallback**
- ✅ **ADR-015**: 배치 처리 → **Celery Beat (V3) → Airflow (V4) / Kubeflow 제외**

### **ADR 016-020** (MLOps & 최적화)
- ✅ **ADR-016**: LLM 모니터링 → **LangSmith (V2~)**
- ✅ **ADR-017**: FastAPI 서버 분리 → **V3 단일 → V4 모델/서비스 분리**
- ✅ **ADR-018**: 이벤트 스트리밍 → **Celery Beat (V3) → Kafka 검토 (V4+)**
- ✅ **ADR-019**: LLM 출력 구조화 → **LangChain 미사용 (비용), LangChain + LangGraph 사용**
- ✅ **ADR-020**: RAG Reranker → **Cohere Rerank (V3~)**

### **ADR 021-025** (RAG 최적화)
- ✅ **ADR-021**: Embedding 모델 통일 → **Gemini 단일 모델 + BM25 Hybrid**
- ✅ **ADR-022**: AI 파이프라인 테스트 → **n8n (개발/테스트)**
- ✅ **ADR-023**: 면접 질문 Pool 캐싱 → **Question Pool + RAG Fallback (80% 비용 절감)**
- ✅ **ADR-024**: VectorDB 필터링 → **메타데이터 필터링 (V3) → RDB Hybrid 검토 (V4)**
- ✅ **ADR-025**: RDB ↔ VectorDB 동기화 → **실시간 (V1-V2) → 배치 (V3) → CDC (V4+)**

### **ADR 026-030** (서버 & UX & API)
- ✅ **ADR-026**: Gunicorn + Uvicorn 서버 구조 → **V3 멀티프로세스 → V4 추론 서버 분리**
- ✅ **ADR-027**: AI 페르소나 도입 → **상황별 페르소나 (커리어 코치, 면접관, 캘린더)**
- ✅ **ADR-028**: API 통합 전략 → **텍스트 추출 API 통합 (5개 → 4개)**
- ✅ **ADR-029**: LangChain PydanticOutputParser → **LLM 출력 자동 구조화 (Pydantic 모델 활용)**
- ⚠️ **ADR-030**: (예정)

---

## 🔍 주요 검수 항목

### **1. 일관성 검사**

#### ✅ **메타데이터 형식 일관성**
```markdown
### 📋 메타데이터

| 항목 | 내용 |
|------|------|
| **상태** | ✅ 승인됨 (Accepted) |
| **작성일** | 2026-01-XX |
| **결정자** | AI팀 / 개발팀 |
| **관련 기능** | ... |
```
- ✅ 모든 ADR이 동일한 형식 사용

#### ✅ **섹션 구조 일관성**
```markdown
1. 메타데이터
2. 컨텍스트 (Context)
3. 선택지 분석 (Options)
4. 결정 (Decision)
5. 근거 (Rationale)
6. 트레이드오프 (Trade-offs) - 선택적
7. 이력
```
- ✅ 모든 ADR이 동일한 섹션 구조 사용

---

### **2. 내용 검증**

#### ✅ **기술 결정 명확성**
- ✅ 각 ADR이 명확한 기술 선택 제시
- ✅ 선택 근거가 구체적으로 작성됨
- ✅ 코드 예시 포함

#### ✅ **버전별 전략 일관성**
```
V1~V2: MVP, 간단한 구현
V3: 프로덕션 최적화
V4: 고도화, 분리
V5+: 자체 모델 학습 (Kubeflow 검토)
```
- ✅ 모든 ADR이 버전별 전략 일관성 유지

---

### **3. 최근 업데이트 반영**

#### ✅ **ADR-015: Kubeflow 제외 이유 추가**
```
- Kubeflow 옵션 추가
- 현재 프로젝트에서 제외하는 이유 명확화
- 모델 학습 없음 강조
```

#### ✅ **ADR-021: Embedding 모델 통일**
```
- SBERT 제거
- Gemini Embedding 단일 모델
- BM25 Hybrid Search 추가
```

#### ✅ **ADR-025: RDB ↔ VectorDB 동기화**
```
- 실시간 vs 배치 동기화 전략
- sync_status 필드 관리
- 30분마다 배치 동기화
```

#### ✅ **ADR-026: Gunicorn + Uvicorn**
```
- 멀티프로세스 서버 구조
- 추론 서버 분리 (V4)
- Worker 수 계산 공식
```

#### ✅ **ADR-027: AI 페르소나**
```
- 상황별 페르소나 정의
- 커리어 코치, 면접관, 캘린더 어시스턴트
- PersonaManager 클래스
```

---

## 🎯 핵심 기술 결정 요약

### **데이터 & 저장**
- ✅ **VectorDB**: ChromaDB (자체 호스팅)
- ✅ **Embedding**: Gemini 단일 모델 + BM25 Hybrid
- ✅ **Reranker**: Cohere Rerank (V3)
- ✅ **동기화**: 배치 (30분마다, V3)

### **서버 & 인프라**
- ✅ **서버**: Gunicorn + Uvicorn (멀티프로세스)
- ✅ **배치**: Celery Beat (V3) → Airflow (V4)
- ✅ **모니터링**: LangSmith (V2~)
- ✅ **K8s**: V3 도입

### **LLM & AI**
- ✅ **LLM**: Gemini 3 Flash + GPT-5 mini Fallback
- ✅ **스트리밍**: SSE
- ✅ **출력 구조화**: LangChain + LangGraph
- ✅ **페르소나**: 상황별 (커리어 코치, 면접관, 캘린더)

### **최적화**
- ✅ **Question Pool**: 캐시 히트율 80% (비용 80% 절감)
- ✅ **메타데이터 필터링**: VectorDB 네이티브
- ✅ **테스트**: n8n (개발/테스트)

---

## ⚠️ 주의사항

### **1. Kubeflow 제외 (중요!)**
```
현재 프로젝트:
- LLM: Gemini API (학습 안 함)
- Embedding: Gemini API (학습 안 함)
- YOLO: 사전 학습된 모델

→ Kubeflow 불필요!
→ V5+ 자체 모델 학습 시작 시 검토
```

### **2. SBERT 제거 (중요!)**
```
Before: SBERT (자체 서빙)
After: Gemini Embedding (단일 모델)

이유: 벡터 공간 호환성 문제
```

### **3. 배치 처리 vs 모델 학습**
```
Celery Beat / Airflow: 데이터 처리
Kubeflow: 모델 학습

현재는 데이터 처리만 필요!
```

---

## 💡 권장 사항

### **1. ADR-028~030 작성 제안**
- **ADR-028**: 프롬프트 버전 관리 전략
- **ADR-029**: 에러 핸들링 & 재시도 전략
- **ADR-030**: 비용 최적화 전략

### **2. 문서 업데이트**
- ✅ AI-Wiki.md에 ADR-026, ADR-027 반영 필요
- ✅ 아키텍처 다이어그램 업데이트 (Gunicorn+Uvicorn, 페르소나)

### **3. 코드 구현**
- ⚠️ PersonaManager 클래스 구현
- ⚠️ Gunicorn + Uvicorn 설정
- ⚠️ RDB ↔ VectorDB 동기화 배치 작업

---

## ✅ 최종 검수 결과

**전체 ADR 문서 상태: 양호 ✅**

- ✅ 일관성: 모든 ADR이 동일한 형식 사용
- ✅ 명확성: 기술 결정이 명확하게 작성됨
- ✅ 최신성: 최근 변경사항 반영 (Kubeflow 제외, Embedding 통일 등)
- ✅ 완성도: ADR-001 ~ ADR-027 완성

**다음 단계:**
1. AI-Wiki.md 업데이트 (ADR-026, ADR-027 반영)
2. ADR-028~030 작성 검토
3. 코드 구현 시작

---

**검수 완료일**: 2026-01-13  
**검수자**: AI팀
