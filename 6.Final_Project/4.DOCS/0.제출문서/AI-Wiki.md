# Devths AI/ML Engineering Wiki

### ~~**[Serving URL:](#)**~~ (개발 후 추가 예정)

**AI Repository:** [GitHub Link](https://github.com/100-hours-a-week/9-team-Devths-AI)

---

## 📚 목차

### 프로젝트 개요
> - [[AI] 00. 서비스 시나리오]([AI]-00_서비스_시나리오)
> - [[AI] 01. 버전별 로드맵]([AI]-01_버전별_로드맵)

### 시스템 설계
> - [[AI] 02. 시퀀스 다이어그램]([AI]-02_시퀀스_다이어그램)
> - [[AI] 03. 아키텍처 설계]([AI]-03_아키텍처_설계)
> - [[AI] 09. VectorDB 설계]([AI]-09_VectorDB_설계)

### 설계 과제 (단계 1~8)
> - [[AI] 04. 모델 API 설계]([AI]-04_모델_API_설계) - 단계 1
> - [[AI] 05. 추론 성능 최적화]([AI]-05_추론_성능_최적화) - 단계 2
> - [[AI] 03. 아키텍처 설계]([AI]-03_아키텍처_설계) - 단계 3 (모듈화 포함)
> - [[AI] 06. 멀티스텝 파이프라인]([AI]-06_멀티스텝_파이프라인) - 단계 4
> - [[AI] 07. RAG 컨텍스트 보강]([AI]-07_RAG_컨텍스트_보강) - 단계 5
> - [[AI] 08. 외부 도구 통합]([AI]-08_외부_도구_통합) - 단계 6
> - [[AI] 10. 인프라 확장성 모니터링]([AI]-10_인프라_확장성_모니터링) - 단계 7
> - [[AI] 11. 테스트 전략]([AI]-11_테스트_전략) - 품질 보증
> - [[AI] 12. Pydantic 스키마]([AI]-12_Pydantic_스키마) - Request/Response 검증
> - [[AI] 13. 모델 선정]([AI]-13_모델_선정) - LLM/VLM/Embedding 선정
> - [[AI] 14. Cloud AI]([AI]-14_Cloud_AI) - 클라우드 AI 서비스 전략
> - [[AI] 99. 최종 통합 회고]([AI]-99_최종_통합_회고) - 단계 8

### 기술 의사결정 (ADR)

**[[AI] 00. ADR 001-005]([AI]-00_ADR_-001‐005)** - 기본 아키텍처
<details>
<summary>상세 목차 보기</summary>

- **ADR-001**: LLM 스트리밍 통신 방식 (SSE vs WebSocket)
- **ADR-002**: 분석 결과 저장 형식 (JSON vs Text)
- **ADR-003**: VectorDB 선택 (ChromaDB)
- **ADR-004**: 비동기 처리 방식 (Polling)
- **ADR-005**: 면접 Q&A 저장 전략 (개별 저장)

</details>

**[[AI] 01. ADR 006-010]([AI]-01_ADR_-006‐010)** - 인프라 및 통합
<details>
<summary>상세 목차 보기</summary>

- **ADR-006**: 인프라 진화 전략 (EC2 → Docker → K8s)
- **ADR-007**: Kubernetes vs Kubeflow 도입 시기
- **ADR-008**: OCR + 임베딩 내부 통합 처리
- **ADR-009**: 채팅 컨텍스트 관리 (Redis + LangChain Memory)
- **ADR-010**: OCR 서비스 선택 (Gemini Vision)

</details>

**[[AI] 02. ADR 011-015]([AI]-02_ADR_-011‐015)** - 고급 기능
<details>
<summary>상세 목차 보기</summary>

- **ADR-011**: 서버리스 API 전략 (Lambda 검토)
- **ADR-012**: 캘린더 서비스 차별화 전략
- **ADR-013**: 팀 블로그 / Wiki 운영 방식
- **ADR-014**: LLM 모델 선정 및 Fallback 전략
- **ADR-015**: 배치 처리 전략 (Celery Beat → Airflow)

</details>

**[[AI] 03. ADR 016-020]([AI]-03_ADR_-016‐020)** - 모니터링 및 최적화
<details>
<summary>상세 목차 보기</summary>

- **ADR-016**: LLM 모니터링 도구 (LangSmith vs LangFuse)
- **ADR-017**: FastAPI 서버 분리 전략 (모델 서버 vs 서비스 서버)
- **ADR-018**: 이벤트 스트리밍 (Celery Beat → Kafka)
- **ADR-019**: LLM 출력 구조화 (LangChain + LangGraph)
- **ADR-020**: RAG Reranker 도입 (Cohere Rerank)

</details>

**[[AI] 04. ADR 021-025]([AI]-04_ADR_-021‐025)** - 최적화 ⭐ NEW
<details>
<summary>상세 목차 보기</summary>

- **ADR-021**: Embedding 모델 통일 (Gemini 단일 모델 + BM25 Hybrid)
- **ADR-022**: AI 파이프라인 테스트 (n8n 도입)
- **ADR-023**: 면접 질문 Pool 캐싱 전략 (80% 비용 절감)
- **ADR-024**: VectorDB 필터링 전략
- **ADR-025**: (예정)

</details>

**[[AI] 05. ADR 026-030]([AI]-05_ADR_-026‐030)** - 서버 구조 및 모델 전략
<details>
<summary>상세 목차 보기</summary>

- **ADR-026**: Gunicorn + Uvicorn 서버 구조
- **ADR-027**: AI 페르소나 도입
- **ADR-028**: (예정)
- **ADR-029**: (예정)
- **ADR-030**: VectorDB 선택 (ChromaDB)

</details>

**[[AI] 06. ADR 031-035]([AI]-06_ADR_-031‐035)** - 모델 학습/최적화 및 서빙
<details>
<summary>상세 목차 보기</summary>

- **ADR-031**: 파인튜닝/강화학습 미적용 결정
- **ADR-032**: Gemini API 선택 (LLM + VLM 통합)
- **ADR-033**: Ollama 도입 (면접 모드 한정 - 보안 강화)
- **ADR-034**: 임베딩 모델 전략 (Gemini → BERT)
- **ADR-035**: 종합 모델 전략 (V1-V3)

</details>

**[[AI] 07. ADR 036-040]([AI]-07_ADR_-036‐040)** - PoC/Pilot 기술 선정
<details>
<summary>상세 목차 보기</summary>

- **ADR-036**: OCR vs VLM 선택 전략 (CLOVA + Gemini Vision)
- **ADR-037**: vLLM 로컬 서빙 모델 선정 (EXAONE vs Qwen)
- **ADR-038**: 임베딩 모델 마이그레이션 전략 (Gemini → ko-sroberta)
- **ADR-039**: VectorDB 전략 (ChromaDB 유지 - V3 최적)
- **ADR-040**: 부하 테스트 및 배치 처리 전략

</details>

**[[AI] 08. ADR 041-045]([AI]-08_ADR_-041‐045)** - 인프라 확장 및 하이브리드 스케일링 ⭐ NEW
<details>
<summary>상세 목차 보기</summary>

- **ADR-041**: 하이브리드 스케일링 전략 (RunPod + Lambda)
- **ADR-042**: (예정)
- **ADR-043**: (예정)
- **ADR-044**: (예정)
- **ADR-045**: (예정)

</details>

---

## 🚀 주요 AI 기능

| 기능 | 설명 | 사용 기술 |
|------|------|----------|
| **이력서 분석** | 강점/약점 분석 + 개선점 제안 | LLM + RAG + BM25 Hybrid |
| **채용공고 매칭** | 이력서-채용공고 적합도 분석 | Gemini Embedding + LLM |
| **모의 면접** | AI 면접관 (꼬리질문 + 평가) | LangGraph + Question Pool |
| **캘린더 에이전트** | 자연어로 일정 CRUD | Tool Calling |
| **개인정보 마스킹** | 첨부파일 개인정보 자동 마스킹 | VLM + YOLO |

---

## 🛠 기술 스택

| 구분 | 기술 |
|------|------|
| **Framework** | FastAPI (Python 3.10) |
| **LLM** | Gemini 3 Flash/Pro + GPT-5 mini (Fallback) |
| **Workflow** | LangChain + LangGraph |
| **Embedding** | **Gemini Embedding** (단일 모델, ADR-021) |
| **VectorDB** | ChromaDB |
| **Retriever** | BM25 + Vector Hybrid Search (ADR-021) |
| **Reranker** | Cohere Rerank API (ADR-020) |
| **Cache** | Redis (Session + Question Pool) |
| **OCR** | CLOVA OCR / Gemini Vision |
| **Monitoring** | LangSmith (ADR-016) |
| **Testing** | n8n (ADR-022) |
| **Batch** | Celery Beat (ADR-015) |
| **Infrastructure** | **RunPod (70%) + Lambda (30%)** 하이브리드 (ADR-041) |

---

## 📋 ADR 요약

### ADR 001-010 (기본 아키텍처)
| # | 제목 | 결정 |
|---|------|------|
| 001 | LLM 스트리밍 통신 | SSE 선택 |
| 002 | 분석 결과 저장 | JSON (하이브리드) |
| 003 | VectorDB 선택 | ChromaDB |
| 004 | 비동기 처리 방식 | Polling |
| 005 | 면접 Q&A 저장 | 매 문답 개별 저장 |
| 006 | 인프라 진화 전략 | EC2 → Docker → K8s |
| 007 | K8s 도입 시기 | V3 |
| 008 | OCR + 임베딩 | AI Server 내부 통합 |
| 009 | 채팅 컨텍스트 | Redis + LangChain Memory |
| 010 | OCR 서비스 | Gemini Vision (V1) |

### ADR 011-020 (고급 기능)
| # | 제목 | 결정 |
|---|------|------|
| 011 | 서버리스 API | V3에서 Lambda 검토 |
| 012 | 캘린더 차별화 | V2: 준비 가이드 |
| 013 | Wiki 운영 | GitHub Wiki + 블로그 |
| 014 | LLM 모델 선정 | Gemini 3 Flash + GPT-5 mini Fallback |
| 015 | 배치 처리 전략 | Celery Beat → Airflow 검토 |
| 016 | LLM 모니터링 | LangSmith (V2~) |
| 017 | FastAPI 서버 분리 | V3: 단일 → V4: 모델/서비스 분리 |
| 018 | 이벤트 스트리밍 | Celery Beat (V3) → Kafka 검토 (V4+) |
| 019 | LLM 출력 구조화 | LangChain + LangGraph (V3~) |
| 020 | RAG Reranker | Cohere Rerank (V3~) |

### ADR 021-025 (최적화) ⭐ NEW
| # | 제목 | 결정 |
|---|------|------|
| 021 | **Embedding 모델 통일** | **Gemini 단일 모델 + BM25 Hybrid** |
| 022 | AI 파이프라인 테스트 | n8n 도입 (개발/테스트) |
| 023 | 면접 질문 Pool 캐싱 | Question Pool + RAG Fallback (80% 비용 절감) |
| 024 | VectorDB 필터링 | 메타데이터 필터링 (V3) → RDB Hybrid 검토 (V4) |
| 025 | (예정) | - |

### ADR 041-045 (인프라 확장) ⭐ NEW
| # | 제목 | 결정 |
|---|------|------|
| 041 | **하이브리드 스케일링** | **RunPod (70%) + Lambda (30%)** - 24% 비용 절감 |
| 042 | (예정) | - |
| 043 | (예정) | - |
| 044 | (예정) | - |
| 045 | (예정) | - |

---

## 🎯 V3 핵심 기술 결정

### Embedding 전략 (ADR-021)
- ✅ **Gemini Embedding 단일 모델** (벡터 공간 통일)
- ✅ **BM25 + Vector Hybrid Search** (정확도 10~15% 향상)
- ❌ SBERT 자체 서빙 제외 (호환성 문제)

### RAG 최적화 (ADR-020, ADR-021)
```
BM25 (키워드 30%) + Gemini Embedding (의미 70%)
  ↓
Hybrid Search (20개)
  ↓
Cohere Reranker (3개)
  ↓
LLM
```

### 비용 절감 (ADR-023)
- ✅ **Question Pool + RAG Fallback**
- ✅ 캐시 히트율 80% → 비용 80% 절감 ($7.5 → $1.5/월)

---

## 📞 팀 정보

- **팀명:** Devths (9팀)
- **프로젝트:** 카카오테크 부트캠프 3기
