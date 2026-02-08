# SSE 스트리밍 오류 처리 이력 (AI)

9-team-Devths-AI 저장소에서 채팅 SSE 스트리밍·리포트 모드·504/422 대응을 위해 진행한 작업을 **전체 개발 과정(v1 → v2)** 순서로 정리한 문서입니다.

---

# Part 1. 
## v1 시대 — 스키마·422·504·SSE 기반 구축


## 1. API 스키마·Swagger 보강

### 1.1 ChatRequest Swagger 예시 및 면접 리포트 예시 추가

| 항목 | 내용 |
|------|------|
| **커밋** | [108cbc79](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/108cbc79c225bf17422194f53e50a0675bd961d5) |
| **메시지** | ChatRequest Swagger 예시에 model 필드 추가 및 면접 리포트 예시 추가 |

**처리 내용**
- Swagger(OpenAPI) 문서에서 `ChatRequest` 예시에 `model` 필드를 추가해 클라이언트가 어떤 값을 보내야 하는지 명확히 함.
- 면접 리포트 모드 사용 예시를 문서에 추가해, 리포트 모드 호출 시 요청 형식을 안내.

**목적**
- API 스키마 불일치로 인한 422 Validation Error를 예방하기 위한 문서·예시 정리.

---

## 2. 리포트 모드 및 채팅 API 스키마 리팩터

### 2.1 리포트 모드 추가 및 채팅 API 스키마 정리

| 항목 | 내용 |
|------|------|
| **커밋** | [3c92d4bb](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/3c92d4bbaa2e26ce295befc72b8886584bb84553) |
| **메시지** | Add report mode and refactor chat API schema |

**처리 내용**
- 면접 **리포트 모드** 기능 추가: 스트리밍 채팅과 구분된 리포트 생성 플로우 구현.
- 채팅 API 스키마 리팩터: 리포트 모드에 맞는 요청/응답 구조 정리 및 스키마 통일.

**목적**
- 면접 종료 후 한 번에 리포트를 받는 use case 지원 및 스키마 일관성 확보.

---

## 3. 422 검증 오류 대응

### 3.1 로깅 개선 및 422 검증 오류 처리 추가

| 항목 | 내용 |
|------|------|
| **커밋** | [fc88d09b](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/fc88d09b2f59704e71990ebea7bb38b7640039e5) |
| **메시지** | 로깅 개선 및 422 검증 오류 처리 추가 |

**처리 내용**
- 422 Unprocessable Entity(요청 바디 검증 실패) 발생 시 로그를 보강해 원인 파악이 쉽도록 함.
- 422 상황에 대한 서버 측 처리(예: 예외 핸들링·에러 메시지 정리) 추가.

**목적**
- 리포트 모드·일반 채팅 요청 시 `model` 등 필드 누락/형식 오류로 422가 나는 문제를 진단·대응.

### 3.2 면접 리포트 모드 422 검증 오류 처리

| 항목 | 내용 |
|------|------|
| **커밋** | [b38257a1](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/b38257a1df672540787510c277bd27e0b38c5aed) |
| **메시지** | Handle 422 validation error for interview report mode |

**처리 내용**
- **면접 리포트 모드** 전용 422 처리 로직 추가.
- 리포트 모드 요청에서 필수 필드 누락·타입 불일치 시 적절한 에러 응답과 메시지 반환.
- 클라이언트가 수정할 수 있도록 어떤 필드에서 검증에 실패했는지 명시.

**목적**
- 리포트 모드 호출 시 422가 나는 경우를 줄이고, 나더라도 원인을 빠르게 파악할 수 있도록 함.

---

## 4. 504 타임아웃 방지 (로깅·SSE Heartbeat)

### 4.1 로깅 개선 및 SSE Heartbeat로 504 방지

| 항목 | 내용 |
|------|------|
| **커밋** | [4a06f0e2](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/4a06f0e2c497a94ba4cd49819e7e367ca0a8460d) |
| **메시지** | Improve logging and add SSE heartbeat for 504 timeout prevention |

**처리 내용**
- **SSE heartbeat** 도입: 스트리밍 중 오래 걸리는 LLM 응답 구간에서도 주기적으로 빈 데이터(또는 heartbeat 전용 이벤트)를 보내 연결이 유지되도록 함.
- 로깅 개선: 스트리밍 구간·에러 구간에서 로그를 추가해 504 발생 직전 상황을 추적 가능하게 함.

**목적**
- 프록시·로드밸런서·클라이언트에서 **일정 시간 데이터가 없으면 연결을 끊어 504 Gateway Timeout**이 나는 문제를 완화.

---

## 5. SSE 포맷 표준화 및 ChromaDB None 필터

### 5.1 SSE 포맷 통일 및 ChromaDB None 필터링

| 항목 | 내용 |
|------|------|
| **커밋** | [08916a45](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/08916a45e1051be841c50eef101d648afd81d14e) |
| **메시지** | Standardize SSE format and add ChromaDB None filtering |

**처리 내용**
- **SSE 포맷 표준화**: `event`, `data`, 주석 라인 등 SSE 스펙에 맞게 형식을 통일해 다양한 클라이언트·프록시에서 안정적으로 파싱되도록 함.
- **ChromaDB None 필터링**: RAG/벡터 검색 시 `metadata` 등에 `None`이 들어가 ChromaDB 쿼리 오류가 나는 경우를 방지하기 위해, 쿼리 전에 `None` 값을 제거하거나 필터 조건에서 제외하는 처리 추가.

**목적**
- SSE 파싱 오류 및 ChromaDB 예외로 인한 스트리밍 중단을 줄이기 위함.

---

## 6. 채팅 엔드포인트 SSE 헤더 추가

### 6.1 채팅 엔드포인트에 SSE 헤더 추가

| 항목 | 내용 |
|------|------|
| **커밋** | [9b712c3b](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/9b712c3bb6b52203cb5383008b312d4789a0dfc0) |
| **메시지** | Add SSE headers to chat endpoint |

**처리 내용**
- 채팅 스트리밍 응답에 **SSE용 HTTP 헤더** 명시적 설정:
  - `Content-Type: text/event-stream`
  - `Cache-Control: no-cache`
  - `Connection: keep-alive`
  - 필요 시 `X-Accel-Buffering: no` (Nginx 버퍼링 비활성화) 등.
- 클라이언트와 중간 프록시가 응답을 SSE 스트림으로 올바르게 인식하도록 함.

**목적**
- 브라우저·프록시가 응답을 일반 JSON으로 취급하거나 버퍼링해 스트리밍이 깨지는 현상을 방지.

---

# Part 2. 
## v1 후반 — 면접·분석·SSE 에러·보안 보강


## 7. 채팅방 제목 추출·분석 포맷 개선

| 항목 | 내용 |
|------|------|
| **관련 커밋** | d1502fc (채팅방 제목 자동 추출), c41277c (파일 업로드 API에 채팅방 제목 추출), 35f899c (프롬프트 응답 형식 개선), 18ef830 (formatted_text 필드), a60589f (분석 결과 포맷 개선), 2a6a988 (면접 5회 완료 후 자동 피드백) 등 |
| **처리 내용** | 채팅방 제목 자동 추출(회사명/채용직무), 분석 결과 포맷 개선, 면접 5회 완료 후 피드백 생성 등. |
| **목적** | UX·응답 일관성 개선. |

---

## 8. 면접 질문·꼬리질문·JSON 파싱·예외 처리

| 항목 | 내용 |
|------|------|
| **관련 커밋** | de43876 (기술면접 프롬프트 템플릿), e8f1aea (면접 질문 JSON 파싱 로직 개선), 851a5fb (면접 질문 생성 예외 처리 개선), 9c2dfe8 (꼬리질문 평가 로직), 5b64cea (JSON 파싱 실패 수정), 098ca9d (면접 질문 예시 프롬프팅) 등 |
| **처리 내용** | 면접 질문 세트 JSON 파싱·예외 처리 보강, 꼬리질문 평가 로직, 기술면접 프롬프트 템플릿, 질문 예시 프롬프팅 추가. |
| **목적** | 면접 플로우 안정화 및 스트리밍 중 파싱 오류 감소. |

---

## 9. SSE 에러 응답 구조 개선 (v1)

| 항목 | 내용 |
|------|------|
| **커밋** | 728cba6 |
| **메시지** | refactor: SSE 에러 응답 구조 개선 |
| **처리 내용** | v1 채팅 스트리밍 구간에서 SSE 에러 응답 형식을 정리해 클라이언트/BE가 에러를 일관되게 파싱할 수 있도록 함. |
| **목적** | 스트림 내 에러 전달 방식 통일의 기반. |

---

## 10. Gemini 빈 응답·분석 API 안정화

| 항목 | 내용 |
|------|------|
| **관련 커밋** | 4af8155 (Gemini 빈 응답 해결, 분석 API 단계별 분할), 03f1eaf (안정 모델·재시도), 1ff6498 (빈 응답 시 None 체크), 93e0033 (분석 결과 JSON 파싱, Gemini JSON 모드), 414ed2d (분석 실패 시 fallback·빈 맥락 방지) 등 |
| **처리 내용** | 분석 API 단계별 분할 호출, 재시도·fallback, JSON 파싱 실패 대응, 빈 응답 None 체크. |
| **목적** | 분석·면접 구간에서 LLM 불안정으로 인한 스트리밍 중단·오류 감소. |

---

## 11. 세션 캐시 및 비스트리밍 LLM 응답

| 항목 | 내용 |
|------|------|
| **커밋** | 8ffda2f |
| **메시지** | feat: 세션 캐시 및 비스트리밍 LLM 응답 추가 |
| **처리 내용** | 면접 세션 캐시 도입, 질문 세트 생성 등 구간에 비스트리밍(non-stream) LLM 호출 적용. |
| **목적** | 면접 초기화·질문 생성 구간 안정화 및 응답 형식 일관성. |

---

## 12. 로그 인젝션(Log Injection) 방어 (v1)

| 항목 | 내용 |
|------|------|
| **관련 커밋** | 2324d5a (sanitize_log_input 적용·포맷), 483d119 (safe_info 래퍼), 7c7299b (요청 모델명 sanitize), 77006e7 (Log Injection 수정·포맷팅), 737a597, ba7e63d, 51816c7, 13893d0 (CodeQL Log Injection 대응) 등 |
| **처리 내용** | 로그에 기록되는 사용자 입력·모델명 등에 **sanitize** 적용(개행·제어 문자 제거). `safe_info` 등 래퍼로 CRLF Injection 방어. |
| **목적** | 로그 인젝션·개인정보 노출 위험 완화 및 코드 스캔 알림 대응. |

---

## 13. 면접 기능 개선 및 보안 강화 (프롬프트 인젝션)

| 항목 | 내용 |
|------|------|
| **관련 커밋** | 2339c52 (면접 기능 개선 - 타이핑 효과, 비스트리밍 응답, 보안 취약점 수정), ce21bed (면접 기능 개선 및 보안 강화 - 프롬프트/로그 인젝션 방어), 19d5537 (프롬프트 인젝션 로깅에 safe_warning 적용) |
| **처리 내용** | **프롬프트 인젝션** 검사 도입: `check_prompt_injection()`. BLOCK 시 요청 거부·에러 응답, WARNING 시 로깅 후 진행. 타이핑 효과·비스트리밍 응답 보강. |
| **목적** | 시스템 프롬프트 탈취·역할 변경·인젝션 명령어 등 위험 입력 차단. |

---

## 14. 422 전역 핸들러 (main.py)

| 항목 | 내용 |
|------|------|
| **위치** | `app/main.py` |
| **처리 내용** | `RequestValidationError` 전역 `exception_handler` 등록. 422 발생 시 요청 URL, Method, Body(최대 2000자), Errors(detail) 로깅 후 `JSONResponse(status_code=422, content={"detail": exc.errors()})` 반환. |
| **목적** | 422 원인(필드 누락·타입 불일치) 디버깅 용이. (v1 시절 로깅 개선·422 처리의 전역 정리) |

---

# Part 3. 
## v2 전환 — 아키텍처·라우트 분리·SSE 에러 통일


## 15. 아키텍처 모듈화 (DI·인프라 분리)

| 항목 | 내용 |
|------|------|
| **커밋** | 43d8043 |
| **메시지** | feat: 아키텍처 모듈화 - DI 기반 인프라 및 도메인 분리 |
| **처리 내용** | DI 기반 인프라·도메인 계층 분리, 라우트·서비스 구조 정리. v2 라우트 분리의 기반. |
| **목적** | 유지보수·테스트·SSE/채팅 로직 확장 용이. |

---

## 16. API routes v1/v2 분리 및 모듈화

| 항목 | 내용 |
|------|------|
| **커밋** | 017e20a |
| **메시지** | feat: API routes v1/v2 분리 및 모듈화 |
| **처리 내용** | 채팅·마스킹·텍스트 추출 등 API를 **v1(하위 호환)** / **v2(현재)** 로 분리. v2 채팅은 `POST /ai/chat` 에서 스트리밍(SSE) 제공. v1은 `POST /ai/v1/chat` 등으로 유지. |
| **목적** | 스키마·동작 정리 및 SSE 엔드포인트 일원화. |

---

## 17. v2 채팅 SSE 헤더

| 항목 | 내용 |
|------|------|
| **위치** | `app/api/routes/v2/chat.py` — `StreamingResponse` |
| **처리 내용** | `media_type="text/event-stream"`, `Cache-Control: no-cache`, `Connection: keep-alive`, `X-Accel-Buffering: no` 명시. (v1 §6과 동일 정책을 v2에 적용) |
| **목적** | Nginx 등 프록시 버퍼링 비활성화 및 스트리밍 인식 안정화. |

---

## 18. SSE 에러 이벤트 통일 포맷 (v2)

| 항목 | 내용 |
|------|------|
| **위치** | `app/api/routes/v2/_sse_errors.py` |
| **처리 내용** | **백엔드(Spring)가 SSE 스트리밍 중 에러를 인식**할 수 있도록, 모든 에러를 통일된 JSON 포맷으로 전송하는 `sse_error_event(code, status, message, fallback)` 도입. |
| **SSE 이벤트 타입** | `chunk`(정상 텍스트), `summary`(채팅방 제목), `session_state`(면접 세션), **`error`**(에러 발생), `[DONE]`(스트림 종료). |
| **에러 payload** | `type: "error"`, `error: { code, status, message }`, `fallback`(사용자 표시용). |
| **목적** | BE·프론트가 에러 시 동일한 구조로 파싱하여 처리·UX 일관성 확보. |

---

## 19. v2 채팅 엔드포인트 예외 처리 (SSE 에러 코드)

| 에러 코드 | HTTP status | 발생 상황 | fallback 예시 |
|-----------|-------------|-----------|----------------|
| **PROMPT_BLOCKED** | 400 | 프롬프트 인젝션 차단(RiskLevel.BLOCK) | 프롬프트 인젝션 관련 안내 |
| **VECTORDB_ERROR** | 404 | VectorDB에 문서 없음(분석 모드) | 업로드된 이력서/채용공고를 찾을 수 없습니다 |
| **INTERNAL_ERROR** | 500 | 일반 채팅·면접 진행 중 미처리 예외 | 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해주세요. |
| **PARSE_FAILED** | 500 | 면접 질문 세트·꼬리질문 JSON 파싱 실패 | 질문 세트/꼬리질문 생성 중 오류 안내 |
| **SESSION_NOT_FOUND** | 404 | 면접 세션에서 현재 질문 ID를 찾을 수 없음 | 세션 오류: 현재 질문을 찾을 수 없습니다. |
| **LLM_ERROR** | 500 | 면접 리포트 생성 중 LLM 예외 | 면접 리포트 생성 중 오류가 발생했습니다. 잠시 후 다시 시도해주세요. |

- 모든 경우 **HTTP 200 유지**, 본문에 `data: {"type":"error", ...}\n\n` 전송 후 필요 시 `data: [DONE]\n\n`으로 스트림 종료.

---

## 20. 예외 처리 개선 및 SSE 에러 핸들링 추가

| 항목 | 내용 |
|------|------|
| **커밋** | 5631976 |
| **메시지** | fix: 예외 처리 개선 및 SSE 에러 핸들링 추가 |
| **처리 내용** | v2 `generate_chat_stream()` 내 일반 대화·분석·면접·리포트 구간에서 예외 발생 시 `sse_error_event()`로 에러 이벤트 전송 후 `[DONE]`으로 종료. 스트리밍 중 예외가 나도 연결을 끊지 않고 클라이언트/BE가 에러 payload만으로 처리 가능하도록 정리. |
| **목적** | 스트리밍 중 예외 발생 시에도 HTTP 200 + error 이벤트로 안정적으로 전달. |

---

## 21. 로그 인젝션 추가 수정 (v2·마스킹 등)

| 항목 | 내용 |
|------|------|
| **관련 커밋** | 615960f (task_id 로깅 시 sanitize), 6fbd074 (v1/masking.py Log Injection 취약점 해결), 26aa75a, be738c0 (CodeQL alert 135·136 Log Injection), 32b1530 (security: Fix Log Injection with sanitize function) |
| **위치** | `app/utils/log_sanitizer.py`, 채팅·마스킹 등 로깅 구간 |
| **처리 내용** | task_id·마스킹 등 추가 구간에 sanitize 적용. `safe_info`, `safe_warning` 등으로 CRLF Injection 방어 강화. |
| **목적** | v2·마스킹 포함 전 구간 로그 인젝션·코드 스캔 대응. |

---

# Part 4.
## v2 확장 — 면접 평가 API 및 토론 시스템

## 22. 면접 답변 평가 API (LLM-as-a-Judge)

| 항목 | 내용 |
|------|------|
| **커밋** | 27c62ba |
| **메시지** | feat: Add interview evaluation API (analyze + debate) |
| **처리 내용** | 면접 종료 후 답변 평가를 위한 2단계 시스템 구현. 1단계: Gemini 3 Pro (thinking 모드)로 각 Q&A를 4개 기준(관련성·구체성·논리성·STAR 기법)으로 평가. 2단계: 사용자 수동 트리거로 GPT-4o가 독립 분석 후 Gemini와 토론하여 합의 도출. |
| **주요 파일** | `app/domain/evaluation/analyzer.py`, `app/domain/evaluation/debate_graph.py`, `app/api/routes/v2/evaluation.py`, `app/schemas/evaluation.py` |
| **목적** | 면접 답변에 대한 심층 피드백 제공 및 LangGraph 기반 토론 패턴 도입. |

---

## 23. Evaluation 모듈 lint·format 정리

| 항목 | 내용 |
|------|------|
| **관련 커밋** | cbdc5e2 (ruff lint 에러 해결), f0f3f19 (ruff format 적용) |
| **처리 내용** | evaluation 모듈 전체에 ruff lint·format 적용. import 정리, 타입 힌트 수정, 코드 스타일 통일. |
| **목적** | CI 파이프라인 통과 및 코드 일관성 확보. |

---

## 24. 면접 질문 생성 속도 개선

| 항목 | 내용 |
|------|------|
| **커밋** | 8d36874 |
| **메시지** | feat: 면접 질문 생성 속도 개선 및 타이핑 효과 추가 |
| **처리 내용** | 면접 질문 생성 시 LLM 호출 최적화로 응답 속도 개선. 클라이언트에 타이핑 효과용 청크 전송 추가. |
| **목적** | 면접 질문 생성 대기 시간 단축 및 UX 개선. |

---

## 25. 면접 질문 생성 실패 시 예외 처리 강화

| 항목 | 내용 |
|------|------|
| **관련 커밋** | 851a5fb (예외 처리 개선), 68212a0 (질문 길이 제한), 5b64cea (JSON 파싱 실패 수정) |
| **처리 내용** | 면접 질문 JSON 파싱 실패 시 SSE 에러 이벤트(`PARSE_FAILED`)로 안정 전달. 질문 길이 제한 추가로 LLM 과다 응답 방지. |
| **목적** | 면접 진행 중 파싱 오류로 인한 스트리밍 중단 방지. |

---

## 26. 면접 질문 반복 방지 및 데이터셋 통합

| 항목 | 내용 |
|------|------|
| **관련 커밋** | 8e94ab4 (질문 반복 방지 + 서비스명 변경), aabf9c7 (면접 데이터셋 통합), 098ca9d (질문 예시 프롬프팅) |
| **처리 내용** | 이미 출제된 질문을 프롬프트에 포함시켜 중복 방지. InterView_Datasets 통합으로 질문 품질 향상. |
| **목적** | 면접 5회 진행 중 동일 질문 반복 문제 해결. |

---

## 27. Gemini 모델 안정화 (빈 응답·재시도·fallback)

| 항목 | 내용 |
|------|------|
| **관련 커밋** | 4af8155 (빈 응답 해결, 단계별 분할), 1372b56 (gemini-3-flash-preview 모델명 수정), 03f1eaf (안정 모델·재시도), 414ed2d (fallback·빈 맥락 방지) |
| **처리 내용** | Gemini 모델 빈 응답 시 None 체크 및 재시도. 분석 API를 단계별 분할 호출로 변경. 모델명 오류 수정. |
| **목적** | LLM 불안정으로 인한 면접·분석 구간 중단 최소화. |

---

## 요약: 오류 유형별 대응

| 오류/현상 | 대응 단계 | 조치 요약 |
|-----------|-----------|-----------|
| **422 Validation Error** | §2·§3, §14 | Swagger 예시·스키마 정리, 리포트 모드 전용 처리, main.py 전역 핸들러(Body/Errors 로깅) |
| **504 Timeout** | §4 | 로깅 개선, SSE heartbeat로 유휴 구간에도 데이터 전송 |
| **SSE 파싱/연결** | §5, §6, §17 | SSE 포맷 표준화, SSE 헤더 명시(v1·v2), ChromaDB None 필터 |
| **SSE 스트림 내 에러** | §9, §18~§20 | v1 SSE 에러 구조 개선 → v2 통일 포맷(sse_error_event)·에러 코드별 fallback |
| **프롬프트 인젝션** | §13, §19 | BLOCK 시 400 + PROMPT_BLOCKED SSE(v2), WARNING 시 로깅 |
| **로그 인젝션** | §12, §21 | sanitize_log_input, safe_info/safe_warning (v1·v2·마스킹) |
| **면접 질문 파싱 실패** | §8, §25 | JSON 파싱 예외 처리, 질문 길이 제한, PARSE_FAILED SSE 에러 |
| **Gemini 빈 응답** | §10, §27 | 단계별 분할 호출, 재시도·fallback, None 체크 |
| **면접 답변 평가** | §22 | LLM-as-a-Judge (Gemini 3 Pro), LangGraph 토론 시스템 |

---

## 문서 이력

| 날짜 | 내용 |
|------|------|
| (최초 작성) | v1 SSE·422/504·ChromaDB·헤더 등 커밋 정리 |
| 2026-02-08 | v1부터 v2까지 전체 개발 과정(커밋 처음~끝) 흐름으로 재작성, 머지 섹션 제외 |
| 2026-02-08 | Part 4 추가: 면접 평가 API(analyze+debate), 질문 생성 속도 개선, 예외 처리 강화, Gemini 안정화 |
