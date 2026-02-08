# SSE 스트리밍 구현 및 오류 처리 이력 (AI + BE + FE)

9-team-Devths 프로젝트에서 **SSE 스트리밍 구현 및 오류 처리**와 관련하여 AI·BE·FE 3개 저장소에서 진행한 작업을 **시간 흐름순**으로 정리한 문서입니다.

> **핵심 이슈:** SSE 연결 끊김 문제 및 재연결 전략, 스트리밍 응답 파싱 및 렌더링 방식, 에러 발생 시 사용자 경험 처리

- **AI 저장소**: [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI)
- **BE 저장소**: [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE)
- **FE 저장소**: [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE)

---

# Part 1.
## 채팅 API 기반 구축 및 SSE 스트리밍 적용

## 1. AI 채팅방·메시지 API 구현 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [a203310](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/a203310), [73c1c24](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/73c1c24), [976dd8f](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/976dd8f), [c26c8aa](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/c26c8aa) |
| **메시지** | feat: AI 채팅방 목록 조회 / 생성 / 삭제 / 채팅 내용 불러오기 API 구현 (#28) |

**처리 내용**
- AI 채팅방 목록 조회·생성·삭제 API 및 AI 채팅 내용 불러오기 API 구현.
- 채팅방·메시지 도메인·리포지토리·서비스·컨트롤러 구성으로 이후 **챗봇 SSE 스트리밍** 연동의 기반 마련.

**목적**
- 프론트·AI 서버(FastAPI)와 연동할 채팅 API의 기본 CRUD와 데이터 모델 확보.

---

## 2. 공통 fetch 클라이언트 및 응답 타입 정의 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **커밋** | [db0733d](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/db0733d) (공통 fetch 클라이언트), [8da4d08](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/8da4d08) (응답 타입 정의) |

**처리 내용**
- BE API 호출용 **공통 fetch 클라이언트** 구현: `apiRequest`, `api.get` / `api.post` 등 메서드 제공.
- `NEXT_PUBLIC_API_BASE_URL` 기반 URL 구성, 인증 토큰(`Authorization`), `credentials: 'include'` 지원.
- **ApiResponse&lt;T&gt;**, **ApiErrorResponse** 타입 정의로 성공/에러 응답 구분.

**목적**
- 채팅·분석·SSE 엔드포인트를 제외한 일반 API 요청의 일관된 처리 및 422/404/500 등 상태 코드·에러 응답 처리 기반 마련.

---

## 3. 구글 OAuth 로그인 및 인증 기반 확보 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **관련 커밋** | [f724320](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/f724320), [158cc58](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/158cc58), [7be7609](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/7be7609) |

**처리 내용**
- Google OAuth 로그인 후 **authCode**를 BE에 전달·토큰 발급·저장.
- 콜백에서 회원 여부 분기, 토큰 저장 유틸 정리. 이후 **채팅·SSE 요청 시 동일 토큰**으로 인증 가능한 기반 확보.

**목적**
- BE 챗봇 응답 API(SSE 스트리밍) 호출 시 `Authorization` 인증 헤더를 붙여 401·403을 줄이기 위한 기반.

---

## 4. API 스키마·Swagger 보강 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [108cbc79](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/108cbc79) |
| **메시지** | ChatRequest Swagger 예시에 model 필드 추가 및 면접 리포트 예시 추가 |

**처리 내용**
- Swagger(OpenAPI) 문서에서 `ChatRequest` 예시에 `model` 필드를 추가해 클라이언트가 어떤 값을 보내야 하는지 명확히 함.
- 면접 리포트 모드 사용 예시를 문서에 추가.

**목적**
- API 스키마 불일치로 인한 422 Validation Error 예방.

---

## 5. 리포트 모드 추가 및 채팅 API 스키마 정리 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [3c92d4bb](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/3c92d4bb) |
| **메시지** | Add report mode and refactor chat API schema |

**처리 내용**
- 면접 **리포트 모드** 기능 추가: 스트리밍 채팅과 구분된 리포트 생성 플로우 구현.
- 채팅 API 스키마 리팩터: 리포트 모드에 맞는 요청/응답 구조 정리 및 스키마 통일.

**목적**
- 면접 종료 후 한 번에 리포트를 받는 use case 지원 및 스키마 일관성 확보.

---

## 6. 이력서/채용 공고 분석 FastAPI 연동 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [4c35c03](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/4c35c03) |
| **메시지** | feat: 이력서/채용 공고 분석 FastAPI와 연동 (#55) |

**처리 내용**
- 이력서·채용 공고 분석을 **FastAPI(AI 서버)** 와 연동.
- 분석 요청·폴링·결과 수신 플로우 구현으로, 이후 404/500/비동기 오류 대응의 토대가 됨.

**목적**
- BE ↔ AI 서버 간 분석 API 연동 및 오류·재시도 처리 가능한 구조 확보.

---

## 7. 챗봇 응답 API 구현 및 SSE Streaming 적용 (BE — 핵심)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [f9f0bb4](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/f9f0bb4) |
| **메시지** | feat: 챗봇 응답 API 구현 및 SSE Streaming 적용 (#55) |

**처리 내용**
- **챗봇 응답 API** 구현: 클라이언트 요청을 BE가 받아 AI 서버(FastAPI)로 전달.
- **SSE(Server-Sent Events) 스트리밍** 적용: AI 서버의 스트리밍 응답을 그대로 클라이언트에 전달해 실시간 토큰 스트리밍 제공.
- `FastApiClient` 등으로 AI 서버 호출·스트림 전달·에러 전파 처리.

**목적**
- BE가 AI 채팅 SSE 스트림을 중계하여, 프론트가 하나의 백엔드 엔드포인트만으로 스트리밍 채팅을 사용할 수 있게 함.

---

## 8. LLM 채팅 SSE 전환 및 요청 스키마 반영 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **커밋** | [2425b46](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/2425b46) |
| **메시지** | feat: LLM 채팅 SSE 전환 및 요청 스키마 반영 |

**처리 내용**
- 기존 일반 REST 호출 방식에서 **SSE 스트리밍** 방식으로 전환.
- BE 채팅 API 스키마(mode, context, session_id 등) 반영.

**목적**
- 면접·채팅 모두 SSE 스트리밍으로 통일된 응답 수신.

---

# Part 2.
## 422/504 오류 대응 및 SSE 안정화

## 9. 422 검증 오류 대응 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **관련 커밋** | [fc88d09b](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/fc88d09b) (로깅 개선 및 422 처리), [b38257a1](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/b38257a1) (면접 리포트 모드 422 처리) |

**처리 내용**
- 422 Unprocessable Entity 발생 시 로그 보강으로 원인 파악 용이.
- **면접 리포트 모드** 전용 422 처리 로직 추가: 필수 필드 누락·타입 불일치 시 어떤 필드에서 검증 실패했는지 명시.

**목적**
- 리포트 모드·일반 채팅 요청 시 `model` 등 필드 누락/형식 오류로 422가 나는 문제를 진단·대응.

---

## 10. 504 타임아웃 방지 — SSE Heartbeat 도입 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [4a06f0e2](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/4a06f0e2) |
| **메시지** | Improve logging and add SSE heartbeat for 504 timeout prevention |

**처리 내용**
- **SSE heartbeat** 도입: 스트리밍 중 오래 걸리는 LLM 응답 구간에서도 주기적으로 빈 데이터(또는 heartbeat 전용 이벤트)를 보내 연결 유지.
- 로깅 개선: 스트리밍·에러 구간에서 로그 추가로 504 발생 직전 상황 추적 가능.

**목적**
- 프록시·로드밸런서·클라이언트에서 **일정 시간 데이터가 없으면 연결을 끊어 504 Gateway Timeout**이 나는 문제 완화.

**회고**
- SSE 연결 끊김의 가장 흔한 원인. LLM 응답 생성에 수 초 이상 걸리면 Nginx 등 프록시가 타임아웃하여 연결을 끊음. heartbeat는 이를 방지하는 가장 직접적인 해결책.

---

## 11. SSE 포맷 표준화 및 ChromaDB None 필터 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [08916a45](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/08916a45) |
| **메시지** | Standardize SSE format and add ChromaDB None filtering |

**처리 내용**
- **SSE 포맷 표준화**: `event`, `data`, 주석 라인 등 SSE 스펙에 맞게 형식 통일.
- **ChromaDB None 필터링**: RAG/벡터 검색 시 `metadata`에 `None`이 들어가 ChromaDB 쿼리 오류가 나는 경우를 방지, `None` 값 제거·필터 처리.

**목적**
- SSE 파싱 오류 및 ChromaDB 예외로 인한 스트리밍 중단 감소.

---

## 12. 채팅 엔드포인트에 SSE 헤더 추가 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [9b712c3b](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/9b712c3b) |
| **메시지** | Add SSE headers to chat endpoint |

**처리 내용**
- 채팅 스트리밍 응답에 **SSE용 HTTP 헤더** 명시적 설정:
  - `Content-Type: text/event-stream`
  - `Cache-Control: no-cache`
  - `Connection: keep-alive`
  - `X-Accel-Buffering: no` (Nginx 버퍼링 비활성화)

**목적**
- 브라우저·프록시가 응답을 일반 JSON으로 취급하거나 버퍼링해 스트리밍이 깨지는 현상 방지.

---

## 13. 챗봇 메시지 정렬 오류 수정 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [6724ba8](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/6724ba8) |
| **메시지** | chore: 챗봇 메시지 조회 시 정렬 순서 DESC에서 ASC로 변경 (#31) |

**처리 내용**
- 채팅 메시지 조회 시 정렬을 **DESC → ASC**로 변경해, 시간순(과거→최신)으로 메시지 표시.

**목적**
- 클라이언트·SSE 스트리밍과 연동할 때 대화 순서가 올바르게 표시되도록 함.

---

## 14. 422 전역 핸들러 (AI — main.py)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **위치** | `app/main.py` |

**처리 내용**
- `RequestValidationError` 전역 `exception_handler` 등록.
- 422 발생 시 요청 URL, Method, Body(최대 2000자), Errors(detail) 로깅 후 `JSONResponse(status_code=422, content={"detail": exc.errors()})` 반환.

**목적**
- 422 원인(필드 누락·타입 불일치) 디버깅 용이.

---

## 15. 422·504 대응 UX (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **참고** | [트러블슈팅 정리](./SSE_스트리밍_트러블슈팅_정리.md) §6·§7 |

**처리 내용**
- **422**: 요청 실패 시 BE가 반환하는 `detail`(검증 오류 목록) 또는 메시지를 화면에 표시해 사용자가 필드·형식을 수정할 수 있도록 안내.
- **504**: 스트림이 끊겼을 때 "연결이 만료되었습니다. 다시 시도해주세요." 메시지 표시 및 재요청 버튼 제공.

**목적**
- 422(스키마·필드 오류)·504(타임아웃) 상황에서 사용자가 원인을 인지하고 재시도할 수 있도록 UX 보완.

---

# Part 3.
## SSE 스트리밍 연결 안정화 (BE)

## 16. SSE 스트리밍 Access Denied 문제 해결 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **관련 커밋** | [a2afa85](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/a2afa85) (Access Denied 해결 #106), [878ff21](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/878ff21) (shouldNotFilterAsyncDispatch 오버라이드 #106) |

**처리 내용**
- SSE 스트리밍 응답 시 Spring Security 필터가 비동기 디스패치를 차단하여 **403 Access Denied** 발생.
- `shouldNotFilterAsyncDispatch()` 오버라이드로 비동기 SSE 디스패치 허용.

**목적**
- SSE 스트리밍 중 403 Access Denied 오류 해결.

**회고**
- Spring Security의 기본 동작이 비동기 디스패치를 필터링하는 것이 원인. SSE 스트리밍은 비동기 디스패치를 사용하므로, 해당 필터를 명시적으로 비활성화해야 함.

---

## 17. SSE 응답 body null 문제 및 로깅 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [2b7a868](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/2b7a868) |
| **메시지** | fix: FastAPI SSE 응답 body 확인을 위한 로그 추가 (#106) |

**처리 내용**
- AI Server로부터 받은 SSE 응답 body가 null로 들어오는 문제 디버깅을 위한 로그 추가.
- SSE 이벤트 수신 시 raw body를 로깅하여 파싱 전 데이터 확인 가능.

**목적**
- SSE 중계 과정에서 데이터 유실 원인 추적.

---

## 18. ServerSentEvent 자동 파싱으로 변경 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [cf5dfdd](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/cf5dfdd) |
| **메시지** | fix: ServerSentEvent 타입을 사용하여 자동 파싱으로 변경 (#106) |

**처리 내용**
- 수동 문자열 파싱 대신 Spring의 `ServerSentEvent<String>` 타입을 활용하여 SSE 이벤트 자동 파싱.
- event/data/id 필드를 구조적으로 접근 가능하도록 개선.

**목적**
- SSE 파싱 안정화 및 코드 간소화.

**회고**
- 수동 문자열 파싱은 줄바꿈·인코딩 차이로 깨지기 쉬움. Spring이 제공하는 `ServerSentEvent<String>` 타입을 사용하면 파싱 로직이 프레임워크에 위임되어 안정적.

---

## 19. SSE CRLF 줄바꿈 파싱 보정 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **커밋** | [d734867](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/d734867) |
| **메시지** | fix: SSE CRLF 줄바꿈 파싱 보정 (#69) |

**처리 내용**
- SSE 이벤트 파싱 시 `\r\n`(CRLF)과 `\n`(LF) 혼용으로 이벤트 경계를 잘못 인식하는 문제.
- 줄바꿈 패턴을 정규식으로 정규화하여 안정적 파싱.

**목적**
- AI Server(Python) → BE(Java) → FE(JS) 경유 시 줄바꿈 불일치 문제 해결.

**회고**
- 3단 구조(Python → Java → JS)를 거치면서 줄바꿈 문자가 변환되는 문제. OS 및 런타임별 줄바꿈 차이를 고려한 파싱이 필요.

---

# Part 4.
## SSE 에러 이벤트 체계화 (AI v1 → v2)

## 20. SSE 에러 응답 구조 개선 — v1 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | 728cba6 |
| **메시지** | refactor: SSE 에러 응답 구조 개선 |

**처리 내용**
- v1 채팅 스트리밍 구간에서 SSE 에러 응답 형식을 정리해 클라이언트/BE가 에러를 일관되게 파싱할 수 있도록 함.

**목적**
- 스트림 내 에러 전달 방식 통일의 기반.

---

## 21. 아키텍처 모듈화 및 v1/v2 분리 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **관련 커밋** | 43d8043 (DI 기반 인프라·도메인 분리), 017e20a (API routes v1/v2 분리) |

**처리 내용**
- DI 기반 인프라·도메인 계층 분리, 라우트·서비스 구조 정리. v2 라우트 분리의 기반.
- 채팅·마스킹·텍스트 추출 등 API를 **v1(하위 호환)** / **v2(현재)** 로 분리.

**목적**
- 유지보수·테스트·SSE/채팅 로직 확장 용이. 스키마·동작 정리 및 SSE 엔드포인트 일원화.

---

## 22. SSE 에러 이벤트 통일 포맷 — v2 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **위치** | `app/api/routes/v2/_sse_errors.py` |

**처리 내용**
- **백엔드(Spring)가 SSE 스트리밍 중 에러를 인식**할 수 있도록, 모든 에러를 통일된 JSON 포맷으로 전송하는 `sse_error_event(code, status, message, fallback)` 도입.
- **SSE 이벤트 타입**: `chunk`(정상 텍스트), `summary`(채팅방 제목), `session_state`(면접 세션), **`error`**(에러 발생), `[DONE]`(스트림 종료).
- **에러 payload**: `type: "error"`, `error: { code, status, message }`, `fallback`(사용자 표시용).

**에러 코드 표**

| 에러 코드 | HTTP status | 발생 상황 | fallback 예시 |
|-----------|-------------|-----------|----------------|
| **PROMPT_BLOCKED** | 400 | 프롬프트 인젝션 차단(RiskLevel.BLOCK) | 프롬프트 인젝션 관련 안내 |
| **VECTORDB_ERROR** | 404 | VectorDB에 문서 없음(분석 모드) | 업로드된 이력서/채용공고를 찾을 수 없습니다 |
| **INTERNAL_ERROR** | 500 | 일반 채팅·면접 진행 중 미처리 예외 | 일시적인 오류가 발생했습니다 |
| **PARSE_FAILED** | 500 | 면접 질문 세트·꼬리질문 JSON 파싱 실패 | 질문 세트/꼬리질문 생성 중 오류 안내 |
| **SESSION_NOT_FOUND** | 404 | 면접 세션에서 현재 질문 ID를 찾을 수 없음 | 세션 오류: 현재 질문을 찾을 수 없습니다 |
| **LLM_ERROR** | 500 | 면접 리포트 생성 중 LLM 예외 | 면접 리포트 생성 중 오류가 발생했습니다 |

- 모든 경우 **HTTP 200 유지**, 본문에 `data: {"type":"error", ...}\n\n` 전송 후 필요 시 `data: [DONE]\n\n`으로 스트림 종료.

**목적**
- BE·프론트가 에러 시 동일한 구조로 파싱하여 처리·UX 일관성 확보.

---

## 23. 예외 처리 개선 및 SSE 에러 핸들링 추가 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | 5631976 |
| **메시지** | fix: 예외 처리 개선 및 SSE 에러 핸들링 추가 |

**처리 내용**
- v2 `generate_chat_stream()` 내 일반 대화·분석·면접·리포트 구간에서 예외 발생 시 `sse_error_event()`로 에러 이벤트 전송 후 `[DONE]`으로 종료.
- 스트리밍 중 예외가 나도 연결을 끊지 않고 클라이언트/BE가 에러 payload만으로 처리 가능.

**목적**
- 스트리밍 중 예외 발생 시에도 HTTP 200 + error 이벤트로 안정적으로 전달.

---

## 24. SSE 이벤트 파싱 및 에러 이벤트 처리 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **참고** | [이력(AI)](./SSE_스트리밍_오류처리_이력(AI).md) §18·§19 |

**처리 내용**
- 수신한 각 이벤트에서 `data`를 JSON 파싱. **`type === "error"`** 이면 에러 이벤트로 간주.
- **`fallback`**(사용자 표시용 메시지) 또는 **`error.code`**에 따라 UI 분기(토스트·에러 영역 표시, 재시도 유도).
- `data: [DONE]` 수신 시 스트림 종료 처리.

**목적**
- BE·AI가 보내는 통일된 에러 포맷을 프론트에서 일관되게 파싱해, 422/504/스트림 내부 오류 시 사용자에게 안내 메시지 표시.

---

# Part 5.
## 분석 API 오류·동작 보완 (BE ↔ AI)

## 25. jobPost DTO 필드명 매칭 문제 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [a8d769f](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/a8d769f) |
| **메시지** | fix: jobPost DTO 필드명 매칭 문제 해결 (#68) |

**처리 내용**
- 채용 공고 분석 시 **jobPost DTO** 필드명이 AI 서버 또는 내부 스키마와 불일치해 발생하던 오류 수정.

**목적**
- 분석 요청 시 DTO 매핑 오류로 인한 실패 제거.

---

## 26. "비동기 작업을 찾을 수 없습니다" 문제 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [0ca1689](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/0ca1689) |
| **메시지** | fix: '비동기 작업을 찾을 수 없습니다' 문제 해결을 위해 DB 커밋 전에 task를 조회하던 로직 수정 |

**처리 내용**
- 비동기 분석 **task**를 DB **커밋 전**에 조회해 "비동기 작업을 찾을 수 없습니다"가 나던 문제 수정.
- 트랜잭션 커밋 순서·task 조회 시점 조정.

**목적**
- 분석 결과 폴링 시 404/500 유사 오류 및 **무한 폴링** 원인 제거.

---

## 27. ExternalTaskId 제거 및 taskId 통합 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [342aced](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/342aced) |
| **메시지** | feat: ExternalTaskId 제거 후 백엔드 taskId로 통합 (#70) |

**처리 내용**
- **ExternalTaskId** 제거 후 **백엔드 taskId** 하나로 통일.
- 클라이언트·AI 서버와의 task 식별자 계약 단순화.

**목적**
- task 조회 실패·404/500 원인 중 하나인 ID 불일치 제거.

---

## 28. 이력서 분석 엔드포인트 URI 매핑 및 404 재시도 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **관련 커밋** | [dc10953](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/dc10953) (URI 매핑 실패 해결 #70), [c9f4520](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/c9f4520) (404 에러 시 재시도 #70) |

**처리 내용**
- 이력서 분석 요청 시 **엔드포인트 URI 매핑 실패**로 404/500이 나던 문제 수정.
- AI 서버 호출 시 **404**가 발생하면 **재시도**하도록 처리(일시적 라우팅/배포 지연 대응).

**목적**
- 분석 API 호출 단계에서의 404/500 제거 및 완화.

---

## 29. 이력서 분석 비동기 처리 및 OCR (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **관련 커밋** | [65460a8](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/65460a8) (self-invocation Lazy 처리 #55), [fc5c743](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/fc5c743) (OCR 데이터 저장 #70), [0fb3055](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/0fb3055) (taskId 로깅 추가) |

**처리 내용**
- 비동기 처리 시 self-invocation 이슈 → **Lazy** 방식 호출로 수정.
- OCR 결과 DB 저장 로직 추가.
- taskId 로깅으로 404/500/무한 폴링 원인 분석 지원.

**목적**
- 비동기 분석 플로우 안정화 및 디버깅 용이.

---

# Part 6.
## 스트리밍 렌더링 및 UX 개선 (FE + AI)

## 30. SSE 타이핑 공백 렌더링 보존 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **커밋** | [79cc203](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/79cc203) |
| **메시지** | feat: SSE 타이핑 공백 렌더링 보존 |

**처리 내용**
- SSE 청크 수신 시 공백 문자가 HTML 렌더링에서 무시되는 문제.
- 연속 공백·줄바꿈을 `white-space: pre-wrap` 등으로 보존.

**목적**
- AI 응답의 코드 블록·들여쓰기 등 공백이 포함된 텍스트가 정확히 표시.

**회고**
- 스트리밍 렌더링에서 공백 보존은 코드 관련 응답(면접 기술 질문 등)에서 특히 중요. `white-space: pre-wrap`이 가장 간단한 해결책.

---

## 31. 스트리밍 응답 타이핑 애니메이션 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **커밋** | [e4e141d](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/e4e141d) |
| **메시지** | feat: 스트리밍 응답 타이핑 애니메이션 추가 (#92) |

**처리 내용**
- SSE 청크 수신 시 한 글자씩 타이핑되는 애니메이션 효과 구현.
- AI 면접 질문·응답이 자연스럽게 출력되는 UX.

**목적**
- LLM 스트리밍 응답의 체감 속도 및 UX 개선.

---

## 32. 면접 질문 생성 속도 개선 및 타이핑 효과 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [8d36874](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/8d36874) |
| **메시지** | feat: 면접 질문 생성 속도 개선 및 타이핑 효과 추가 |

**처리 내용**
- 면접 질문 생성 시 LLM 호출 최적화로 응답 속도 개선.
- 클라이언트에 타이핑 효과용 청크 단위 전송 추가.

**목적**
- 면접 질문 생성 대기 시간 단축 및 UX 개선.

---

## 33. 채팅방 제목 추출·분석 포맷 개선 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **관련 커밋** | d1502fc (채팅방 제목 자동 추출), c41277c (파일 업로드 API 제목 추출), 35f899c (프롬프트 응답 형식 개선), 18ef830 (formatted_text 필드), a60589f (분석 결과 포맷 개선) |

**처리 내용**
- 채팅방 제목 자동 추출(회사명/채용직무), 분석 결과 포맷 개선.
- SSE `summary` 이벤트로 채팅방 제목 전송.

**목적**
- UX·응답 일관성 개선.

---

## 34. 분석 완료 알림 및 채팅방 제목 업데이트 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **관련 커밋** | [766074e](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/766074e) (summary → title 업데이트 #120), [6b873a3](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/6b873a3) (분석 완료 알림 읽음 처리 #150) |

**처리 내용**
- AI Server 분석 결과의 `summary` 필드로 채팅방 제목 자동 업데이트.
- 채팅방 바로 입장 시에도 분석 완료 알림 읽음 처리.

**목적**
- 채팅방 목록에서 분석 대상(회사/직무) 즉시 식별 가능.

---

## 35. 면접 종료 확인 안내 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **관련 커밋** | [d68d4d2](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/d68d4d2) (면접 종료 확인 안내 추가 #110), [665782a](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/665782a) (면접 종료 확인 안내 삭제 #110) |

**처리 내용**
- 면접 종료 확인 안내 UI 추가 후 UX 피드백 반영하여 삭제.

**목적**
- 사용자 경험 최적화 (과도한 안내 제거).

---

# Part 7.
## 보안 강화 — 로그 인젝션·프롬프트 인젝션

## 36. 로그 인젝션 방어 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **관련 커밋** | 2324d5a (sanitize_log_input 적용), 483d119 (safe_info 래퍼), 7c7299b (요청 모델명 sanitize), 77006e7 (Log Injection 수정·포맷팅), 615960f (task_id 로깅 시 sanitize), 6fbd074 (v1/masking.py Log Injection 해결) 등 |

**처리 내용**
- 로그에 기록되는 사용자 입력·모델명 등에 **sanitize** 적용(개행·제어 문자 제거).
- `safe_info`, `safe_warning` 등 래퍼로 CRLF Injection 방어.
- v2·마스킹 포함 전 구간 대응.

**목적**
- 로그 인젝션·개인정보 노출 위험 완화 및 CodeQL 코드 스캔 알림 대응.

---

## 37. 로그 인젝션 방어 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **관련 커밋** | [a8a7c84](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/a8a7c84) (Log sanitized #55), [49b71cd](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/49b71cd) (todoId 로그 제거 #118), [72cc914](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/72cc914) (eventId 로그 제거 #99) |

**처리 내용**
- 사용자 입력·외부 값이 로그에 그대로 노출되지 않도록 **로그 산티라이즈** 적용.
- 사용자 입력 값(todoId, eventId)이 로그에 직접 기록되는 취약점 제거.

**목적**
- Log Injection·개인정보 노출 위험 완화 및 CodeQL 코드 스캔 알림 대응.

---

## 38. 프롬프트 인젝션 방어 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **관련 커밋** | 2339c52 (면접 기능 개선 - 보안 취약점 수정), ce21bed (프롬프트/로그 인젝션 방어), 19d5537 (프롬프트 인젝션 로깅에 safe_warning 적용) |

**처리 내용**
- **프롬프트 인젝션** 검사 도입: `check_prompt_injection()`.
- BLOCK 시 요청 거부·SSE 에러 응답(`PROMPT_BLOCKED`), WARNING 시 로깅 후 진행.

**목적**
- 시스템 프롬프트 탈취·역할 변경·인젝션 명령어 등 위험 입력 차단.

---

## 39. 보안·정적 분석 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **관련 커밋** | [46c5844](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/46c5844) (CodeQL 규칙 통일), [d86bf51](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/d86bf51) (보안 취약점 확인 및 차단) |

**처리 내용**
- CodeQL 기반 정적 분석·보안 취약점 점검.
- 채팅·사용자 입력을 다루는 구간에서 XSS·인젝션 등 방지.

**목적**
- SSE·채팅 UI에서 사용자 입력·에러 메시지 표시 시 보안 이슈 방지.

---

# Part 8.
## Gemini 모델 안정화 및 LLM 오류 대응 (AI)

## 40. Gemini 빈 응답·분석 API 안정화 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **관련 커밋** | 4af8155 (빈 응답 해결, 분석 API 단계별 분할), 03f1eaf (안정 모델·재시도), 1ff6498 (빈 응답 None 체크), 93e0033 (분석 결과 JSON 파싱, Gemini JSON 모드), 414ed2d (fallback·빈 맥락 방지) |

**처리 내용**
- 분석 API 단계별 분할 호출, 재시도·fallback, JSON 파싱 실패 대응, 빈 응답 None 체크.
- Gemini 모델명 오류 수정(1372b56).

**목적**
- 분석·면접 구간에서 LLM 불안정으로 인한 스트리밍 중단·오류 감소.

**회고**
- LLM은 항상 일관된 응답을 보장하지 않음. 빈 응답·파싱 실패·모델 불안정에 대한 방어적 프로그래밍이 필수.

---

## 41. 면접 질문 파싱·예외 처리 강화 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **관련 커밋** | e8f1aea (JSON 파싱 개선), 851a5fb (예외 처리 개선), 5b64cea (JSON 파싱 실패 수정), 68212a0 (질문 길이 제한) |

**처리 내용**
- 면접 질문 JSON 파싱 실패 시 SSE 에러 이벤트(`PARSE_FAILED`)로 안정 전달.
- 질문 길이 제한 추가로 LLM 과다 응답 방지.
- markdown 코드블록 포함 시에도 JSON 부분만 추출.

**목적**
- 면접 진행 중 파싱 오류로 인한 스트리밍 중단 방지.

---

## 42. 세션 캐시 및 비스트리밍 LLM 응답 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [8ffda2f](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/8ffda2f) |
| **메시지** | feat: 세션 캐시 및 비스트리밍 LLM 응답 추가 |

**처리 내용**
- 면접 세션 캐시 도입: 세션별 질문·답변 히스토리를 메모리에 캐싱.
- 질문 세트 생성 등 구간에 비스트리밍(non-stream) LLM 호출 적용.

**목적**
- 면접 초기화·질문 생성 구간 안정화 및 응답 형식 일관성.

---

# Part 9.
## 분석 시 500 오류 및 Rate Limit (BE)

## 43. 분석 시 500 오류 및 Rate Limit (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **관련 커밋** | [d949056](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/d949056) (분석 시도 시 500), [e16daa0](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/e16daa0) (Rate Limit 기능 추가 #131) |

**처리 내용**
- 분석 API 호출 시 500 에러 수정.
- API Rate Limit 기능 추가로 과다 요청 방지.

**목적**
- 서버 안정성 확보 및 AI Server 과부하 방지.

---

## 44. DTO Validation 강화 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **관련 커밋** | [ba44e81](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/ba44e81) (cascade Valid), [36788ea](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/36788ea) (Size 어노테이션 범위), [54c1e4e](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/54c1e4e) (길이 제한) |

**처리 내용**
- 중첩 DTO에 `@Valid` cascade 적용으로 하위 객체 검증 누락 방지.
- `@Size` 어노테이션 범위 수정으로 입력 길이 제한 정확 적용.

**목적**
- 잘못된 요청이 AI Server까지 도달하기 전에 BE에서 422로 차단.

---

# Part 10.
## 환경 변수·배포 설정

## 45. 환경 변수 및 배포 설정 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **관련 커밋** | [433ab42](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/433ab42), [669f2b5](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/669f2b5) (환경 변수), [c9b39e8](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/c9b39e8) (CodeDeploy·S3 배포) |

**처리 내용**
- `NEXT_PUBLIC_API_BASE_URL` 등 환경별 API 베이스 URL 주입.
- 정적 빌드·S3 배포 설정으로 BE·AI와 동일 도메인/CORS 이슈 완화.

**목적**
- 로컬·스테이징·운영에서 올바른 BE(및 SSE 엔드포인트)를 바라보도록 하고, 배포 후에도 스트리밍 연결이 안정적으로 동작.

---

# 요약: 이슈 유형별 대응

## SSE 연결 끊김 문제 및 재연결 전략

| 이슈 | 원인 | 해결 | 관련 단계 |
|------|------|------|-----------|
| 504 Gateway Timeout | LLM 응답 지연 시 프록시가 연결 끊음 | SSE heartbeat 도입 | §10 |
| SSE 헤더 미설정 | 프록시가 응답을 일반 HTTP로 인식·버퍼링 | `Content-Type`, `Cache-Control`, `X-Accel-Buffering` 명시 | §12, §21 |
| Access Denied (403) | Spring Security 비동기 디스패치 차단 | `shouldNotFilterAsyncDispatch()` 오버라이드 | §16 |
| SSE body null | 중계 과정 데이터 유실 | raw body 로깅 → ServerSentEvent 자동 파싱으로 전환 | §17, §18 |

## 스트리밍 응답 파싱 및 렌더링 방식

| 이슈 | 원인 | 해결 | 관련 단계 |
|------|------|------|-----------|
| SSE 포맷 불일치 | event/data 구조 비표준 | SSE 포맷 표준화 | §11 |
| CRLF 줄바꿈 파싱 오류 | Python→Java→JS 경유 시 줄바꿈 변환 | 정규식으로 줄바꿈 정규화 | §19 |
| 공백 렌더링 손실 | HTML이 연속 공백 무시 | `white-space: pre-wrap` 적용 | §30 |
| 수동 문자열 파싱 불안정 | 인코딩·줄바꿈 차이 | `ServerSentEvent<String>` 자동 파싱 | §18 |
| 면접 질문 JSON 파싱 실패 | LLM이 markdown 코드블록 포함 | JSON 부분만 추출, 길이 제한 | §41 |

## 에러 발생 시 사용자 경험 처리

| 이슈 | 원인 | 해결 | 관련 단계 |
|------|------|------|-----------|
| 에러 시 스트림 끊김 | 예외 발생 시 연결 즉시 종료 | HTTP 200 유지 + error 이벤트 전송 후 [DONE] | §22, §23 |
| 에러 포맷 불일치 | AI/BE/FE 각각 다른 에러 구조 | `sse_error_event()` 통일 포맷 | §22 |
| 422 검증 오류 | 필드 누락·타입 불일치 | 전역 핸들러, 상세 에러 메시지 반환 | §9, §14 |
| 사용자 에러 인지 불가 | 에러 메시지 미노출 | fallback 메시지 UI 표시, 재시도 유도 | §15, §24 |
| 프롬프트 인젝션 | 악의적 사용자 입력 | `check_prompt_injection()` + SSE PROMPT_BLOCKED | §38 |

## 보안 대응

| 이슈 | 대응 | 관련 단계 |
|------|------|-----------|
| 로그 인젝션 (AI) | sanitize_log_input, safe_info/safe_warning | §36 |
| 로그 인젝션 (BE) | 로그 산티라이즈, 사용자 입력 제거 | §37 |
| 프롬프트 인젝션 (AI) | check_prompt_injection(), BLOCK/WARNING | §38 |
| XSS·인젝션 (FE) | CodeQL 정적 분석 | §39 |

---

## 문서 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-08 | AI·BE·FE 3개 저장소 SSE 관련 커밋 분석 및 통합 문서 초안 작성 |

---

*이 문서는 [SSE_스트리밍_오류처리_이력(AI).md](./SSE_스트리밍_오류처리_이력(AI).md), [SSE_스트리밍_오류처리_이력(BE).md](./SSE_스트리밍_오류처리_이력(BE).md), [SSE_스트리밍_오류처리_이력(FE).md](./SSE_스트리밍_오류처리_이력(FE).md), [SSE_스트리밍_트러블슈팅_정리.md](./SSE_스트리밍_트러블슈팅_정리.md)와 함께 참고합니다.*
