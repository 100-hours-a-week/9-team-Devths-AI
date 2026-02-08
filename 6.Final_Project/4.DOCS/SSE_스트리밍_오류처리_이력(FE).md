# SSE 스트리밍 오류 처리 이력 (FE)

[9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) 저장소에서 AI 채팅·SSE 스트리밍 **수신**·API 연동·오류 처리(422/504·에러 이벤트)와 관련된 작업을 **시간 흐름순**으로 정리한 문서입니다.  
프론트는 BE 하나의 엔드포인트로 스트리밍을 받는 구조이며, [SSE_스트리밍_트러블슈팅_정리.md](./SSE_스트리밍_트러블슈팅_정리.md)·[이력(BE)](./SSE_스트리밍_오류처리_이력(BE).md)·[이력(AI)](./SSE_스트리밍_오류처리_이력(AI).md)와 함께 참고합니다.

---

## 1. 공통 API 클라이언트 및 응답 타입

### 1.1 공통 fetch 클라이언트 추가

| 항목 | 내용 |
|------|------|
| **커밋** | [db0733d](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/db0733d) |
| **메시지** | chore: 공통 fetch 클라이언트 추가 |

**처리 내용**
- BE API 호출용 **공통 fetch 클라이언트** 구현. `apiRequest`, `api.get` / `api.post` 등 메서드 제공.
- `NEXT_PUBLIC_API_BASE_URL` 기반 URL 구성, 인증 토큰(`Authorization`), `credentials: 'include'` 지원.
- 응답의 `Content-Type`에 따라 JSON 파싱 여부 분기, `ok` / `status` / `json` / `res` 반환.

**목적**
- 채팅·분석·SSE 엔드포인트를 제외한 일반 API 요청의 일관된 처리 및 422/404/500 등 상태 코드·에러 응답 처리 기반 마련.

---

### 1.2 API 응답 타입 정의

| 항목 | 내용 |
|------|------|
| **커밋** | [8da4d08](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/8da4d08) |
| **메시지** | chore: API 응답 타입 정의 |

**처리 내용**
- **ApiResponse&lt;T&gt;** (message, data, timestamp), **ApiErrorResponse** (message, data: null, timestamp) 타입 정의.
- 공통 클라이언트와 연동해 성공/에러 응답을 타입으로 구분해 처리할 수 있도록 함.

**목적**
- 422·500 등 에러 응답 시 `data`가 null인 경우와 성공 응답을 구분해, UI에서 에러 메시지 표시·재시도 유도에 활용.

---

## 2. 인증·세션과 API 연동 기반

### 2.1 구글 OAuth 로그인 및 토큰 저장

| 항목 | 내용 |
|------|------|
| **관련 커밋** | [f724320](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/f724320), [158cc58](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/158cc58), [7be7609](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/7be7609) 등 |
| **메시지** | feat: 구글 로그인 API 요청 함수 추가 및 콜백에서 호출 연결 / 구글 OAuth 콜백 로그인 플로우 안정화 / 콜백에서 회원 여부 분기 및 토큰 저장 |

**처리 내용**
- Google OAuth 로그인 후 **authCode**를 BE에 전달·토큰 발급·저장.
- 콜백에서 회원 여부 분기, 토큰 저장 유틸 정리. 이후 **채팅·SSE 요청 시 동일 토큰**으로 인증 가능한 기반 확보.

**목적**
- BE 챗봇 응답 API(SSE 스트리밍) 호출 시 `Authorization` 등 인증 헤더를 붙여 401·403을 줄이기 위한 기반.

---

## 3. 채팅·SSE 스트리밍 수신 및 에러 처리 (연동 관점)

> 아래는 **프론트 ↔ BE ↔ AI** 3단 구조에서 FE가 담당하는 역할과, BE·AI 이력 문서·트러블슈팅 정리와 맞추어 정리한 항목입니다.  
> 상세 구현 커밋은 [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) 저장소 **develop** 브랜치 또는 채팅/SSE 관련 PR에서 확인·보강할 수 있습니다.

### 3.1 SSE 스트리밍 수신 방식

| 항목 | 내용 |
|------|------|
| **참고** | [SSE_스트리밍_트러블슈팅_정리.md](./SSE_스트리밍_트러블슈팅_정리.md) §2, §3 |
| **처리 내용** | BE가 AI 서버 스트림을 중계하므로, **프론트는 BE의 챗봇 응답 엔드포인트 한 곳**에만 연결. `EventSource` 또는 `fetch` + `ReadableStream`으로 `text/event-stream` 수신. |
| **목적** | 단일 엔드포인트로 실시간 토큰 스트리밍 수신, 504/버퍼링 이슈는 BE·AI 측 heartbeat·헤더로 완화된 스트림을 받음. |

---

### 3.2 SSE 이벤트 파싱 및 에러 이벤트 처리

| 항목 | 내용 |
|------|------|
| **참고** | [이력(AI)](./SSE_스트리밍_오류처리_이력(AI).md) §18·§19 — `type`, `error.code`, `fallback` |
| **처리 내용** | 수신한 각 이벤트에서 `data`를 JSON 파싱. **`type === "error"`** 이면 에러 이벤트로 간주하고, **`fallback`**(사용자 표시용 메시지) 또는 **`error.code`**에 따라 UI 분기(토스트·에러 영역 표시, 재시도 유도). `data: [DONE]` 수신 시 스트림 종료 처리. |
| **목적** | BE·AI가 보내는 통일된 에러 포맷을 프론트에서 일관되게 파싱해, 422/504/스트림 내부 오류 시 사용자에게 안내 메시지 표시. |

---

### 3.3 422·504 대응 UX

| 항목 | 내용 |
|------|------|
| **참고** | [트러블슈팅 정리](./SSE_스트리밍_트러블슈팅_정리.md) §6·§7, [이력(AI)](./SSE_스트리밍_오류처리_이력(AI).md) §3·§4 |
| **처리 내용** | **422**: 요청 실패 시 BE가 반환하는 `detail`(검증 오류 목록) 또는 메시지를 화면에 표시해 사용자가 필드·형식을 수정할 수 있도록 안내. **504**: 스트림이 끊겼을 때 “연결이 만료되었습니다. 다시 시도해주세요.” 등 메시지 표시 및 재요청 버튼 제공. |
| **목적** | 422(스키마·필드 오류)·504(타임아웃) 상황에서 사용자가 원인을 인지하고 재시도할 수 있도록 UX 보완. |

---

## 4. 기타 연동·배포 관련

### 4.1 환경 변수 및 배포 설정

| 항목 | 내용 |
|------|------|
| **관련 커밋** | [433ab42](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/433ab42), [669f2b5](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/669f2b5) — 환경 변수 주입 / [c9b39e8](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/c9b39e8) CodeDeploy·S3 배포 등 |
| **처리 내용** | `NEXT_PUBLIC_API_BASE_URL` 등 환경별 API 베이스 URL 주입. 정적 빌드·S3 배포 설정으로 BE·AI와 동일 도메인/ CORS 이슈 완화 가능한 구성. |
| **목적** | 로컬·스테이징·운영에서 올바른 BE(및 SSE 엔드포인트)를 바라보도록 하고, 배포 후에도 스트리밍 연결이 안정적으로 동작하도록 함. |

---

### 4.2 보안·정적 분석

| 항목 | 내용 |
|------|------|
| **관련 커밋** | [46c5844](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/46c5844) CodeQL 규칙 통일, [d86bf51](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/d86bf51) 보안 취약점 확인 및 차단 등 |
| **처리 내용** | CodeQL 기반 정적 분석·보안 취약점 점검. 채팅·사용자 입력을 다루는 구간에서 XSS·인젝션 등 방지. |
| **목적** | SSE·채팅 UI에서 사용자 입력·에러 메시지 표시 시 보안 이슈가 나지 않도록 사전 점검. |

---

## 요약: 오류·기능 유형별 대응 (FE)

| 유형 | 대응 내용 | 참고 |
|------|-----------|------|
| **공통 API·에러 응답** | 공통 fetch 클라이언트, ApiResponse/ApiErrorResponse 타입 | §1 |
| **인증** | 구글 OAuth·토큰 저장, 채팅/SSE 요청 시 인증 헤더 사용 | §2 |
| **SSE 수신** | BE 단일 엔드포인트로 EventSource 또는 fetch+ReadableStream | §3.1, 트러블슈팅 정리 |
| **SSE 에러 이벤트** | type===error, fallback/error.code 파싱·UI 표시 | §3.2, 이력(AI) §18·§19 |
| **422·504 UX** | 검증 오류 안내, 타임아웃 시 메시지·재시도 유도 | §3.3 |
| **배포·보안** | 환경 변수·배포 설정, CodeQL·보안 점검 | §4 |

---

## 문서 이력

| 날짜 | 내용 |
|------|------|
| (초안) | AI·BE 이력 문서 및 트러블슈팅 정리 3종 참고, 9-team-Devths-FE 로컬 커밋 반영. 채팅/SSE 수신·에러 처리 항목은 연동 관점으로 정리·보강 예정. |

---

*상세 커밋은 [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) 저장소 **develop** 브랜치(457 commits) 및 채팅/SSE 관련 PR에서 추가로 반영할 수 있습니다.*
