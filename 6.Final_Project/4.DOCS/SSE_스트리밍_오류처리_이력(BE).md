# SSE 스트리밍 오류 처리 이력 (BE)

[9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) 저장소에서 AI 채팅·SSE 스트리밍·이력서/채용공고 분석(FastAPI 연동)·404/500 오류 대응을 위해 진행한 커밋을 **시간 흐름순**으로 정리한 문서입니다.

---

## 1. AI 채팅 API 기반 구축

### 1.1 AI 채팅방·채팅 메시지 API 구현

| 항목 | 내용 |
|------|------|
| **커밋** | [a203310](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/a20331028924be062ca833497c6fb390aa6b21df), [73c1c24](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/73c1c24bbe505e66ff419d3b6c173b92d4ecffb7), [976dd8f](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/976dd8fbdc8bf2c3d05363a29a6353e3c43034b4), [c26c8aa](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/c26c8aaeb159519d95ec505113f3b8eb5dabfadf) |
| **메시지** | feat: AI 채팅방 목록 조회 / 생성 / 삭제 / 채팅 내용 불러오기 API 구현 (#28) |

**처리 내용**
- AI 채팅방 목록 조회·생성·삭제 API 및 AI 채팅 내용 불러오기 API 구현.
- 채팅방·메시지 도메인·리포지토리·서비스·컨트롤러 구성으로 이후 **챗봇 SSE 스트리밍** 연동의 기반 마련.

**목적**
- 프론트·AI 서버(FastAPI)와 연동할 채팅 API의 기본 CRUD와 데이터 모델 확보.

---

## 2. 챗봇 메시지 조회 정렬

### 2.1 챗봇 메시지 조회 시 정렬 순서 변경

| 항목 | 내용 |
|------|------|
| **커밋** | [6724ba8](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/6724ba85514ec5015988095bbf08f87d3a51ffc3) |
| **메시지** | chore: 챗봇 메시지 조회 시 정렬 순서 DESC에서 ASC로 변경 (#31) |

**처리 내용**
- 채팅 메시지 조회 시 정렬을 **DESC → ASC**로 변경해, 시간순(과거→최신)으로 메시지가 나오도록 함.

**목적**
- 클라이언트·SSE 스트리밍과 연동할 때 대화 순서가 올바르게 표시되도록 함.

---

## 3. 이력서/채용 공고 분석 FastAPI 연동 및 로깅·비동기 보강

### 3.1 이력서/채용 공고 분석 FastAPI와 연동

| 항목 | 내용 |
|------|------|
| **커밋** | [4c35c03](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/4c35c031cae9e118520e02dc65527ae97e5d4227) |
| **메시지** | feat: 이력서/채용 공고 분석 FastAPI와 연동 (#55) |

**처리 내용**
- 이력서·채용 공고 분석을 **FastAPI(AI 서버)** 와 연동.
- 분석 요청·폴링·결과 수신 플로우 구현으로, 이후 404/500/비동기 오류 대응의 토대가 됨.

**목적**
- BE ↔ AI 서버 간 분석 API 연동 및 오류·재시도 처리 가능한 구조 확보.

---

### 3.2 Log Sanitized (로깅 보안)

| 항목 | 내용 |
|------|------|
| **커밋** | [a8a7c84](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/a8a7c84ddbf0cf8eca1b78b3182eaa666121c955) |
| **메시지** | chore: Log sanitizied (#55) |

**처리 내용**
- 사용자 입력·외부 값이 로그에 그대로 노출되지 않도록 **로그 산티라이즈** 적용.
- Log Injection·개인정보 노출 위험 완화.

**목적**
- 채팅·분석 등 사용자 데이터를 다루는 구간에서 로깅 시 보안·규정 대응.

---

### 3.3 이력서 분석 비동기 처리 self-invocation Lazy 처리

| 항목 | 내용 |
|------|------|
| **커밋** | [65460a8](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/65460a83c9d99dd4ddb910a354bbbb8d1649c31f) |
| **메시지** | feat: 이력서 분석 비동기 처리 self-invocation Lazy 처리 (#55) |

**처리 내용**
- 이력서 분석 **비동기 처리** 시 같은 빈 내부 호출(self-invocation)로 트랜잭션·프록시 이슈가 나지 않도록 **Lazy** 방식으로 호출하도록 수정.

**목적**
- 비동기 작업이 DB 커밋 전에 조회되며 발생하던 **“비동기 작업을 찾을 수 없습니다”** 유형 오류를 줄이기 위한 구조 정리.

---

## 4. 챗봇 응답 API 및 SSE 스트리밍 적용 (핵심)

### 4.1 챗봇 응답 API 구현 및 SSE Streaming 적용

| 항목 | 내용 |
|------|------|
| **커밋** | [f9f0bb4](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/f9f0bb46df0c6a25af0788fd052284f06cd075cb) |
| **메시지** | feat: 챗봇 응답 API 구현 및 SSE Streaming 적용 (#55) |

**처리 내용**
- **챗봇 응답 API** 구현: 클라이언트 요청을 BE가 받아 AI 서버(FastAPI)로 전달.
- **SSE(Server-Sent Events) 스트리밍** 적용: AI 서버의 스트리밍 응답을 그대로 클라이언트에 전달해 실시간 토큰 스트리밍 제공.
- `FastApiClient` 등으로 AI 서버 호출·스트림 전달·에러 전파 처리.

**목적**
- BE가 AI 채팅 SSE 스트림을 중계하여, 프론트가 하나의 백엔드 엔드포인트만으로 스트리밍 채팅을 사용할 수 있게 함.

---

### 4.2 PR #67 머지 — 챗봇 FastAPI 연동

| 항목 | 내용 |
|------|------|
| **커밋** | [6212f97](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/6212f9702d2450421dea1c0fe823d2abfc727428) |
| **메시지** | Merge pull request #67 from 100-hours-a-week/feat/chatbot-fastapi |

**처리 내용**
- 위 4.1(챗봇 응답 API + SSE Streaming) 및 관련 FastAPI 연동을 포함한 PR #67을 기본 브랜치에 반영.

---

## 5. 분석 API 오류·동작 보완

### 5.1 jobPost DTO 필드명 매칭 문제 해결

| 항목 | 내용 |
|------|------|
| **커밋** | [a8d769f](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/a8d769f7b4e32e2b2b315916b1e2a9037fc708f6) |
| **메시지** | fix: jobPost DTO 필드명 매칭 문제 해결 (#68) |

**처리 내용**
- 채용 공고 분석 시 **jobPost DTO** 필드명이 AI 서버 또는 내부 스키마와 불일치해 발생하던 오류(422/500 유사) 수정.

**목적**
- 분석 요청 시 DTO 매핑 오류로 인한 실패를 제거.

---

### 5.2 “비동기 작업을 찾을 수 없습니다” 문제 해결

| 항목 | 내용 |
|------|------|
| **커밋** | [0ca1689](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/0ca1689783b2d26f30d6c69a7e7e5178c07b5554) |
| **메시지** | fix: '비동기 작업을 찾을 수 없습니다' 문제 해결을 위해 DB 커밋 전에 task를 조회하던 로직 수정 |

**처리 내용**
- 비동기 분석 **task**를 DB **커밋 전**에 조회해 “비동기 작업을 찾을 수 없습니다”가 나던 문제 수정.
- 트랜잭션 커밋 순서·task 조회 시점을 조정해 폴링 시 task가 항상 조회되도록 함.

**목적**
- 분석 결과 폴링 시 404/500 유사 오류 및 **무한 폴링** 원인 제거 (PR #71, #72와 연계).

---

### 5.3 ExternalTaskId 제거 및 백엔드 taskId 통합

| 항목 | 내용 |
|------|------|
| **커밋** | [342aced](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/342aced90545537540c9426f1922fb9fe0373f68) |
| **메시지** | feat: ExternalTaskId 제거 후 백엔드 taskId로 통합 (#70) |

**처리 내용**
- **ExternalTaskId** 제거 후 **백엔드 taskId** 하나로 통일.
- 클라이언트·AI 서버와의 task 식별자 계약 단순화로 매핑 오류·불일치 가능성 감소.

**목적**
- task 조회 실패·404/500 원인 중 하나인 ID 불일치 제거.

---

### 5.4 이력서 분석 엔드포인트 URI 매핑 실패 해결

| 항목 | 내용 |
|------|------|
| **커밋** | [dc10953](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/dc10953c8d9312279c8a987b80333f6f7cd8f4ad) |
| **메시지** | fix: 이력서 분석 엔드포인트 URI 매핑 실패 해결 (#70) |

**처리 내용**
- 이력서 분석 요청 시 **엔드포인트 URI 매핑 실패**로 404/500이 나던 문제 수정.
- 올바른 경로·메서드로 FastAPI 분석 API 호출이 이루어지도록 수정.

**목적**
- 분석 API 호출 단계에서의 404/500 제거.

---

### 5.5 404 에러 시 재시도 처리

| 항목 | 내용 |
|------|------|
| **커밋** | [c9f4520](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/c9f45202d25641d49937b94df3b561b2a145b083) |
| **메시지** | fix: 404 에러 시 재시도 처리 (#70) |

**처리 내용**
- AI 서버 또는 분석 API 호출 시 **404**가 발생하면 **재시도**하도록 처리.
- 일시적 라우팅/배포 지연으로 인한 404를 완화.

**목적**
- 분석·채팅 연동 구간에서 404로 인한 사용자 오류 감소.

---

### 5.6 OCR 데이터 저장 로직 추가

| 항목 | 내용 |
|------|------|
| **커밋** | [fc5c743](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/fc5c7434f7c11745a853141a369f7d3e7079b815) |
| **메시지** | feat: OCR 데이터 저장 로직 추가 (#70) |

**처리 내용**
- 분석 결과 중 **OCR 결과**를 DB에 저장하는 로직 추가.
- 분석 플로우 완결 및 결과 조회·재활용 가능.

**목적**
- 분석 500/데이터 부재로 인한 후속 오류를 줄이고, 분석 이력 관리 보강.

---

### 5.7 taskId 로깅 추가

| 항목 | 내용 |
|------|------|
| **커밋** | [0fb3055](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/0fb30552d85289ec1428605aeae3cbbe337707e9) |
| **메시지** | fix: taskId 로깅 추가 |

**처리 내용**
- 비동기 분석 요청·폴링 시 **taskId**를 로그에 남기도록 추가.
- 404/500/무한 폴링 원인 분석 시 task 추적이 가능하도록 함.

**목적**
- 분석·SSE 연동 장애 시 원인 파악 용이.

---

## 6. 머지 커밋 (PR 반영)

| PR | 커밋 | 메시지 |
|----|------|--------|
| **#69** | [eb22d28](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/eb22d2847ca26e339844ce61a1cdb0cc3309995e) | Merge PR #69 (fix/analyze-500) |
| **#71** | [4d3303f](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/4d3303fd5f063b73a85fa1fd1110af955c2e3bcb) | Merge PR #71 (fix/analyze-infinite-polling) |
| **#72** | [ba2a0ff](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/ba2a0ffea5796137a099565a12a6c6ebe2536cc8) | Merge PR #72 (fix/analyze-infinite-polling) |
| **#73** | [b69345e](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/b69345e9d36f2a6ff9cbddb476439f4be1cbf524) | Merge PR #73 (fix/analyze-500-error) |
| **#74** | [1e35fe2](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/1e35fe2d679df21ac73b39eade9528f8736282c2) | Merge PR #74 (fix/analyze-text-404) |
| **#75** | [43948ac](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/43948acec4c23564b9e787c518b0b5e4d830c11b) | Merge PR #75 (fix/analyze-404) |

**처리 내용**
- 분석 500 오류 수정(#69, #73), 무한 폴링 수정(#71, #72), 분석/텍스트 404 수정(#74, #75) 등을 기본 브랜치에 반영.

---

---

# Part 2.
## SSE 스트리밍 연결 안정화


## 7. SSE 스트리밍 Access Denied 문제 해결

| 항목 | 내용 |
|------|------|
| **관련 커밋** | [a2afa85](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/a2afa85) (Access Denied 해결 #106), [878ff21](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/878ff21) (shouldNotFilterAsyncDispatch 오버라이드 #106) |

**처리 내용**
- SSE 스트리밍 응답 시 Spring Security 필터가 비동기 디스패치를 차단하여 403 Access Denied 발생.
- `shouldNotFilterAsyncDispatch()` 오버라이드로 비동기 SSE 디스패치 허용.

**목적**
- SSE 스트리밍 중 403 Access Denied 오류 해결.

---

## 8. SSE 응답 body null 문제 및 로깅

| 항목 | 내용 |
|------|------|
| **커밋** | [2b7a868](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/2b7a868) |
| **메시지** | fix: FastAPI SSE 응답 body 확인을 위한 로그 추가 (#106) |

**처리 내용**
- AI Server로부터 받은 SSE 응답 body가 null로 들어오는 문제 디버깅을 위한 로그 추가.
- SSE 이벤트 수신 시 raw body를 로깅하여 파싱 전 데이터 확인 가능.

**목적**
- SSE 중계 과정에서 데이터 유실 원인 추적.

---

## 9. ServerSentEvent 자동 파싱으로 변경

| 항목 | 내용 |
|------|------|
| **커밋** | [cf5dfdd](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/cf5dfdd) |
| **메시지** | fix: ServerSentEvent 타입을 사용하여 자동 파싱으로 변경 (#106) |

**처리 내용**
- 수동 문자열 파싱 대신 Spring의 `ServerSentEvent<String>` 타입을 활용하여 SSE 이벤트 자동 파싱.
- event/data/id 필드를 구조적으로 접근 가능하도록 개선.

**목적**
- SSE 파싱 안정화 및 코드 간소화.

---

# Part 3.
## 면접 상태 관리 — 401/인덱싱/세션 동기화


## 10. 면접 모드 시작/종료 구현

| 항목 | 내용 |
|------|------|
| **커밋** | [e71963f](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/e71963f) |
| **메시지** | feat: 면접 모드 시작/종료 구현 (#66) |

**처리 내용**
- 면접 세션 생성·종료 API 구현: `interview_status` 상태 관리.
- 면접 시작 시 AI Server에 `interview_question` 모드로 SSE 요청, 종료 시 `interview_report` 모드로 리포트 요청.

**목적**
- 면접 플로우(시작 → 질문 → 답변 → 종료 → 평가)의 BE 측 상태 관리 기반 구축.

---

## 11. 면접 평가 요청 시 401 발생 (1차)

| 항목 | 내용 |
|------|------|
| **커밋** | [144b8d9](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/144b8d9) |
| **메시지** | fix: 면접 평가 요청 시 401 발생 |

**처리 내용**
- 면접 리포트(평가) 요청 시 AI Server 호출에서 인증 헤더(`X-API-Key`)가 누락되어 401 발생.

**목적**
- 면접 평가(리포트) 호출 정상화.

---

## 12. FastApiInterviewEvaluationRequest DTO 수정

| 항목 | 내용 |
|------|------|
| **관련 커밋** | [1194c94](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/1194c94) (DTO에 roomId, userId 추가 #112), [260990e](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/260990e) (FastAPI 422 확인용 로그 추가 #112) |

**처리 내용**
- AI Server 면접 평가 요청 DTO에 `roomId`, `userId` 필드가 누락되어 422 발생.
- DTO에 필드 추가 및 422 응답 시 요청/응답 body 상세 로깅 추가.

**목적**
- BE → AI Server 면접 평가 요청 시 422 Validation Error 해결 및 디버깅 용이.

---

## 13. 면접 모드 종료 시 메시지 저장 누락

| 항목 | 내용 |
|------|------|
| **커밋** | [ef72616](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/ef72616) |
| **메시지** | fix: 면접 모드 종료 시 일부 메시지 저장 누락 및 상태값 미반영 문제 해결 (#116) |

**처리 내용**
- 면접 종료 시 마지막 답변·리포트 메시지가 DB에 저장되지 않는 문제 수정.
- 면접 `status` 값이 `COMPLETED`로 업데이트되지 않는 문제 해결.

**목적**
- 면접 종료 후 채팅방 재진입 시 이전 대화·평가 결과 확인 가능.

---

## 14. 재접속 시 면접 모드 재개 불가 문제

| 항목 | 내용 |
|------|------|
| **관련 커밋** | [898c163](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/898c163) (면접 모드 재개 불가 해결 #132), [6f1a5f6](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/6f1a5f6) (마지막 답변 누락 해결 #132) |

**처리 내용**
- 채팅방 재접속 시 면접 세션이 `IN_PROGRESS` 상태인데도 면접 모드로 복귀하지 못하는 문제.
- 세션 상태 기반으로 면접 모드 자동 복귀 로직 추가.
- 마지막 답변이 히스토리에서 누락되는 off-by-one 문제 수정.

**목적**
- 네트워크 끊김·새로고침 후에도 면접 세션 유지.

---

## 15. AI 챗봇 메시지 정렬 오류 및 off-by-one

| 항목 | 내용 |
|------|------|
| **관련 커밋** | [d088bd7](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/d088bd7) (메시지 정렬 오류 수정), [9b305c1](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/9b305c1) (마지막 메시지 누락 해결) |

**처리 내용**
- 채팅방 재입장 시 메시지 무작위 정렬 문제 → `createdAt` ASC 정렬 적용.
- 페이지네이션 경계에서 마지막 메시지가 빠지는 off-by-one → 쿼리 조건(`<` → `<=`) 수정.

**목적**
- 채팅 히스토리 표시 순서 보장 및 메시지 누락 방지.

---

## 16. 5번째 질문 답변 시 401 발생

| 항목 | 내용 |
|------|------|
| **커밋** | [49c09a5](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/49c09a5) |
| **메시지** | fix: 5번째 질문 답변 시 401 발생 해결 (#136) |

**처리 내용**
- 면접 5번째(마지막) 질문 답변 시 평가 API 호출 과정에서 인증 토큰이 누락되어 401 발생.
- 평가 호출 시 인증 컨텍스트 전파 로직 수정.

**목적**
- 면접 마지막 질문 정상 평가 처리.

---

## 17. 6번째 질문 생성 방지 (Off-by-one)

| 항목 | 내용 |
|------|------|
| **관련 커밋** | [187ba1d](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/187ba1d) (6번째 질문 생성 방지 #136), [216ef15](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/216ef15) (5번째 답변 검증 수정 #136), [ecb9e81](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/ecb9e81) (개수 증가 로직 위치 변경 #136), [3384d8e](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/3384d8e) (재접속 시 6번째 질문 해결 #136) |

**처리 내용**
- 면접 5문 완료 후 AI Server에 불필요한 6번째 질문 생성 요청이 나가는 문제.
- `question_count` 체크 로직 추가 및 증가 타이밍 수정.
- 5번째 질문에 답변할 수 없던 검증 로직 버그 수정 (`<` → `<=`).
- 면접 정상 종료 후 재접속 시에도 6번째 질문이 생성되는 문제 추가 해결.

**목적**
- 면접 종료 조건(5문 완료)의 정확한 트리거 및 off-by-one 에러 해소.

---

## 18. 면접 질문 생성 실패 시 카운트 방지

| 항목 | 내용 |
|------|------|
| **커밋** | [ca109bc](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/ca109bc) |
| **메시지** | fix: 면접 질문 생성 실패 시 질문 개수 카운트하지 않게 변경 (#149) |

**처리 내용**
- AI Server에서 질문 생성에 실패(SSE 에러 이벤트)했는데 BE에서 `question_count`가 증가하는 문제.
- 성공 응답 수신 후에만 카운트 증가하도록 수정.

**목적**
- 질문 생성 실패 시 면접이 조기 종료되는 문제 방지.

---

# Part 4.
## 보안·DTO 검증·기타 보강


## 19. DTO Validation 강화

| 항목 | 내용 |
|------|------|
| **관련 커밋** | [ba44e81](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/ba44e81) (DocumentAnalysisRequest cascade Valid), [36788ea](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/36788ea) (Size 어노테이션 범위 변경), [54c1e4e](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/54c1e4e) (이력서/채용공고/닉네임 길이 제한) |

**처리 내용**
- 중첩 DTO에 `@Valid` cascade 적용으로 하위 객체 검증 누락 방지.
- `@Size` 어노테이션 범위 수정으로 입력 길이 제한 정확히 적용.

**목적**
- 잘못된 요청이 AI Server까지 도달하기 전에 BE에서 422로 차단.

---

## 20. Log Injection 방어

| 항목 | 내용 |
|------|------|
| **관련 커밋** | [49b71cd](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/49b71cd) (todoId 로그 제거 #118), [72cc914](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/72cc914) (eventId 로그 제거 #99) |

**처리 내용**
- 사용자 입력 값(todoId, eventId)이 로그에 직접 기록되는 Log Injection 취약점 제거.

**목적**
- CodeQL 코드 스캔 알림 대응 및 보안 강화.

---

## 21. 분석 완료 알림 및 채팅방 제목 업데이트

| 항목 | 내용 |
|------|------|
| **관련 커밋** | [766074e](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/766074e) (summary → title 업데이트 #120), [6b873a3](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/6b873a3) (분석 완료 알림 읽음 처리 #150) |

**처리 내용**
- AI Server 분석 결과의 `summary` 필드로 채팅방 제목 자동 업데이트.
- 채팅방 바로 입장 시에도 분석 완료 알림 읽음 처리.

**목적**
- 채팅방 목록에서 분석 대상(회사/직무) 즉시 식별 가능.

---

## 22. 분석 시 500 오류 및 Rate Limit

| 항목 | 내용 |
|------|------|
| **관련 커밋** | [d949056](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/d949056) (분석 시도 시 500), [e16daa0](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/e16daa0) (Rate Limit 기능 추가 #131) |

**처리 내용**
- 분석 API 호출 시 500 에러 수정.
- API Rate Limit 기능 추가로 과다 요청 방지.

**목적**
- 서버 안정성 확보 및 AI Server 과부하 방지.

---

## 요약: 오류·기능 유형별 대응

| 유형 | 대응 단계 | 조치 요약 |
|------|-----------|-----------|
| **SSE 스트리밍** | §4, §7~§9 | 챗봇 SSE 중계, Access Denied 수정, ServerSentEvent 자동 파싱 |
| **채팅 기반** | §1, §2 | AI 채팅방·메시지 API, 메시지 조회 정렬(ASC) |
| **분석 연동·비동기** | §3, §5 | FastAPI 분석 연동, 비동기 Lazy, taskId 통합, OCR 저장 |
| **404/500·매핑** | §5, §22 | 분석 URI 매핑, 404 재시도, 분석 500 수정, Rate Limit |
| **면접 401 인증** | §11, §12, §16 | API Key 누락, DTO 필드 보강, 인증 컨텍스트 전파 |
| **면접 off-by-one** | §15, §17, §18 | 6번째 질문 방지, question_count 타이밍, 실패 시 카운트 방지 |
| **면접 세션 복구** | §10, §13, §14 | 상태값 반영, 재접속 시 자동 복귀, 메시지 저장 누락 |
| **DTO 검증** | §12, §19 | cascade Valid, Size 범위, 422 로깅 |
| **보안** | §20 | Log Injection 방어 (user-provided value 제거) |

