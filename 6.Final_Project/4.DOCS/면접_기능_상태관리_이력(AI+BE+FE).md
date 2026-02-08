# 면접 기능 상태 관리 이력 (AI + BE + FE)

9-team-Devths 프로젝트에서 **면접 기능 상태 관리**와 관련하여 AI·BE·FE 3개 저장소에서 진행한 작업을 **시간 흐름순**으로 정리한 문서입니다.

> **핵심 이슈:** 면접 세션 상태 동기화(AI-BE-FE), Off-by-one 에러 및 인덱싱 문제, 면접 종료 조건 및 평가 트리거

---

# Part 1.
## 면접 기능 기반 구축

## 1. 면접 모드 시작/종료 구현 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [e71963f](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/e71963f) |
| **메시지** | feat: 면접 모드 시작/종료 구현 (#66) |

**처리 내용**
- 면접 세션 생성·종료 API 구현: `interview_status` 상태 관리 (READY → IN_PROGRESS → COMPLETED).
- 면접 시작 시 AI Server에 `interview_question` 모드로 SSE 요청, 종료 시 `interview_report` 모드로 리포트 요청.
- 면접 세션 엔티티에 `question_count`, `current_question_id` 필드 추가.

**목적**
- 면접 플로우(시작 → 질문 → 답변 → 종료 → 평가)의 BE 측 상태 관리 기반 구축.

---

## 2. 면접 모드 상태와 API 연동 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **커밋** | [1a1a263](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/1a1a263) |
| **메시지** | feat: 면접 모드 상태와 API 연동 |

**처리 내용**
- FE에서 면접 시작·종료 API 호출 및 UI 상태 관리.
- 면접 모드 진입 시 일반 채팅 입력 비활성화, 면접 전용 UI 전환.
- 면접 진행 상태(질문 대기/답변 입력/평가 중)에 따른 화면 분기.

**목적**
- BE의 면접 세션 상태를 FE에서 정확히 반영하여 UX 일관성 확보.

---

## 3. 기술면접 프롬프트 템플릿 및 질문 생성 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **관련 커밋** | de43876 (기술면접 프롬프트 템플릿), 098ca9d (질문 예시 프롬프팅), aabf9c7 (면접 데이터셋 통합) |

**처리 내용**
- 기술면접 프롬프트 템플릿 추가: 이력서·채용공고 기반 질문 생성 가이드.
- 면접 질문 예시를 프롬프트에 포함시켜 LLM이 적절한 난이도·형식으로 질문 생성.
- InterView_Datasets 통합으로 질문 품질 향상.

**목적**
- 면접 질문 품질 및 일관성 확보.

---

## 4. 면접 5회 완료 후 자동 피드백 생성 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [2a6a988](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/2a6a988) |
| **메시지** | feat: 면접 모의 5회 완료 후 자동 피드백 생성 |

**처리 내용**
- 면접 질문 5회 진행 완료 감지 시 자동으로 피드백(리포트) 생성 트리거.
- 전체 Q&A 히스토리를 수집하여 LLM에 리포트 생성 요청.

**목적**
- 면접 종료 시점에서 BE의 평가 요청 없이도 AI가 자동으로 피드백을 준비.

---

# Part 2.
## 면접 질문 생성 안정화 (AI)

## 5. 면접 질문 JSON 파싱 로직 개선 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **관련 커밋** | e8f1aea (JSON 파싱 로직 개선), 5b64cea (JSON 파싱 실패 수정), 68212a0 (질문 길이 제한) |

**처리 내용**
- LLM이 생성한 면접 질문 JSON을 파싱할 때 markdown 코드블록(` ```json `)이 포함되어 파싱 실패하는 문제 수정.
- JSON 외 텍스트가 섞인 경우에도 JSON 부분만 추출하는 로직 추가.
- 질문 텍스트 길이 제한 추가로 LLM 과다 응답 방지.

**목적**
- 면접 질문 생성 시 파싱 오류로 인한 SSE 에러(`PARSE_FAILED`) 감소.

---

## 6. 면접 질문 생성 예외 처리 개선 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [851a5fb](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/851a5fb) |
| **메시지** | feat: 면접 질문 생성 예외 처리 개선 |

**처리 내용**
- 면접 질문 생성 실패 시 SSE 에러 이벤트(`PARSE_FAILED`)로 안정적 전달.
- try-except 블록 보강으로 미처리 예외 방지.

**목적**
- 질문 생성 실패가 전체 스트리밍을 중단시키지 않도록 방어.

---

## 7. 면접 질문 생성 속도 개선 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [8d36874](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/8d36874) |
| **메시지** | feat: 면접 질문 생성 속도 개선 및 타이핑 효과 추가 |

**처리 내용**
- 면접 질문 생성 시 LLM 호출 최적화로 응답 속도 개선.
- 클라이언트에 타이핑 효과용 청크 단위 전송 추가.

**목적**
- 면접 질문 생성 대기 시간 단축 (사용자 체감 속도 개선).

---

## 8. 세션 캐시 및 비스트리밍 LLM 응답 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [8ffda2f](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/8ffda2f) |
| **메시지** | feat: 세션 캐시 및 비스트리밍 LLM 응답 추가 |

**처리 내용**
- 면접 세션 캐시 도입: 세션별 질문·답변 히스토리를 메모리에 캐싱하여 매번 DB 조회 불필요.
- 질문 세트 생성 등 구간에 비스트리밍(non-stream) LLM 호출 적용으로 JSON 파싱 안정화.

**목적**
- 면접 초기화·질문 생성 구간 안정화 및 응답 형식 일관성.

---

## 9. 꼬리질문 평가 로직 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [9c2dfe8](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/9c2dfe8) |
| **메시지** | feat: 면접 꼬리질문 평가 로직 개선 |

**처리 내용**
- 사용자 답변의 깊이·완성도를 평가하여 꼬리질문 필요 여부 판단.
- 꼬리질문 생성 시 이전 답변 컨텍스트를 포함하여 연관성 있는 질문 생성.

**목적**
- 면접의 깊이를 높이고, 답변이 부족한 경우 추가 질문으로 보완 기회 제공.

---

## 10. 면접 질문 반복 방지 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [8e94ab4](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/8e94ab4) |
| **메시지** | feat(prompting): 면접 질문 반복 방지 + 서비스명 변경 |

**처리 내용**
- 이미 출제된 질문 목록을 프롬프트에 포함시켜 LLM이 중복 질문을 생성하지 않도록 함.
- 세션 캐시의 질문 히스토리를 활용.

**목적**
- 면접 5회 진행 중 동일 질문 반복 문제 해결.

---

# Part 3.
## 면접 평가 트리거 — 401/422 오류 (BE)

## 11. 면접 평가 요청 시 401 발생 (BE - 1차)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [144b8d9](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/144b8d9) |
| **메시지** | fix: 면접 평가 요청 시 401 발생 |

**처리 내용**
- 면접 리포트(평가) 요청 시 AI Server 호출에서 인증 헤더(`X-API-Key`)가 누락되어 401 발생.
- FastApiClient에 API Key 헤더 추가.

**목적**
- 면접 평가(리포트) 호출 정상화.

**회고**
- BE → AI Server 호출 시 인증 헤더 누락은 단순한 실수지만, 면접 종료 시점에만 트리거되어 늦게 발견됨.

---

## 12. FastApiInterviewEvaluationRequest DTO 수정 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **관련 커밋** | [1194c94](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/1194c94) (DTO에 roomId, userId 추가 #112), [260990e](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/260990e) (FastAPI 422 확인용 로그 #112) |

**처리 내용**
- AI Server 면접 평가 요청 DTO에 `roomId`, `userId` 필드가 누락되어 AI Server에서 422 발생.
- DTO에 필드 추가 및 422 응답 시 요청/응답 body 상세 로깅 추가.

**목적**
- BE ↔ AI Server 간 스키마 불일치로 인한 422 해결.

**회고**
- AI 측 스키마 변경이 BE DTO에 반영되지 않아 발생. API 명세 문서와 DTO 동기화의 중요성.

---

## 13. 5번째 질문 답변 시 401 발생 (BE - 2차)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [49c09a5](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/49c09a5) |
| **메시지** | fix: 5번째 질문 답변 시 401 발생 해결 (#136) |

**처리 내용**
- 면접 5번째(마지막) 질문 답변 시 평가 API 호출 과정에서 인증 토큰이 누락되어 401 발생.
- 평가 호출 시 인증 컨텍스트 전파(SecurityContext propagation) 로직 수정.

**목적**
- 면접 마지막 질문 답변 후 정상 평가 처리.

**회고**
- 비동기 호출 시 SecurityContext가 전파되지 않는 Spring 기본 동작이 원인. `@Async` + SecurityContext 전파 패턴 필요.

---

## 14. 면접 마지막 질문 응답 완료 후 평가 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **관련 커밋** | [8993997](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/8993997) (면접 마지막 질문 응답 완료 후 평가 #69), [642fc37](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/642fc37) (최종 답변 후 평가 호출 흐름 복구 #110) |

**처리 내용**
- 면접 마지막(5번째) 질문에 대한 응답이 SSE로 완료된 후 평가(리포트) API를 호출하는 타이밍 제어.
- 평가 호출 흐름이 중간에 끊기는 문제 복구.

**목적**
- 면접 종료 → 평가 트리거의 정확한 시점 보장.

---

# Part 4.
## Off-by-one 에러 — 6번째 질문 생성 방지 (BE)

## 15. 6번째 질문 생성 방지

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **관련 커밋** | [187ba1d](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/187ba1d) (6번째 질문 생성 방지 #136), [216ef15](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/216ef15) (5번째 답변 검증 수정 #136), [ecb9e81](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/ecb9e81) (개수 증가 로직 위치 변경 #136) |

**처리 내용**
- 면접 5문 완료 후 AI Server에 불필요한 6번째 질문 생성 요청이 나가는 문제.
- **원인 1**: `question_count` 체크 시점이 답변 처리 전이 아닌 후에 있어, 5번째 답변 후 6번째 질문을 요청.
- **원인 2**: 5번째 질문에 답변 자체가 불가능한 검증 로직 버그 (`question_count < MAX` → `question_count <= MAX`).
- **수정**: 카운트 증가 타이밍을 질문 생성 성공 후로 변경, 검증 조건 수정.

**목적**
- 면접 종료 조건(5문 완료)의 정확한 트리거.

**회고**
- 전형적인 off-by-one 에러. "5문까지 답변 가능" vs "5문까지 질문 생성 가능"의 경계 조건이 BE·AI 간에 다르게 해석되어 발생.

---

## 16. 재접속 시 6번째 질문 생성 문제 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [3384d8e](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/3384d8e) |
| **메시지** | fix: 면접 정상 종료 후 채팅방 재접속 시 6번째 질문 생성 문제 해결 (#136) |

**처리 내용**
- 면접 정상 종료(COMPLETED) 후 채팅방에 재접속하면 면접 세션 상태를 잘못 읽어 6번째 질문을 AI Server에 요청하는 문제.
- 종료된 세션에 대해서는 질문 생성을 시도하지 않도록 상태 체크 추가.

**목적**
- 종료된 면접에서 불필요한 AI Server 호출 방지.

---

## 17. 면접 질문 생성 실패 시 카운트 방지 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [ca109bc](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/ca109bc) |
| **메시지** | fix: 면접 질문 생성 실패 시 질문 개수 카운트하지 않게 변경 (#149) |

**처리 내용**
- AI Server에서 질문 생성에 실패(SSE 에러 이벤트)했는데 BE에서 `question_count`가 증가하는 문제.
- 성공 응답 수신 후에만 카운트 증가하도록 수정.

**목적**
- 질문 생성 실패 시 면접이 조기 종료되는 문제 방지.

**회고**
- "요청했다 = 질문이 생성됐다"로 간주하면 안 됨. AI Server 응답을 확인한 후에만 상태를 변경해야 함.

---

## 18. 면접 모드 마지막 답변 중복 방지 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **커밋** | [d31c8c4](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/d31c8c4) |
| **메시지** | fix: 면접 모드 마지막 답변을 /evaluation으로만 보내 중복 질문 방지 |

**처리 내용**
- 면접 마지막(5번째) 답변이 일반 채팅 API(`/chat`)와 평가 API(`/evaluation`) 양쪽으로 전송되어, 6번째 질문이 생성되고 답변이 중복 저장되는 문제.
- 마지막 답변은 `/evaluation` 엔드포인트로만 전송하도록 분기 처리.

**목적**
- 면접 종료 시점에서 불필요한 질문 생성 요청 차단 및 답변 중복 방지.

**회고**
- FE에서 "마지막 질문인지" 판단하는 로직이 BE의 `question_count`와 동기화되어야 함. 클라이언트-서버 간 상태 동기화의 중요성.

---

# Part 5.
## 면접 세션 동기화 — 재접속/복구 (BE + FE)

## 19. 면접 모드 종료 시 메시지 저장 누락 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [ef72616](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/ef72616) |
| **메시지** | fix: 면접 모드 종료 시 일부 메시지 저장 누락 및 상태값 미반영 문제 해결 (#116) |

**처리 내용**
- 면접 종료 시 마지막 답변·리포트 메시지가 DB에 저장되지 않는 문제 수정.
- 면접 `status` 값이 `COMPLETED`로 업데이트되지 않는 문제 해결.

**목적**
- 면접 종료 후 채팅방 재진입 시 이전 대화·평가 결과 확인 가능.

---

## 20. 재접속 시 면접 모드 재개 불가 문제 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **관련 커밋** | [898c163](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/898c163) (면접 모드 재개 불가 해결 #132), [6f1a5f6](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/6f1a5f6) (마지막 답변 누락 해결 #132) |

**처리 내용**
- 채팅방 재접속 시 면접 세션이 `IN_PROGRESS` 상태인데도 면접 모드로 복귀하지 못하는 문제.
- 세션 상태 기반으로 면접 모드 자동 복귀 로직 추가.
- 마지막 답변이 히스토리에서 누락되는 off-by-one 문제 수정.

**목적**
- 네트워크 끊김·새로고침 후에도 면접 세션 유지.

---

## 21. current interview 조회 연동 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **커밋** | [872e83d](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/872e83d) |
| **메시지** | feat: current interview 조회 연동 (#92) |

**처리 내용**
- 채팅방 진입 시 현재 진행 중인 면접 세션이 있는지 BE에 조회.
- 진행 중 세션이 있으면 면접 모드로 자동 복귀, 없으면 일반 채팅 모드.

**목적**
- 페이지 새로고침·재접속 시에도 면접 상태가 유지되는 UX 구현.

---

## 22. 면접 진행 중 이탈 차단 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **관련 커밋** | [8f3b6fd](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/8f3b6fd) (면접 진행 중 이탈 차단 #110), [24494a2](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/24494a2) (채팅 스트리밍 중 이동 차단 #110), [d68d4d2](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/d68d4d2) (면접 종료 확인 안내 추가 #110), [665782a](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/665782a) (면접 종료 확인 안내 삭제 #110) |

**처리 내용**
- 면접 진행 중 브라우저 뒤로가기/페이지 이동 시 확인 모달 표시.
- 채팅 스트리밍(SSE) 수신 중에도 이동 차단.
- 면접 종료 확인 안내 UI 추가 후 UX 피드백 반영하여 삭제.

**목적**
- 면접 중 실수로 이탈하여 세션 상태가 꼬이는 문제 방지.

**회고**
- 면접 종료 확인 안내를 추가했다가 UX가 과하다는 피드백으로 삭제. 이탈 차단만 유지.

---

## 23. 스트리밍 요청 401 시 토큰 재발급 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **커밋** | [e1dc40b](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/e1dc40b) |
| **메시지** | fix: 스트리밍 요청 401 시 토큰 재발급 (#92) |

**처리 내용**
- SSE 스트리밍 요청 중 401 응답을 받으면 리프레시 토큰으로 자동 재발급 후 재요청.
- 면접 진행 중 토큰 만료 시에도 세션이 끊기지 않도록 함.

**목적**
- 면접 중 인증 만료로 인한 세션 중단 방지.

---

## 24. 답변 생성 중 입력 시 토스트 안내 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **커밋** | [59a34f5](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/59a34f5) |
| **메시지** | feat: 답변 생성 중 입력 시 토스트 안내 (#107) |

**처리 내용**
- AI가 SSE 스트리밍으로 응답을 생성 중일 때 사용자가 입력하면 토스트 메시지로 "답변 생성 중입니다" 안내.

**목적**
- 면접 중 AI 응답 대기 상태를 사용자에게 명확히 전달.

---

# Part 6.
## SSE 스트리밍 연동 보강 (FE)

## 25. SSE CRLF 줄바꿈 파싱 보정 (FE)

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

---

## 26. SSE 타이핑 공백 렌더링 보존 (FE)

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

---

## 27. 스트리밍 응답 타이핑 애니메이션 (FE)

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

## 28. LLM 채팅 SSE 전환 및 요청 스키마 반영 (FE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-FE](https://github.com/100-hours-a-week/9-team-Devths-FE) |
| **커밋** | [2425b46](https://github.com/100-hours-a-week/9-team-Devths-FE/commit/2425b46) |
| **메시지** | feat: LLM 채팅 SSE 전환 및 요청 스키마 반영 |

**처리 내용**
- 기존 일반 REST 호출 방식에서 SSE 스트리밍 방식으로 전환.
- BE 채팅 API 스키마(mode, context, session_id 등) 반영.

**목적**
- 면접·채팅 모두 SSE 스트리밍으로 통일된 응답 수신.

---

# 요약: 이슈 유형별 대응

## 면접 세션 상태 동기화 (AI-BE-FE)

| 이슈 | 원인 | 해결 | 관련 단계 |
|------|------|------|-----------|
| 재접속 시 면접 모드 복귀 불가 | BE 세션 상태 체크 누락 | 세션 상태 기반 자동 복귀 | §20, §21 |
| 면접 종료 시 메시지 저장 누락 | DB 커밋 타이밍 문제 | 트랜잭션 순서 조정 | §19 |
| 면접 중 이탈로 세션 꼬임 | FE 이탈 미차단 | 이탈 차단 모달 | §22 |
| FE-BE 면접 상태 불일치 | current interview 미조회 | 진입 시 상태 조회 API | §21 |

## Off-by-one 에러 및 인덱싱 문제

| 이슈 | 원인 | 해결 | 관련 단계 |
|------|------|------|-----------|
| 6번째 질문 생성 | `question_count < MAX` (off-by-one) | `<=` 수정, 카운트 타이밍 변경 | §15 |
| 5번째 질문 답변 불가 | 검증 로직 `<` vs `<=` | 검증 조건 수정 | §15 |
| 재접속 시 6번째 질문 | COMPLETED 세션 체크 누락 | 종료 세션 질문 생성 차단 | §16 |
| 질문 생성 실패인데 카운트 증가 | 요청 시점에 카운트 증가 | 성공 응답 후 카운트 | §17 |
| 마지막 답변 중복 전송 (FE) | chat + evaluation 이중 호출 | evaluation만으로 분기 | §18 |
| 마지막 메시지 누락 (BE) | 페이지네이션 경계 `<` | `<=` 수정 | §20 |

## 면접 종료 조건 및 평가 트리거

| 이슈 | 원인 | 해결 | 관련 단계 |
|------|------|------|-----------|
| 평가 요청 401 (1차) | API Key 헤더 누락 | 헤더 추가 | §11 |
| 평가 요청 422 | DTO 필드 누락 (roomId, userId) | DTO 수정 | §12 |
| 5번째 답변 시 401 (2차) | SecurityContext 미전파 | 컨텍스트 전파 수정 | §13 |
| FE 평가 호출 타이밍 | 마지막 SSE 완료 전 호출 | SSE 완료 후 트리거 | §14 |
| 스트리밍 중 토큰 만료 | 리프레시 미처리 | 401 시 자동 재발급 | §23 |

---

## 문서 이력

| 날짜 | 내용 |
|------|------|
| 2026-02-08 | AI·BE·FE 3개 저장소 면접 관련 커밋 분석 및 면접 기능 상태 관리 문서 초안 작성 |

---

*이 문서는 [SSE_스트리밍_오류처리_이력(AI).md](./SSE_스트리밍_오류처리_이력(AI).md), [SSE_스트리밍_오류처리_이력(BE).md](./SSE_스트리밍_오류처리_이력(BE).md), [SSE_스트리밍_오류처리_이력(FE).md](./SSE_스트리밍_오류처리_이력(FE).md)와 함께 참고합니다.*
