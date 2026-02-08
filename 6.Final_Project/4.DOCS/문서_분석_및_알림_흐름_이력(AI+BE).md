# 문서 분석 및 알림 흐름 이력 (AI + BE)

9-team-Devths 프로젝트에서 **이력서·채용공고 문서 분석** 및 **비동기 분석 완료 알림**과 관련하여 AI·BE 2개 저장소에서 진행한 작업을 **시간 흐름순**으로 정리한 문서입니다.

> **핵심 이슈:** 비동기 분석 완료 알림 시스템, AI 분석 실패 시 fallback 처리, DocumentAnalysisRequest DTO·검증

- **AI 저장소**: [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI)
- **BE 저장소**: [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE)

---

# Part 1.
## 분석 API 연동 기반 (BE ↔ AI)

## 1. 이력서/채용 공고 분석 FastAPI 연동 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [4c35c03](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/4c35c031cae9e118520e02dc65527ae97e5d4227) |
| **메시지** | feat: 이력서/채용 공고 분석 FastAPI와 연동 (#55) |

**처리 내용**
- 이력서·채용 공고 분석을 **FastAPI(AI 서버)** 와 연동.
- 분석 요청·비동기 task 생성·폴링·결과 수신 플로우 구현.
- 이후 404/500/비동기 오류 대응 및 **분석 완료 알림**의 토대가 됨.

**목적**
- BE ↔ AI 서버 간 문서 분석 API 연동 및 오류·재시도·알림 연동 가능한 구조 확보.

---

## 2. 이력서 분석 비동기 처리 self-invocation Lazy 처리 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [65460a8](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/65460a83c9d99dd4ddb910a354bbbb8d1649c31f) |
| **메시지** | feat: 이력서 분석 비동기 처리 self-invocation Lazy 처리 (#55) |

**처리 내용**
- 이력서 분석 **비동기 처리** 시 같은 빈 내부 호출(self-invocation)로 트랜잭션·프록시 이슈가 나지 않도록 **Lazy** 방식으로 호출하도록 수정.

**목적**
- 비동기 작업이 DB 커밋 전에 조회되며 발생하던 **“비동기 작업을 찾을 수 없습니다”** 유형 오류 제거 → 분석 완료 알림·폴링 안정화.

---

## 3. “비동기 작업을 찾을 수 없습니다” 문제 해결 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [0ca1689](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/0ca1689783b2d26f30d6c69a7e7e5178c07b5554) |
| **메시지** | fix: '비동기 작업을 찾을 수 없습니다' 문제 해결을 위해 DB 커밋 전에 task를 조회하던 로직 수정 |

**처리 내용**
- 비동기 분석 **task**를 DB **커밋 전**에 조회해 “비동기 작업을 찾을 수 없습니다”가 나던 문제 수정.
- 트랜잭션 커밋 순서·task 조회 시점을 조정해 폴링 시 task가 항상 조회되도록 함.

**목적**
- 분석 결과 폴링·**분석 완료 알림** 연동 시 404/무한 폴링 원인 제거.

---

# Part 2.
## AI 오프닝 메시지·분석 단계별 분할·fallback (AI)

## 4. 분석 후 AI 오프닝 메시지 생성 기능 추가 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [f27a05d](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/f27a05d), [ef757bd](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/ef757bd) |
| **메시지** | feat: 분석 후 AI 오프닝 메시지 생성 기능 추가 |

**처리 내용**
- 이력서·채용공고 **분석이 완료된 뒤** 대화 시작 시 사용자에게 보여줄 **오프닝 메시지**를 AI가 생성하도록 기능 추가.
- 분석 결과 컨텍스트를 바탕으로 “OO회사 OO 직무에 지원하시는군요…” 등 맞춤 인사·요약 문장 생성.
- 텍스트 추출·분석 파이프라인 내에서 오프닝 메시지 생성 실패 시 로깅·fallback 처리로 빈 맥락 방지.

**목적**
- 문서 분석 완료 후 채팅 진입 시 사용자 경험 개선 및 분석 결과를 자연어로 안내.

---

## 5. Gemini 빈 응답 해결 및 분석 API 단계별 분할 호출 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [4af8155](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/4af8155) |
| **메시지** | fix: Gemini 빈 응답 문제 해결 - 분석 API 단계별 분할 호출 |

**처리 내용**
- **분석 API를 단계별 분할 호출**로 변경: 이력서 분석 → 채용공고 분석 → 매칭도 분석 순으로 각각 별도 LLM 호출.
- 한 번에 긴 분석을 요청하던 방식에서 **3단계(이력서/채용공고/매칭)** 로 나누어 타임아웃·빈 응답 위험 감소.
- Gemini 빈 응답 시 None 체크·재시도 로직 보강(1ff6498 등과 연계).

**목적**
- 분석 구간에서 LLM 불안정으로 인한 스트리밍 중단·500 오류 감소 및 오프닝 메시지 품질 확보.

---

## 6. 분석 API 안정 모델·재시도 로직 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [03f1eaf](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/03f1eaf), [1ff6498](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/1ff6498) |
| **메시지** | fix: 분석 API에 안정 모델(gemini-2.0-flash) 사용 및 재시도 로직 추가 / fix: Gemini 빈 응답 시 None 체크 추가 |

**처리 내용**
- 분석 구간에 **안정 모델(gemini-2.0-flash)** 적용 및 **재시도 로직** 추가.
- Gemini 응답이 None/빈 문자열일 때 None 체크 후 재시도 또는 fallback 반환.

**목적**
- 분석 실패 시 사용자에게 빈 화면이 나오지 않도록 방어.

---

## 7. 분석 결과 JSON 파싱 및 fallback·빈 맥락 방지 (AI)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-AI](https://github.com/100-hours-a-week/9-team-Devths-AI) |
| **커밋** | [93e0033](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/93e0033) (분석 결과 JSON 파싱, Gemini JSON 모드), [414ed2d](https://github.com/100-hours-a-week/9-team-Devths-AI/commit/414ed2d) (분석 실패 시 fallback·빈 맥락 방지) |

**처리 내용**
- 분석 결과 **JSON 파싱** 시 Gemini JSON 모드 적용·파싱 실패 시 **fallback** 구조 반환.
- **분석 실패 시 fallback** 메시지 및 **오프닝 메시지에 분석 내용이 비어 보이는 것** 방지(빈 맥락 방지).

**목적**
- AI 분석 실패 시에도 BE·FE에 일관된 구조 전달 및 오프닝 메시지 빈 값 방지.

---

# Part 3.
## DocumentAnalysisRequest DTO·검증 (BE)

## 8. DocumentAnalysisRequest DTO 수정·Validation 강화 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [ba44e81](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/ba44e81) (DocumentAnalysisRequest cascade Valid), [36788ea](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/36788ea) (Size 어노테이션 범위 변경), [54c1e4e](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/54c1e4e) (이력서/채용공고/닉네임 길이 제한) |

**처리 내용**
- **DocumentAnalysisRequest** 및 중첩 **DocumentInfo**에 `@Valid` cascade 적용으로 하위 객체 검증 누락 방지.
- `@Size` 어노테이션 범위 수정으로 이력서·채용공고·닉네임 등 **입력 길이 제한** 정확히 적용.
- 잘못된 요청이 AI Server까지 도달하기 전에 BE에서 **422**로 차단.

**목적**
- 분석 요청 스키마·길이 제한 통일로 422/500 예방 및 AI 부하 완화.

---

## 9. jobPost DTO 필드명 매칭 문제 해결 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [a8d769f](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/a8d769f7b4e32e2b2b315916b1e2a9037fc708f6) |
| **메시지** | fix: jobPost DTO 필드명 매칭 문제 해결 (#68) |

**처리 내용**
- 채용 공고 분석 시 **jobPost DTO** 필드명이 AI 서버 또는 내부 스키마와 불일치해 발생하던 오류(422/500 유사) 수정.

**목적**
- 분석 요청 시 DTO 매핑 오류로 인한 실패 제거.

---

# Part 4.
## 분석 완료 알림·채팅방 제목 업데이트 (BE)

## 10. 분석 완료 알림 읽음 처리 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [6b873a3](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/6b873a3) |
| **메시지** | 분석 완료 알림 읽음 처리 (#150) |

**처리 내용**
- **비동기 분석 완료** 시 사용자에게 알림을 보내고, 사용자가 채팅방에 **입장하거나 알림을 확인**하면 **읽음 처리**되도록 구현.
- 채팅방에 **바로 입장**한 경우에도 분석 완료 알림이 읽음 처리되도록 로직 보완.

**목적**
- 비동기 분석 완료 알림 시스템의 일관된 상태 관리 및 중복 알림·미읽음 잔존 방지.

---

## 11. 채팅방 제목 업데이트 (summary → title) (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [766074e](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/766074e) |
| **메시지** | summary → title 업데이트 (#120) |

**처리 내용**
- AI Server 분석 결과의 **summary** 필드(회사명·직무 등)로 **채팅방 제목** 자동 업데이트.
- 분석 완료 후 채팅방 목록에서 분석 대상(회사/직무)을 즉시 식별할 수 있도록 함.

**목적**
- 분석 완료 알림·채팅 진입 시 UX 일관성 및 채팅방 목록 가독성 향상.

---

## 12. ExternalTaskId 제거 및 백엔드 taskId 통합 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [342aced](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/342aced90545537540c9426f1922fb9fe0373f68) |
| **메시지** | feat: ExternalTaskId 제거 후 백엔드 taskId로 통합 (#70) |

**처리 내용**
- **ExternalTaskId** 제거 후 **백엔드 taskId** 하나로 통일.
- 클라이언트·AI 서버와의 task 식별자 계약 단순화로 분석 요청·폴링·알림 연동 시 매핑 오류 감소.

**목적**
- 분석 완료 알림·폴링 시 task 조회 실패·404 원인 제거.

---

## 13. OCR 데이터 저장 로직 추가 (BE)

| 항목 | 내용 |
|------|------|
| **저장소** | [9-team-Devths-BE](https://github.com/100-hours-a-week/9-team-Devths-BE) |
| **커밋** | [fc5c743](https://github.com/100-hours-a-week/9-team-Devths-BE/commit/fc5c7434f7c11745a853141a369f7d3e7079b815) |
| **메시지** | feat: OCR 데이터 저장 로직 추가 (#70) |

**처리 내용**
- 분석 결과 중 **OCR 결과**를 DB에 저장하는 로직 추가.
- 분석 플로우 완결 및 결과 조회·재활용·알림 연동 시 데이터 일관성 확보.

**목적**
- 분석 500/데이터 부재로 인한 후속 오류 감소 및 분석 이력·알림 정합성 유지.

---

# 요약: 문서 분석·알림 유형별 대응

| 유형 | 담당 | 대응 커밋/내용 | 조치 요약 |
|------|------|----------------|-----------|
| **분석 API 연동** | BE | 4c35c03, 65460a8, 0ca1689 | FastAPI 연동, 비동기 Lazy, task 조회 시점 수정 |
| **오프닝 메시지** | AI | f27a05d, ef757bd | 분석 완료 후 AI 오프닝 메시지 생성 |
| **분석 단계별 분할·fallback** | AI | 4af8155, 03f1eaf, 1ff6498, 93e0033, 414ed2d | 단계별 분할 호출, 재시도·None 체크·JSON fallback·빈 맥락 방지 |
| **DocumentAnalysisRequest DTO** | BE | ba44e81, 36788ea, 54c1e4e, a8d769f | cascade Valid, Size 범위, 필드명 매칭 |
| **분석 완료 알림** | BE | 6b873a3, 766074e | 알림 읽음 처리, summary→채팅방 제목 업데이트 |
| **taskId 통합·OCR 저장** | BE | 342aced, fc5c743 | taskId 단일화, OCR 결과 저장 |

---

# 회고 포인트

## 비동기 분석 완료 알림 시스템

- BE가 분석 요청 후 **비동기 task**를 생성하고, AI 서버 분석 완료 시 **결과 저장·알림 생성**까지 한 흐름으로 처리.
- **분석 완료 알림 읽음 처리**(6b873a3)로 “채팅방 입장 시·알림 확인 시” 읽음 상태가 일관되게 반영됨.
- task 조회 시점(0ca1689), taskId 통합(342aced)으로 폴링·알림 연동 시 404/미조회 이슈를 줄임.

## AI 분석 실패 시 fallback 처리

- AI 측에서 **단계별 분할 호출**(4af8155), **재시도·안정 모델**(03f1eaf, 1ff6498), **JSON 파싱 fallback·빈 맥락 방지**(93e0033, 414ed2d)를 적용해, 분석 일부 실패 시에도 **fallback 구조**를 반환하고 오프닝 메시지가 비어 보이지 않도록 함.
- BE는 DocumentAnalysisRequest **검증 강화**(ba44e81, 54c1e4e)로 잘못된 요청을 422로 차단해 AI 부하·실패 경로를 줄임.

---

## 문서 이력

| 날짜 | 내용 |
|------|------|
| (초안) | AI·BE 저장소 문서 분석·알림 관련 커밋 반영, SSE·면접 통합 이력 문서 형식에 맞춰 작성 |

---

*이 문서는 [SSE_스트리밍_구현_및_오류처리_이력(AI+BE+FE).md](./SSE_스트리밍_구현_및_오류처리_이력(AI+BE+FE).md), [SSE_스트리밍_오류처리_이력(AI).md](./SSE_스트리밍_오류처리_이력(AI).md), [SSE_스트리밍_오류처리_이력(BE).md](./SSE_스트리밍_오류처리_이력(BE).md)와 함께 참고합니다.*
