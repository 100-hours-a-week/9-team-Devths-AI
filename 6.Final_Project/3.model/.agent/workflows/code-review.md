---
description: 코드 리뷰를 수행합니다. PR 번호 또는 파일 경로를 받아 코드 품질, 보안, 에러 처리, 성능을 분석합니다.
---

# 코드 리뷰 분석기

## 대상 결정

- 사용자 인수가 숫자면 PR 번호 → `gh pr diff <PR번호>`
- 사용자 인수가 파일 경로면 해당 파일 리뷰
- 인수 없으면 `git diff`로 staged/unstaged 변경 리뷰

## 리뷰 체크리스트

### 1. 코드 품질
- 함수/메서드가 단일 책임 원칙을 따르는가?
- 중복 코드가 있는가?
- 변수/함수 네이밍이 명확한가?
- 불필요한 주석이나 dead code가 있는가?

### 2. 보안
- 사용자 입력이 적절히 검증되는가?
- 민감 정보(API 키, 비밀번호)가 하드코딩되어 있지 않은가?
- SQL/Prompt Injection 방어가 되어 있는가?
- **Log Injection 방지:**
  - `request.*` 속성(session_id, user_id 등)이 `logger.*()` 에 직접 삽입되지 않는가?
  - 사용자 입력값은 로그에 넣지 않고, 서버 계산값만 로깅하는가?
  - `# SAST: request 기반 값은 로그에 넣지 않음` 주석 패턴 준수 여부

### 3. 에러 처리
- 예외가 적절히 처리되는가?
- 에러 메시지가 사용자 친화적인가?
- fallback 로직이 있는가?

### 4. 성능
- 불필요한 DB/API 호출이 있는가?
- N+1 쿼리 문제가 있는가?
- 적절한 캐싱이 적용되어 있는가?

### 5. Python/FastAPI 관련
- Pydantic 모델이 적절히 사용되는가?
- async/await 패턴이 올바르게 사용되는가?
- type hint가 적용되어 있는가?
- Ruff 린팅 규칙 준수 (line-length=100, double quotes)

### 6. 비동기/동시성
- `asyncio.create_task()`가 SSE generator 안에서 호출될 때, 커넥션 종료 시 task가 cancel될 수 있음
- fire-and-forget 패턴은 `BackgroundTasks` 또는 별도 task 관리 고려
- ChromaDB `add()` vs `upsert()` — 중복 ID 시 에러 vs 덮어쓰기 차이 확인

## 출력 형식

```markdown
## 코드 리뷰 결과

### 요약
- 전체 평가: [좋음/보통/개선필요]
- 리뷰 파일 수: N개

### 발견 사항

#### Critical (반드시 수정)
- [파일:라인] 설명

#### Warning (권장 수정)
- [파일:라인] 설명

#### Good (잘된 점)
- 설명

### 개선 제안
1. ...
2. ...
```
