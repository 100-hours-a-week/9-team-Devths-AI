---
description: 커밋 메시지를 생성하고 커밋합니다. conventional commit 포맷을 따릅니다.
---

# 커밋 메시지 생성기

## 커밋 작업 경로 (고정)

- **절대 경로**: `/Users/yoon-dong-gyu/kakao_bootcamp/6.Final_Project/3.model`
- **규칙**: 모든 git 명령은 **이 경로 안의 변경만** 대상으로 한다.
- **pathscope**: Git 루트에서 `3.model/` 또는 `6.Final_Project/3.model/` 경로로 제한

---

## 0단계: 브랜치 보호 체크 (최우선)

// turbo
```bash
git branch --show-current
```

### 보호 브랜치 목록 (직접 커밋 금지)
- `main`
- `develop`
- `release/*`

### 보호 브랜치에 있는 경우

**절대 보호 브랜치에 직접 커밋하지 않습니다.** 다음 절차를 따릅니다:

1. `git diff`와 `git diff --cached`로 변경 내용 분석
2. `git log --oneline -10`으로 최근 커밋 이력 확인
3. 변경 내용을 바탕으로 적절한 feature 브랜치명을 제안

**브랜치명 규칙:**
- 새 기능: `feature/{도메인}-{간단한-설명}`
- 버그 수정: `fix/{간단한-설명}`
- 설정/빌드: `chore/{간단한-설명}`
- 문서: `docs/{간단한-설명}`
- 긴급 수정: `hotfix/{간단한-설명}`

사용자에게 제안 후 확인받고 브랜치 생성:
```bash
git checkout -b feature/{브랜치명}
```

---

## 실행 순서

// turbo
1. **변경 파일 확인**: `git status -- <pathscope>`로 3.model 경로 안의 변경만 확인

// turbo
2. **staged/unstaged 확인**: `git diff --cached -- <pathscope>`로 staged 확인. staged가 없으면 `git diff -- <pathscope>`로 unstaged 확인

// turbo
3. **최근 커밋 스타일**: `git log --oneline -5 -- <pathscope>`로 참고

## 커밋 메시지 포맷

```
type: 한국어 설명
```

또는 scope가 있는 경우:

```
type(scope): 한국어 설명
```

### Type 접두사 (하나만 선택):
- `feat`: 새로운 기능 추가
- `fix`: 버그 수정
- `chore`: 빌드, 설정, 의존성 등 기능 외 변경
- `style`: 코드 스타일 변경 (포매팅, 세미콜론 등)
- `refactor`: 리팩토링 (기능 변경 없음)
- `docs`: 문서 변경
- `test`: 테스트 추가/수정
- `perf`: 성능 개선
- `ci`: CI/CD 설정 변경

## 규칙

- 첫 줄 72자 이내
- 설명은 반드시 한국어 (고유명사 제외: API, SSE, Docker 등은 영어 허용)
- **3.model 경로 안의 파일만** 개별 staging (`git add -A`, `git add .` 금지)
- .env, credentials, 대용량 바이너리 커밋 금지
- 사용자에게 초안을 보여주고 확인 후 커밋

### Staging 제외 패턴 (절대 커밋하지 않는 파일)
- `.agent/` 디렉토리
- `.code_review_result.md`
- `.pr_comments_draft.md`
- `__pycache__/`, `.pytest_cache/`
- `.env`, `.env.*`

---

## CI 사전 검증 (커밋 전 필수)

**하나라도 실패하면 커밋하지 않고, 자동으로 수정을 시도합니다.**

// turbo
### 1단계: Ruff Lint 검사
```bash
cd /Users/yoon-dong-gyu/kakao_bootcamp/6.Final_Project/3.model && poetry run ruff check app/
```

실패 시 자동 수정:
```bash
cd /Users/yoon-dong-gyu/kakao_bootcamp/6.Final_Project/3.model && poetry run ruff check app/ --fix
```

// turbo
### 2단계: Ruff Format 검사
```bash
cd /Users/yoon-dong-gyu/kakao_bootcamp/6.Final_Project/3.model && poetry run ruff format --check app/
```

실패 시 자동 포맷팅:
```bash
cd /Users/yoon-dong-gyu/kakao_bootcamp/6.Final_Project/3.model && poetry run ruff format app/
```

// turbo
### 3단계: Python Import 검증
```bash
cd /Users/yoon-dong-gyu/kakao_bootcamp/6.Final_Project/3.model && poetry run python -c "import app.main; print('Import OK')"
```

// turbo
### 4단계: 보안 패턴 체크 (자동 grep)

**변경 파일에서만 검사합니다.** staged 파일 목록을 기준으로:

```bash
# Log Injection: request/user 입력이 logger에 직접 들어가는지
git diff --cached --name-only -- '*.py' | xargs grep -n 'logger\.' | grep -i 'request\.' || echo 'Log Injection: PASS'
```

**위 grep에 매칭되면 WARNING**. 다음 패턴은 Log Injection 위험:
- `logger.info("...", request.session_id)` — 사용자 제공 값
- `logger.warning("... %s", request.user_id)` — 사용자 제공 값
- `logger.error(f"... {request.something}")` — f-string 직접 삽입

**안전한 대안:**
- 서버에서 계산한 값만 로깅: `result["count"]`, `len(data)` 등
- `# SAST: request 기반 값은 로그에 넣지 않음` 주석 패턴 따르기

추가 수동 확인:
- **SQL/Prompt Injection**: 사용자 입력이 f-string으로 쿼리/프롬프트에 직접 삽입되지 않는지
- **하드코딩된 시크릿**: API 키, 비밀번호가 코드에 직접 포함되지 않았는지
- **SSRF**: 사용자 입력이 URL에 직접 사용되지 않는지

### 검증 결과 출력
```
CI 사전 검증 결과:
- Ruff Lint:    [PASS/FAIL/AUTO-FIXED]
- Ruff Format:  [PASS/FAIL/AUTO-FIXED]
- Import 검증:  [PASS/FAIL]
- 보안 패턴:    [PASS/WARNING]
```

모두 PASS (또는 AUTO-FIXED)여야 커밋을 진행합니다.

---

## 커밋 시

**3.model 경로만** 커밋에 포함. HEREDOC 형식:

```bash
git commit -m "$(cat <<'EOF'
type: 커밋 메시지
EOF
)"
```

---

## Push 전 원격 동기화 (필수)

1. **fetch**: `git fetch origin`
2. **behind 확인**: `git status` 또는 `git branch -vv`
3. **behind이면**: `git pull origin $(git branch --show-current) --no-rebase` → push
4. **behind 아니면**: `git push -u origin $(git branch --show-current)`
5. **규칙**: `git push --force` / `-f` 절대 사용 금지

### Push 거부 시 자동 복구

push가 `rejected (fetch first)` 에러로 실패하면:
```bash
git pull origin $(git branch --show-current) --no-rebase
# merge editor가 열리면 :wq 로 저장
git push -u origin $(git branch --show-current)
```
사용자에게 별도 확인 없이 바로 pull → push 진행 (turbo).

---

## Push 후 GitHub 링크 제공 (필수)

1. `git remote get-url origin`으로 원격 URL 확인
2. `git branch --show-current`로 브랜치명 확인
3. 링크 생성:
   - **브랜치 링크**: `https://github.com/{org}/{repo}/tree/{브랜치명}`
   - **PR 생성 링크**: `https://github.com/{org}/{repo}/compare/{브랜치명}`
4. 사용자에게 브랜치 링크와 PR 생성 링크를 보여준다.
