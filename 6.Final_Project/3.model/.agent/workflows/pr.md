---
description: GitHub Pull Request를 생성합니다. 브랜치 diff를 분석하여 제목, 요약, 테스트 계획을 포함합니다.
---

# Pull Request 생성기

## 인수

- 사용자가 base branch를 지정하면 해당 브랜치 사용 (기본값: `develop`)
- hotfix 브랜치는 `main` 대상

## 실행 순서

// turbo
1. **base branch 결정**
   - 사용자 인수가 있으면 해당 브랜치 사용
   - 없으면 `develop` 기본값
   - `hotfix/*` 브랜치는 `main` 대상

// turbo
2. **컨텍스트 수집** (병렬 실행)
   ```bash
   git branch --show-current
   git log $(git merge-base develop HEAD)..HEAD --oneline
   git diff develop...HEAD --stat
   git diff develop...HEAD
   ```

3. **PR 제목 생성** (70자 이내, 한국어)
   - `feature/*` → `feat: ...`
   - `fix/*` → `fix: ...`
   - `chore/*` → `chore: ...`
   - `docs/*` → `docs: ...`
   - `hotfix/*` → `hotfix: ...`
   - `release/*` → `release: ...`

4. **PR 본문 생성**

```markdown
## 작업한 내용
- [변경 사항 요약 bullet list, 한국어]

## 참고 사항
- [리뷰어가 알아야 할 내용]

## 테스트 계획
- [ ] Ruff lint 통과 (`poetry run ruff check app/`)
- [ ] Ruff format 통과 (`poetry run ruff format --check app/`)
- [ ] Python import 검증 통과
- [ ] 기능 동작 테스트 완료
```

5. **Push & PR 생성**
   ```bash
   git push -u origin $(git branch --show-current)
   gh pr create --title "..." --body "$(cat <<'EOF' ... EOF)"
   ```
   - push가 `rejected (fetch first)`로 실패하면:
     ```bash
     git pull origin $(git branch --show-current) --no-rebase
     git push -u origin $(git branch --show-current)
     ```

## 브랜치 규칙

- `develop`, `main`, `release/*`에 직접 커밋/push 금지
- PR은 반드시 feature/fix/chore/docs/hotfix 브랜치에서 생성
- 현재 보호 브랜치에 있으면 PR 생성을 거부하고 사용자에게 안내

## 규칙

- 절대 force push 금지
- push 시 항상 `-u` 플래그 사용
- PR 상세 내용을 사용자에게 보여주고 확인 후 생성
- 완료 후 PR URL 반환
- **PR 본문에 "Made with Cursor", "Generated with Claude", "Made with Antigravity" 등 도구/브랜드 문구를 넣지 않는다.**

## 배치 호출 시 (/commit + /pr 동시 호출)

- `/commit`과 함께 호출된 경우, 커밋이 완료된 직후 바로 PR 생성 진행
- 사용자 확인 단계를 **최종 결과 보고 1회**로 통합 (중간 중간 묻지 않음)
- 4개 워크플로(/code-review, /commit, /pr, /pr-comments)가 동시에 호출되면 순서: code-review → commit → push & PR → PR comments
