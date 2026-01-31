# Git 워크플로우 가이드

## 브랜치 전략

```
main (프로덕션)
  ↑
develop (개발 통합)
  ↑
feature/기능명 (개별 작업)
```

## 기본 워크플로우

### 1. 새 기능 작업 시작

```bash
# develop 브랜치로 이동
git checkout develop

# 최신 상태로 업데이트
git pull origin develop

# 새 feature 브랜치 생성
git checkout -b feature/기능명
```

**브랜치 네이밍 규칙:**
- `feature/기능명` - 새로운 기능 개발
- `fix/버그명` - 버그 수정
- `chore/작업명` - 설정, 빌드 등 기타 작업
- `docs/문서명` - 문서 작업

**예시:**
```bash
git checkout -b feature/fastapi-setup
git checkout -b fix/login-error
git checkout -b chore/ci-pipeline
```

### 2. 작업 및 커밋

```bash
# 코드 작성 후...

# 변경사항 확인
git status

# 파일 스테이징
git add .
# 또는 특정 파일만
git add app/main.py

# 커밋
git commit -m "feat: 기능 설명(#이슈번호)"
```

**커밋 메시지 규칙:**
- `feat:` - 새로운 기능 추가
- `fix:` - 버그 수정
- `docs:` - 문서 수정
- `refactor:` - 코드 리팩토링
- `test:` - 테스트 코드, 리팩토링 테스트 코드 추가
- `chore:` - 설정 추가
- `style:` - 코드 포맷팅, CSS, 세미콜론 누락, 코드 변경이 없는 경우

**커밋 메시지 형식:**
```
타입: 설명(#이슈번호)
```

**예시:**
```bash
git commit -m "feat: 회원 가입 기능 구현(#12)"
git commit -m "fix: 로그인 시 토큰 만료 오류 수정(#23)"
git commit -m "docs: API 문서 업데이트(#34)"
git commit -m "refactor: 사용자 인증 로직 개선(#45)"
git commit -m "test: 회원가입 API 테스트 코드 추가(#56)"
git commit -m "chore: CI/CD 파이프라인 설정(#67)"
git commit -m "style: 코드 포맷팅 및 린트 규칙 적용(#78)"
```

**⚠️ 주의사항:**
- 커밋 메시지는 한글로 작성
- 이슈 번호는 반드시 포함 (예: `#12`)
- 설명은 명확하고 간결하게

### 3. 원격에 feature 브랜치 푸시

```bash
# 처음 푸시할 때
git push -u origin feature/기능명

# 이후 푸시
git push
```

### 4. GitHub에서 PR 생성

1. GitHub 저장소 페이지로 이동
2. **Pull requests** 탭 클릭
3. **New pull request** 클릭
4. **base:** `develop` ← **compare:** `feature/기능명` 선택
5. PR 제목과 설명 작성
6. **Create pull request** 클릭

**⚠️ 주의사항:**
- PR은 반드시 `develop` 브랜치로 보내야 합니다 (main ❌)
- PR 제목은 명확하게 작성
- 변경사항에 대한 설명 추가

### 5. 리뷰 후 merge되면 로컬 정리

```bash
# develop 브랜치로 이동
git checkout develop

# 최신 상태로 업데이트 (merge된 내용 받기)
git pull origin develop

# 작업 완료된 feature 브랜치 삭제
git branch -d feature/기능명

# 원격 브랜치도 삭제 (선택사항)
git push origin --delete feature/기능명
```

## 자주 사용하는 명령어

### 브랜치 관리

```bash
# 현재 브랜치 확인
git branch

# 모든 브랜치 확인 (원격 포함)
git branch -a

# 브랜치 전환
git checkout 브랜치명

# 브랜치 생성 및 전환
git checkout -b 새브랜치명

# 브랜치 삭제
git branch -d 브랜치명
```

### 변경사항 확인

```bash
# 현재 상태 확인
git status

# 변경사항 상세 확인
git diff

# 커밋 히스토리 확인
git log --oneline -10

# 그래프로 보기
git log --oneline --graph --all -10
```

### 동기화

```bash
# 원격 저장소에서 최신 내용 가져오기
git pull origin develop

# 원격 저장소 정보 업데이트
git fetch origin

# 원격 저장소 확인
git remote -v
```

## 문제 해결

### develop이 업데이트되었을 때 (feature 브랜치 작업 중)

```bash
# 현재 작업 저장
git add .
git commit -m "feat: 작업 중인 내용"

# develop 최신화
git checkout develop
git pull origin develop

# feature 브랜치로 돌아가서 rebase
git checkout feature/기능명
git rebase develop

# 충돌 발생 시 해결 후
git add .
git rebase --continue

# 원격에 강제 푸시 (rebase 후 필요)
git push -f origin feature/기능명
```

### 실수로 develop에 직접 커밋한 경우

```bash
# 1. 현재 커밋을 feature 브랜치로 옮기기
git checkout -b feature/기능명

# 2. develop을 원격 상태로 되돌리기
git checkout develop
git reset --hard origin/develop

# 3. feature 브랜치에서 작업 계속
git checkout feature/기능명
git push origin feature/기능명
```

### 커밋 메시지 수정 (아직 푸시 안 한 경우)

```bash
# 마지막 커밋 메시지 수정
git commit --amend -m "새로운 커밋 메시지"
```

### 변경사항 임시 저장

```bash
# 현재 작업 임시 저장
git stash

# 임시 저장 목록 확인
git stash list

# 임시 저장 내용 복원
git stash pop
```

## 팀 협업 규칙

1. **절대 main에 직접 푸시하지 않기**
2. **develop에도 직접 푸시하지 않기** (PR 사용)
3. **feature 브랜치는 작은 단위로 자주 만들기**
4. **커밋은 의미 있는 단위로 나누기**
5. **PR은 리뷰 후 merge**
6. **merge 후 feature 브랜치는 삭제**
7. **작업 시작 전 항상 `git pull origin develop`**

## 참고 자료

- [Git 공식 문서](https://git-scm.com/doc)
- [GitHub Flow](https://docs.github.com/en/get-started/quickstart/github-flow)
- [Conventional Commits](https://www.conventionalcommits.org/)
