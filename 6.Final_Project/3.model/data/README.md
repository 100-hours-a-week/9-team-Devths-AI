# Interview Dataset

## 개요
- **출처**: [UICHEOL-HWANG/InterView_Datasets](https://huggingface.co/datasets/UICHEOL-HWANG/InterView_Datasets)
- **Train**: 68,251개
- **Valid**: 8,028개
- **총**: 76,279개의 면접 Q&A

## 데이터 구조

### 칼럼
- `experience`: 경력 (NEW, CAREER)
- `ageRange`: 연령대 (-34, 35~49, 50+)
- `occupation`: 직무 (ICT, 경영/회계/사무, 금융/보험, 기계, 건설, 전기/전자, 등)
- `question`: 면접 질문
- `answer`: 모범 답변

### 예시
```json
{
  "experience": "NEW",
  "ageRange": "-34",
  "occupation": "ICT",
  "question": "본인이 속할 팀에서 어떤 업무를 수행 중인지 파악하고 계십니까",
  "answer": "저는 현재 아이씨티 분야에 지원하였습니다. 이 분야에 대해 알고 있는 것을 말씀드려 보겠습니다..."
}
```

## 사용 방법

### 1. 데이터 로드
```bash
poetry run python scripts/load_interview_dataset.py
```

### 2. VectorDB에 임베딩

**로컬 (embedded Chroma)**
```bash
cd 3.model
poetry run python scripts/embed_interview_dataset.py
```

**데이터 파일 선택** (기본: `data/interview_dataset_valid.json`)
```bash
poetry run python scripts/embed_interview_dataset.py --file data/interview_dataset_valid.json
poetry run python scripts/embed_interview_dataset.py --file data/interview_dataset_train.json
# 또는
INTERVIEW_DATASET_FILE=data/interview_dataset_train.json poetry run python scripts/embed_interview_dataset.py
```

### 3. 운영/스테이징에서 임베딩 실행

배포 환경의 Chroma는 서버 모드(`CHROMA_SERVER_HOST`)로 별도 구동되는 경우가 많아, **로컬에서 임베딩 후 커밋하는 방식은 운영 DB에 반영되지 않습니다.** 아래 중 한 가지 방식으로 **배포 대상 Chroma에 접속 가능한 환경**에서 스크립트를 실행해야 합니다.

**필수 환경 변수**
- `GOOGLE_API_KEY` 또는 `GEMINI_API_KEY`: 임베딩용 API 키
- `CHROMA_SERVER_HOST`: Chroma 서버 호스트 (설정 시 서버 모드)
- `CHROMA_SERVER_PORT`: Chroma 서버 포트 (기본 8000)

**방안 A: 배포 서버에서 수동 실행**

배포 아티팩트에 `data/interview_dataset_valid.json`과 `scripts/embed_interview_dataset.py`가 포함된 상태에서, 배포된 서버에 SSH 접속 후:

```bash
cd /path/to/3.model
export CHROMA_SERVER_HOST=vectordb  # 실제 Chroma 호스트명
export CHROMA_SERVER_PORT=8000
export GEMINI_API_KEY=your_key
poetry run python scripts/embed_interview_dataset.py
```

**방안 B: GitHub Actions에서 수동 워크플로 실행**

Chroma 서버가 GitHub runner에서 TCP 접근 가능한 경우(공인 IP + 방화벽 허용 또는 self-hosted runner가 Chroma와 같은 VPC):

- Secrets에 `CHROMA_SERVER_HOST`, `CHROMA_SERVER_PORT`, `GEMINI_API_KEY` 설정
- `workflow_dispatch`로 임베딩 전용 워크플로 실행: checkout → poetry install → 위 환경 변수 주입 → `poetry run python scripts/embed_interview_dataset.py`

**방안 C: CodeDeploy/배포 후 훅에서 1회 실행**

CodeDeploy lifecycle hook(예: AfterInstall)에서 위 스크립트를 실행하도록 설정. 배포마다 돌리지 않으려면 “최초 1회만” 실행하는 플래그나 별도 수동 job을 권장.

### 4. 서비스에서 활용
```python
from app.services.interview_dataset_service import InterviewDatasetService

dataset = InterviewDatasetService()

# 직무별 질문
questions = dataset.get_questions_by_occupation("ICT", limit=10)

# 경력별 질문
questions = dataset.get_questions_by_experience("NEW", limit=10)

# 랜덤 질문
questions = dataset.get_random_questions(limit=10)

# 통계
stats = dataset.get_statistics()
```

## 통합 계획

### Phase 1: 기본 통합 ✅
- [x] 데이터셋 다운로드
- [x] JSON 파일로 저장
- [x] 서비스 클래스 생성

### Phase 2: VectorDB 통합
- [ ] VectorDB에 임베딩
- [ ] RAG 기반 면접 질문 생성
- [ ] 직무/경력별 필터링

### Phase 3: API 연동
- [ ] 면접 질문 API에 통합
- [ ] 맞춤형 질문 생성 개선

## 주의사항
- 데이터 파일은 `.gitignore`에 추가됨 (용량 큰 파일)
- VectorDB 임베딩 시 시간이 소요될 수 있음 (valid 약 10-15분, train은 더 오래 걸릴 수 있음)
- 운영 Chroma가 서버 모드면 반드시 `CHROMA_SERVER_HOST`를 설정한 뒤, 해당 서버에 접근 가능한 환경에서 임베딩 스크립트를 실행해야 함
