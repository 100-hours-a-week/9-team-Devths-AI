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
```bash
poetry run python scripts/embed_interview_dataset.py
```

### 3. 서비스에서 활용
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
- VectorDB 임베딩 시 시간이 소요될 수 있음 (약 10-15분)
