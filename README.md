# 9-team-Devths-AI

KTB Final Project의 FastAPI 기반 AI 모델 서비스입니다.

## 주요 기능

- **PII Masking**: 개인정보 자동 마스킹 (Gemini Vision API, Chandra 모델)
- **RAG Service**: 벡터 DB 기반 문서 검색 및 질의응답
- **LLM Service**: vLLM 및 Gemini API를 활용한 대화 생성
- **Interview Service**: AI 기반 기술 면접 시뮬레이션
- **OCR & Text Extraction**: 문서에서 텍스트 추출

## 프로젝트 구조

```
3.model/
├── app/
│   ├── api/                        # API 레이어
│   │   ├── middleware/             # API 미들웨어
│   │   └── routes/                 # API 엔드포인트
│   │       ├── v1/                 # v1 API (레거시)
│   │       │   ├── ai.py           # AI 라우트 (챗봇, RAG, OCR)
│   │       │   └── masking.py      # 마스킹 라우트
│   │       └── v2/                 # v2 API (모듈화)
│   │           ├── chat.py         # 채팅 API
│   │           ├── text_extract.py # 텍스트 추출 API
│   │           ├── masking.py      # PII 마스킹 API
│   │           ├── task.py         # 비동기 작업 상태 API
│   │           ├── calendar.py     # 캘린더 파싱 API
│   │           ├── _helpers.py     # 공통 헬퍼 함수
│   │           └── _sse_errors.py  # SSE 에러 이벤트 헬퍼
│   │
│   ├── config/                     # 설정 관리
│   │   ├── settings.py             # 환경변수 및 설정
│   │   └── dependencies.py         # DI 의존성 정의
│   │
│   ├── domain/                     # 도메인 레이어 (비즈니스 로직)
│   │   ├── chat/                   # 채팅 도메인
│   │   │   └── chains.py           # LangChain 체인
│   │   ├── interview/              # 면접 도메인
│   │   │   ├── entities.py         # 면접 엔티티
│   │   │   └── graph.py            # LangGraph 워크플로우
│   │   ├── masking/                # 마스킹 도메인
│   │   └── ocr/                    # OCR 도메인
│   │
│   ├── infrastructure/             # 인프라 레이어 (외부 서비스 어댑터)
│   │   ├── llm/                    # LLM 어댑터
│   │   │   ├── base.py             # 기본 인터페이스
│   │   │   ├── gemini.py           # Gemini 어댑터
│   │   │   ├── vllm.py             # vLLM 어댑터
│   │   │   └── langchain_wrapper.py # LangChain 래퍼
│   │   ├── vectordb/               # 벡터 DB 어댑터
│   │   │   ├── base.py             # 기본 인터페이스
│   │   │   └── chroma.py           # ChromaDB 어댑터
│   │   ├── session/                # 세션 관리
│   │   │   ├── base.py             # 기본 인터페이스
│   │   │   ├── memory.py           # 인메모리 세션
│   │   │   └── redis.py            # Redis 세션
│   │   ├── queue/                  # 작업 큐
│   │   │   ├── base.py             # 기본 인터페이스
│   │   │   ├── file_queue.py       # 파일 기반 큐
│   │   │   └── celery_queue.py     # Celery 큐
│   │   └── ocr/                    # OCR 어댑터
│   │       └── base.py             # OCR 인터페이스
│   │
│   ├── services/                   # 서비스 레이어 (레거시, 점진적 마이그레이션)
│   │   ├── llm_service.py          # LLM 통합 서비스
│   │   ├── rag_service.py          # RAG 서비스
│   │   ├── gemini_masking.py       # Gemini 마스킹
│   │   ├── chandra_masking.py      # Chandra 마스킹
│   │   └── cloudwatch_service.py   # CloudWatch 메트릭
│   │
│   ├── schemas/                    # Pydantic 스키마
│   ├── prompts/                    # 프롬프트 관리
│   │   └── templates/              # 프롬프트 템플릿
│   │       ├── chat/               # 채팅 프롬프트
│   │       └── interview/          # 면접 프롬프트
│   ├── core/                       # 핵심 모듈
│   │   └── task_storage.py         # 비동기 태스크 관리
│   ├── utils/                      # 유틸리티
│   │   └── log_sanitizer.py        # 로그 보안 처리
│   ├── shared/                     # 공유 모듈
│   ├── eval/                       # 평가 모듈
│   ├── middlewares/                # 미들웨어
│   ├── parsers/                    # 파서
│   └── main.py                     # FastAPI 앱 진입점
│
├── data/                           # 데이터 디렉토리
├── deploy/                         # 배포 스크립트
│   ├── after_install.sh
│   ├── before_install.sh
│   ├── start_server_deploy.sh
│   ├── stop_server.sh
│   └── validate_service.sh
├── scripts/                        # 유틸리티 스크립트
├── appspec.yml                     # AWS CodeDeploy 설정
├── pyproject.toml                  # Poetry 의존성
├── poetry.lock                     # Poetry 락 파일
└── README.md
```

## 아키텍처

프로젝트는 **Clean Architecture** 패턴을 따릅니다:

1. **API Layer** (`app/api/`): HTTP 요청/응답 처리
2. **Domain Layer** (`app/domain/`): 비즈니스 로직
3. **Infrastructure Layer** (`app/infrastructure/`): 외부 서비스 어댑터
4. **Config Layer** (`app/config/`): DI 및 설정 관리

## 설치 및 실행

### 1. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# 환경변수 설정
# GOOGLE_API_KEY=your_gemini_api_key
# GCP_VLLM_BASE_URL=http://your-vllm-server:8000/v1
```

### 2. 의존성 설치

```bash
# Poetry 사용 (권장)
poetry install

# 또는 pip
pip install -r requirements.txt
```

### 3. 서버 실행

```bash
# Poetry 사용
poetry run uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload

# 또는 직접 실행
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## API 엔드포인트

> 📄 **상세 API 명세(요청/응답 Body, 에러 코드)**: [API_SPEC.md](./API_SPEC.md)

### v2 API (기본, `/ai/...`)

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/ai/text/extract` | 텍스트 추출 + 임베딩 (비동기) |
| `GET` | `/ai/task/{task_id}` | 비동기 작업 상태 조회 |
| `POST` | `/ai/chat` | 채팅 스트리밍 (SSE, 일반/면접/리포트) |
| `POST` | `/ai/calendar/parse` | 캘린더 일정 파싱 |
| `POST` | `/ai/masking/draft` | PII 마스킹 (비동기) |
| `GET` | `/ai/masking/task/{task_id}` | 마스킹 작업 상태 조회 |
| `GET` | `/ai/masking/health` | 마스킹 서비스 헬스 체크 |

### v1 API (레거시, `/ai/v1/...`)

> v2와 동일한 기능이며 하위 호환을 위해 유지됩니다.

| Method | Path | 설명 |
|--------|------|------|
| `POST` | `/ai/v1/chat` | 챗봇 대화 (SSE 스트리밍) |
| `POST` | `/ai/v1/text-extract` | 텍스트 추출 |
| `GET` | `/ai/v1/task/{task_id}` | 작업 상태 조회 |
| `POST` | `/ai/v1/calendar/parse` | 캘린더 파싱 |
| `POST` | `/ai/v1/masking/draft` | PII 마스킹 |
| `GET` | `/ai/v1/masking/task/{task_id}` | 마스킹 작업 상태 조회 |
| `GET` | `/ai/v1/masking/health` | 마스킹 헬스 체크 |

### 헬스 체크

- `GET /health` - 서버 상태 확인

## SSE 이벤트 규약

`/ai/chat` 엔드포인트는 SSE(Server-Sent Events) 스트리밍으로 응답합니다.

### 이벤트 타입

| type | 설명 | 페이로드 예시 |
|------|------|--------------|
| `chunk` | 텍스트 청크 | `{"chunk": "안녕하세요"}` |
| `summary` | 채팅방 제목 (첫 응답 시) | `{"summary": "프로젝트 질문"}` |
| `session_state` | 면접 세션 상태 | `{"session_state": {...}}` |
| `error` | 에러 발생 | 아래 에러 포맷 참조 |
| `[DONE]` | 스트림 종료 | `data: [DONE]` |

### SSE 에러 포맷

모든 SSE 에러는 통일된 JSON 포맷으로 전송됩니다:

```json
{
  "type": "error",
  "error": {
    "code": "INTERNAL_ERROR",
    "status": 500,
    "message": "상세 에러 메시지"
  },
  "fallback": "사용자에게 표시할 메시지"
}
```

### 에러 코드

| code | status | 상황 |
|------|--------|------|
| `PROMPT_BLOCKED` | 400 | 프롬프트 인젝션 차단 |
| `VECTORDB_ERROR` | 404 | 문서 미업로드 (벡터 DB 비어있음) |
| `SESSION_NOT_FOUND` | 404 | 면접 세션 없음 |
| `PARSE_FAILED` | 500 | LLM 응답 JSON 파싱 실패 |
| `INTERNAL_ERROR` | 500 | 서버 내부 오류 |
| `LLM_ERROR` | 500 | LLM 서비스 호출 실패 |

> **Spring 백엔드 처리 가이드**: SSE 이벤트에서 `"type": "error"` 필드를 감지하면 에러로 처리하고, `fallback` 값을 사용자에게 표시하세요.

## 기술 스택

- **FastAPI**: 웹 프레임워크
- **Poetry**: 의존성 관리
- **vLLM**: LLM 추론 최적화
- **Gemini API**: Google AI 모델
- **ChromaDB**: 벡터 데이터베이스
- **LangChain/LangGraph**: LLM 오케스트레이션
- **Redis**: 세션 관리 (선택)
- **Celery**: 작업 큐 (선택)
- **Pydantic**: 데이터 검증
- **Ruff**: 린팅 및 포맷팅

## 환경 변수

```bash
# Gemini API
GOOGLE_API_KEY=your_api_key

# vLLM 설정
GCP_VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=model-name

# Redis (선택)
REDIS_URL=redis://localhost:6379

# 서버 설정
HOST=0.0.0.0
PORT=8080
```

## 배포

AWS CodeDeploy를 사용하여 자동 배포됩니다.

```bash
# GitHub Actions를 통해 자동 배포
# develop 브랜치 푸시 시 자동 실행
```

## 라이선스

이 프로젝트는 KTB Final Project의 일부입니다.
