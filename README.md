# 9-team-Devths-AI

KTB Final Project의 FastAPI 기반 AI 모델 서비스입니다.

## 주요 기능

- **PII Masking**: 개인정보 자동 마스킹 (Gemini Vision API, OpenCV 기반)
- **RAG Service**: 벡터 DB 기반 문서 검색 및 질의응답
- **LLM Service**: vLLM 및 Gemini API를 활용한 대화 생성
- **OCR & Text Extraction**: 문서에서 텍스트 추출

## 프로젝트 구조

```
3.model/
├── app/
│   ├── api/
│   │   └── routes/          # API 엔드포인트
│   │       ├── ai.py        # AI 관련 라우트 (챗봇, RAG, OCR)
│   │       └── masking.py   # 마스킹 라우트
│   ├── core/                # 핵심 모듈
│   │   └── task_storage.py  # 비동기 태스크 관리
│   ├── schemas/             # Pydantic 스키마
│   ├── services/            # 비즈니스 로직
│   │   ├── vllm_service.py  # vLLM 서비스
│   │   ├── llm_service.py   # LLM 통합 서비스
│   │   ├── rag_service.py   # RAG 서비스
│   │   ├── pii_masking.py   # PII 마스킹 (OpenCV)
│   │   ├── gemini_masking.py # Gemini 기반 마스킹
│   │   └── vectordb_service.py # 벡터 DB 관리
│   ├── utils/               # 유틸리티
│   └── main.py              # FastAPI 앱 진입점
├── scripts/                 # 배포 스크립트
│   ├── deploy_gemini.sh    # Gemini 서버 배포
│   ├── deploy_vllm.sh      # vLLM 서버 배포
│   └── health_check.sh     # 헬스 체크
├── requirements.txt         # Python 의존성
├── vllm_requirements.txt    # vLLM 전용 의존성
├── start_server.sh          # 로컬 서버 시작
├── .env.example             # 환경변수 예제
├── .gitignore
└── README.md
```

## 설치 및 실행

### 1. 환경 설정

```bash
# .env 파일 생성
cp .env.example .env

# 환경변수 설정
# GEMINI_API_KEY=your_gemini_api_key
# VLLM_API_BASE=http://your-vllm-server:8000/v1
# VLLM_MODEL_NAME=your-model-name
```

### 2. 의존성 설치

```bash
# 기본 의존성
pip install -r requirements.txt

# vLLM 서버용 (GPU 필요)
pip install -r vllm_requirements.txt
```

### 3. 서버 실행

```bash
# 로컬 실행
bash start_server.sh

# 또는
uvicorn app.main:app --host 0.0.0.0 --port 8080 --reload
```

## GCP 배포

### Quick Start

자세한 내용은 다음 가이드를 참고하세요:

- [GCP_QUICKSTART.md](GCP_QUICKSTART.md) - 빠른 시작 가이드
- [GCP_SIMPLE_GUIDE.md](GCP_SIMPLE_GUIDE.md) - 간단한 배포 가이드
- [GCP_MULTI_SERVER_GUIDE.md](GCP_MULTI_SERVER_GUIDE.md) - 멀티 서버 구성

### 자동 배포 스크립트

```bash
# Gemini API 서버 배포
cd scripts
bash deploy_gemini.sh

# vLLM 서버 배포 (GPU 필요)
bash deploy_vllm.sh

# 헬스 체크
bash health_check.sh
```

## API 엔드포인트

### AI 관련

- `POST /api/ai/chat` - 챗봇 대화
- `POST /api/ai/chat/stream` - 스트리밍 대화
- `POST /api/ai/rag/ingest` - 문서 임베딩
- `POST /api/ai/rag/query` - RAG 질의
- `POST /api/ai/ocr` - OCR 및 텍스트 추출

### 마스킹

- `POST /api/masking/image` - 이미지 PII 마스킹
- `POST /api/masking/gemini` - Gemini 기반 마스킹
- `GET /api/masking/task/{task_id}` - 태스크 상태 확인

### 헬스 체크

- `GET /health` - 서버 상태 확인

## 기술 스택

- **FastAPI**: 웹 프레임워크
- **vLLM**: LLM 추론 최적화
- **Gemini API**: Google AI 모델
- **ChromaDB**: 벡터 데이터베이스
- **OpenCV**: 컴퓨터 비전
- **Pydantic**: 데이터 검증

## 환경 변수

```bash
# Gemini API
GEMINI_API_KEY=your_api_key
GEMINI_MODEL=gemini-1.5-flash

# vLLM 설정
VLLM_API_BASE=http://localhost:8000/v1
VLLM_MODEL_NAME=model-name
VLLM_MAX_TOKENS=2048

# 서버 설정
HOST=0.0.0.0
PORT=8080
```

## 라이선스

이 프로젝트는 KTB Final Project의 일부입니다.
