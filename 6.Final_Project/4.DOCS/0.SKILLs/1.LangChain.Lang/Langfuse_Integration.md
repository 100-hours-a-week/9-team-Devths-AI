## Langfuse 개요

- **Langfuse**: LLM 호출(프롬프트/응답/메타데이터/에러)을 수집해 대시보드로 보는 Observability 도구
- 이 프로젝트에서는 **self-hosted Langfuse**(Docker Compose) + **Python SDK**로 LLM 호출을 추적합니다.

## 1) Self-hosted Langfuse 실행

`3.model/docker-compose.yml` 기준:

- **Langfuse Web UI**: `http://localhost:3001`
- **Postgres(컨테이너)**: 내부 5432, 호스트 5433
- **ClickHouse**: 8123/9000 (localhost)
- **MinIO**: 9090/9091
- **Redis**: 6379

실행:

```bash
cd 3.model
docker-compose up -d
docker-compose ps
```

## 2) Langfuse 계정/프로젝트/API Key

1. `http://localhost:3001` 접속
2. 초기 사용자 생성
3. 프로젝트 생성
4. Settings → API Keys에서 키 발급

발급된 키를 `.env`에 설정:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxx
LANGFUSE_HOST=http://localhost:3001
```

`.env` 수정 후 컨테이너 재시작:

```bash
docker-compose restart langfuse-web langfuse-worker
```

## 3) 코드 통합 구조(권장)

### (A) Langfuse 클라이언트 유틸

- 파일: `3.model/app/utils/langfuse_client.py`
- 책임:
  - 환경변수로 Langfuse 클라이언트 생성
  - `trace_llm_call(...)`로 Trace 생성
  - `create_generation(...)`로 LLM 호출 기록

### (B) LLM 호출 지점에서 기록

- 파일: `3.model/app/services/llm_service.py`
- 적용 위치: `LLMService.generate_response()`

동작:
- 요청 시작 시 `trace_llm_call(...)` 생성
- 스트리밍으로 내려보낸 chunk를 `full_response`로 누적
- 스트리밍 종료 후 `create_generation(...)`으로 input/output 기록
- 예외 발생 시 trace metadata에 error 기록(가능한 경우)

### (C) user_id 전달

- RAG 흐름에서는 `RAGService.chat_with_rag()`가 `user_id`를 알고 있으므로
  - `self.llm.generate_response(..., user_id=user_id)` 형태로 전달합니다.

## 4) 서버 실행(Poetry)

Poetry 기반 실행:

```bash
cd 3.model
PYTHONPATH=. poetry run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

또는 스크립트:

```bash
cd 3.model
./start_server.sh
```

## 5) 확인 방법

- Langfuse UI(`http://localhost:3001`)에서 Traces/Generations가 생성되는지 확인
- 환경변수 미설정 시 Langfuse는 비활성(로그 warning)이며 서비스는 정상 동작

## 6) 운영 팁

- `LANGFUSE_HOST`를 클라우드 Langfuse로 바꾸면 **그 클라우드 인스턴스**에 기록됩니다.
  (self-hosted와 cloud는 데이터가 공유되지 않음)
- 프로덕션에서는 compose의 `CHANGEME` 비밀번호/키들을 반드시 변경

## 7) OCR(이미지/PDF) 페이지 단위 추적

현재 구현은 OCR에 대해 다음 두 레벨로 Langfuse에 기록합니다.

- **파일 단위 trace**: `gemini_extract_text`
- **PDF 페이지 단위 generation**: `gemini_ocr_page_{n}` (페이지별 텍스트 4,000자까지 저장)
- **PDF 파일 요약 generation**: `gemini_ocr_pdf_summary` (전체 텍스트 4,000자까지 저장)
- **내부 이미지 OCR 호출**: `_extract_text_from_image()`에서도 별도 trace/generation을 남김

주의:
- PDF 페이지 수가 많으면 generation 이벤트가 많이 쌓일 수 있습니다.
- 운영 환경에서는 필요 시 페이지 단위 기록을 토글하는 옵션을 추가하는 것을 권장합니다.
