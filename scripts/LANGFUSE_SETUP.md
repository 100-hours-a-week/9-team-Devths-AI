# Langfuse 설정 가이드

Langfuse는 LLM 호출을 추적하고 모니터링하는 도구입니다.

## 1. Docker Compose로 Langfuse 시작

```bash
# Langfuse 서비스 시작
docker-compose up -d

# 서비스 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs -f langfuse-web
```

## 2. 환경 변수 설정

`.env` 파일을 생성하고 다음 변수들을 설정하세요:

```bash
# Langfuse API 키 (웹 UI에서 생성)
LANGFUSE_PUBLIC_KEY=pk-lf-xxxxx
LANGFUSE_SECRET_KEY=sk-lf-xxxxx
LANGFUSE_HOST=http://localhost:3001

# Google API (기존)
GOOGLE_API_KEY=your-google-api-key-here
```

### Langfuse API 키 생성 방법

1. Docker Compose로 Langfuse 시작 후 `http://localhost:3001` 접속
2. 초기 사용자 계정 생성
3. 프로젝트 생성
4. Settings > API Keys에서 Public Key와 Secret Key 생성

## 3. Langfuse 사용 예시

### 기본 사용법

```python
from app.utils.langfuse_client import get_langfuse_client, trace_llm_call, create_generation

# Trace 생성
trace = trace_llm_call(
    name="chat_completion",
    user_id="user123",
    metadata={"endpoint": "/ai/chat"}
)

# Generation 기록
if trace:
    generation = create_generation(
        trace=trace,
        name="gemini_chat",
        model="gemini-3-flash-preview",
        input_text="사용자 질문",
        output_text="AI 응답",
        metadata={"temperature": 0.7}
    )
```

### 데코레이터 사용

```python
from app.utils.langfuse_client import observe_llm_call

@observe_llm_call(name="chat_completion")
async def chat(user_message: str):
    # LLM 호출 코드
    response = await llm_service.generate_response(user_message)
    return response
```

## 4. LLM Service에 통합 예시

`app/services/llm_service.py`에 Langfuse 통합:

```python
from app.utils.langfuse_client import get_langfuse_client, trace_llm_call, create_generation

class LLMService:
    async def generate_response(self, user_message: str, ...):
        # Trace 생성
        trace = trace_llm_call(
            name="gemini_chat",
            metadata={"model": self.model_name}
        )
        
        try:
            # LLM 호출
            response = self.client.models.generate_content_stream(...)
            
            # 응답 수집
            full_response = ""
            for chunk in response:
                if hasattr(chunk, 'text') and chunk.text:
                    full_response += chunk.text
                    yield chunk.text
            
            # Generation 기록
            if trace:
                create_generation(
                    trace=trace,
                    name="gemini_streaming",
                    model=self.model_name,
                    input_text=user_message,
                    output_text=full_response,
                    metadata={"streaming": True}
                )
        except Exception as e:
            if trace:
                trace.update(metadata={"error": str(e)})
            raise
```

## 5. 서비스 포트

- **Langfuse Web UI**: http://localhost:3001 (포트 3000은 프론트엔드에서 사용 중)
- **Weaviate**: http://localhost:8080
- **PostgreSQL**: localhost:5432
- **MinIO Console**: http://localhost:9091
- **Redis**: localhost:6379

## 6. 보안 주의사항

⚠️ **프로덕션 환경에서는 반드시 다음을 변경하세요:**

1. `.env` 파일의 모든 `CHANGEME` 표시된 비밀번호 변경
2. `ENCRYPTION_KEY` 생성: `openssl rand -hex 32`
3. `NEXTAUTH_SECRET` 생성: `openssl rand -hex 32`
4. PostgreSQL, Redis, ClickHouse 비밀번호 변경
5. MinIO 접근 키 변경

## 7. 문제 해결

### Langfuse가 시작되지 않는 경우

```bash
# 로그 확인
docker-compose logs langfuse-web

# 컨테이너 재시작
docker-compose restart langfuse-web
```

### API 키가 작동하지 않는 경우

1. Langfuse Web UI에서 API 키가 올바르게 생성되었는지 확인
2. `.env` 파일의 키 값이 정확한지 확인
3. `LANGFUSE_HOST`가 올바른지 확인 (기본값: http://localhost:3000)
