# 3.model/Dockerfile (AI Endpoint Server, v2)
# Base: Python 3.10 Slim (CPU)
#
# ChromaDB: docker-compose 사용 시 CHROMA_SERVER_HOST=vectordb 로 서버 모드.
# 이미지 단독 실행(embedded 모드) 시 CHROMA_PERSIST_DIR 기본값으로 /data/chroma_db 사용.
# 영속화하려면: -v chroma_volume:/data/chroma_db 로 볼륨 마운트.

FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.8.2 \
    POETRY_HOME="/opt/poetry" \
    POETRY_VIRTUALENVS_IN_PROJECT=true \
    POETRY_NO_INTERACTION=1 \
    CHROMA_PERSIST_DIR=/data/chroma_db

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl poppler-utils build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

WORKDIR /app

COPY pyproject.toml poetry.lock ./
RUN poetry install --without ml,dev --no-root

COPY app ./app
COPY scripts ./scripts

# Embedded 모드 시 ChromaDB 데이터 디렉터리 (볼륨 마운트 권장)
RUN mkdir -p /data/chroma_db
VOLUME ["/data/chroma_db"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ADR-102: Gunicorn 4 Uvicorn workers (Docker Compose 환경 동시성 확장)
# --timeout 300: LLM 호출 등 장시간 처리에서 worker 강제 종료 방지
# uvicorn.workers.UvicornWorker 필수 (ASGI 지원)
CMD ["poetry", "run", "gunicorn", "app.main:app", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000", "--timeout", "300"]
