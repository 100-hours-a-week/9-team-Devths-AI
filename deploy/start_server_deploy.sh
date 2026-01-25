#!/bin/bash

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting FastAPI server (Production Mode with Poetry)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

APP_DIR="/home/ubuntu/ai"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/fastapi-app.log"

cd "$APP_DIR"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 환경변수 로드 (AWS Parameter Store)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "📥 Loading environment variables..."

# .deploy-env 파일에서 브랜치 정보 읽기
if [ -f "$APP_DIR/.deploy-env" ]; then
    source "$APP_DIR/.deploy-env"
    echo "📋 Deploy info: branch=$DEPLOY_BRANCH, timestamp=$DEPLOY_TIMESTAMP"

    # 브랜치에 따라 Parameter Store 경로 설정
    case "$DEPLOY_BRANCH" in
        develop)
            export PARAMETER_STORE_PATH="/Dev/AI/"
            ;;
        release)
            export PARAMETER_STORE_PATH="/Stg/AI/"
            ;;
        main)
            export PARAMETER_STORE_PATH="/Prod/AI/"
            ;;
        *)
            echo "⚠️  Unknown branch: $DEPLOY_BRANCH, using default /Prod/AI/"
            export PARAMETER_STORE_PATH="/Prod/AI/"
            ;;
    esac
else
    echo "⚠️  .deploy-env file not found, using default /Prod/AI/"
    export PARAMETER_STORE_PATH="${PARAMETER_STORE_PATH:-/Prod/AI/}"
fi

echo "📍 Parameter Store Path: $PARAMETER_STORE_PATH"

# 환경변수 로드 스크립트 실행
if [ -f "$APP_DIR/deploy/load_env_from_parameter_store.sh" ]; then
    source "$APP_DIR/deploy/load_env_from_parameter_store.sh"
    if [ $? -ne 0 ]; then
        echo "❌ Failed to load environment variables"
        echo "💡 Falling back to local .env file if exists..."
        if [ -f ".env" ]; then
            export $(cat .env | grep -v '^#' | xargs)
            echo "✅ Environment variables loaded from .env"
        else
            echo "⚠️  No .env file found, continuing with system environment..."
        fi
    fi
else
    echo "⚠️  load_env_from_parameter_store.sh not found"
    if [ -f ".env" ]; then
        export $(cat .env | grep -v '^#' | xargs)
        echo "✅ Environment variables loaded from .env"
    fi
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. Poetry 경로 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

export PATH="/home/ubuntu/.local/bin:$PATH"

# Poetry 설치 확인
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found! Please check after_install.sh"
    exit 1
fi

echo "✅ Poetry version: $(poetry --version)"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 로그 파일 초기화
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"
chown ubuntu:ubuntu "$LOG_FILE"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. 서버 시작 (Poetry를 통한 uvicorn 실행)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "🌐 Starting server on ${HOST:-0.0.0.0}:${PORT:-8000}"
echo "📝 Logs: $LOG_FILE"

# Poetry를 통해 uvicorn 실행
nohup poetry run uvicorn app.main:app \
    --host "${HOST:-0.0.0.0}" \
    --port "${PORT:-8000}" \
    --workers "${WORKERS:-2}" \
    >> "$LOG_FILE" 2>&1 &

# PID 저장
echo $! > /tmp/fastapi-app.pid

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 서버 시작 확인 및 헬스 체크
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "⏳ Waiting for server to start..."
sleep 5

# 프로세스 확인
if pgrep -f "uvicorn app.main:app" > /dev/null; then
    echo "✅ Server started successfully (PID: $(cat /tmp/fastapi-app.pid))"

    # 헬스 체크 (최대 10초 대기)
    for i in {1..10}; do
        if curl -s http://localhost:${PORT:-8000}/health > /dev/null 2>&1; then
            echo "✅ Health check passed"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "🎉 FastAPI server is running on http://${HOST:-0.0.0.0}:${PORT:-8000}"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            exit 0
        fi
        sleep 1
    done

    echo "⚠️  Server started but health check failed (may need more time)"
    echo "📋 Last 20 lines of log:"
    tail -n 20 "$LOG_FILE"
    exit 0
else
    echo "❌ Server failed to start"
    echo "📋 Last 20 lines of log:"
    tail -n 20 "$LOG_FILE"
    exit 1
fi
