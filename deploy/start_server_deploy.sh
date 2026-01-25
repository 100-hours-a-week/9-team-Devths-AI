#!/bin/bash

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Starting FastAPI server (Production Mode with Poetry)..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

APP_DIR="/home/ubuntu/ai"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/fastapi-app.log"

cd "$APP_DIR" || exit 1

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 환경변수 로드 (AWS Parameter Store)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "📥 Loading environment variables..."

# .deploy-env 파일에서 배포 정보 읽기
if [ -f "$APP_DIR/.deploy-env" ]; then
    echo "📄 Found .deploy-env file, loading deployment info..."
    source "$APP_DIR/.deploy-env"
    echo "📋 Deploy info: group=$CODEDEPLOY_DEPLOYMENT_GROUP, branch=$DEPLOY_BRANCH, timestamp=$DEPLOY_TIMESTAMP, sha=${DEPLOY_SHA:0:7}"

    # 1순위: CodeDeploy Deployment Group에 따라 Parameter Store 경로 설정
    if [ -n "$CODEDEPLOY_DEPLOYMENT_GROUP" ]; then
        DEPLOYMENT_GROUP_LOWER=$(echo "$CODEDEPLOY_DEPLOYMENT_GROUP" | tr '[:upper:]' '[:lower:]')

        if [[ "$DEPLOYMENT_GROUP_LOWER" == *"dev"* ]]; then
            export PARAMETER_STORE_PATH="/Dev/AI/"
            echo "🛠️  Environment: Development (from deployment group: $CODEDEPLOY_DEPLOYMENT_GROUP)"
        elif [[ "$DEPLOYMENT_GROUP_LOWER" == *"stg"* ]] || [[ "$DEPLOYMENT_GROUP_LOWER" == *"staging"* ]]; then
            export PARAMETER_STORE_PATH="/Stg/AI/"
            echo "🧪 Environment: Staging (from deployment group: $CODEDEPLOY_DEPLOYMENT_GROUP)"
        elif [[ "$DEPLOYMENT_GROUP_LOWER" == *"prod"* ]]; then
            export PARAMETER_STORE_PATH="/Prod/AI/"
            echo "🚀 Environment: Production (from deployment group: $CODEDEPLOY_DEPLOYMENT_GROUP)"
        else
            echo "⚠️  Unknown deployment group: $CODEDEPLOY_DEPLOYMENT_GROUP"
        fi
    fi

    # 2순위: Deployment Group 매칭 실패 시 브랜치 기반 Fallback
    if [ -z "$PARAMETER_STORE_PATH" ] && [ -n "$DEPLOY_BRANCH" ]; then
        echo "💡 Falling back to branch-based detection..."
        case "$DEPLOY_BRANCH" in
            develop)
                export PARAMETER_STORE_PATH="/Dev/AI/"
                echo "🛠️  Environment: Development (from branch: $DEPLOY_BRANCH)"
                ;;
            release)
                export PARAMETER_STORE_PATH="/Stg/AI/"
                echo "🧪 Environment: Staging (from branch: $DEPLOY_BRANCH)"
                ;;
            main)
                export PARAMETER_STORE_PATH="/Prod/AI/"
                echo "🚀 Environment: Production (from branch: $DEPLOY_BRANCH)"
                ;;
            *)
                echo "⚠️  Unknown branch: $DEPLOY_BRANCH"
                ;;
        esac
    fi

    # 3순위: 여전히 결정되지 않았으면 기본값 사용
    if [ -z "$PARAMETER_STORE_PATH" ]; then
        echo "⚠️  Could not determine environment, using default /Prod/AI/"
        export PARAMETER_STORE_PATH="/Prod/AI/"
    fi
else
    echo "⚠️  .deploy-env file not found at $APP_DIR/.deploy-env"
    echo "💡 Using default Parameter Store path: /Prod/AI/"
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
# 2. Python 및 Poetry 경로 설정
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# pyenv 경로 설정 (존재하는 경우)
if [ -d "/home/ubuntu/.local/share/pyenv" ]; then
    export PYENV_ROOT="/home/ubuntu/.local/share/pyenv"
    export PATH="$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH"

    # pyenv 초기화
    if command -v pyenv &> /dev/null; then
        eval "$(pyenv init -)"
    fi
fi

# Poetry 경로 추가
export PATH="/home/ubuntu/.local/bin:$PATH"

# Poetry 설치 확인
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found! Please check after_install.sh"
    exit 1
fi

echo "✅ Poetry version: $(poetry --version)"
echo "✅ Python version: $(python3 --version)"

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
