#!/bin/bash

# ----------------------------------------------------------------------
# EC2 Docker Deployment Script
# ----------------------------------------------------------------------

APP_DIR="/home/ubuntu/ai"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/deploy.log"
COMPOSE_FILE="$APP_DIR/docker-compose.cicd.yml"
COMPOSE_PROJECT="ai"  # --project-name: compose 컨테이너 관리 일관성 보장
AWS_REGION="ap-northeast-2" # Default region

# AWS CodeDeploy의 환경 변수(PATH) 초기화 문제 방지를 위해 명시적으로 PATH 추가
export PATH=$PATH:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin

# 로그 디렉토리 권한 문제 방지 (ubuntu 홈 디렉토리 하위에 생성)
# 2>/dev/null || true: 권한 부족 등으로 실패해도 스크립트는 계속 진행
mkdir -p "$LOG_DIR" 2>/dev/null || true
touch "$LOG_FILE" 2>/dev/null || true

# 디스크 용량 누적 방지를 위해 매 배포 시 이전 배포 로그를 빈 파일로 덮어쓰기(초기화)합니다.
> "$LOG_FILE" 2>/dev/null || true

# Logging helper
log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "🚀 Starting Docker Deployment..."
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Initialize & Load Environment Variables
cd "$APP_DIR" || { log "❌ Failed to change directory to $APP_DIR"; exit 1; }

# Load deployment info from .deploy-env (created by GitHub Actions)
if [ -f "$APP_DIR/.deploy-env" ]; then
    log "📄 Loading deployment info from .deploy-env..."
    source "$APP_DIR/.deploy-env"
else
    log "⚠️  .deploy-env file not found."
fi

# Determine Environment (Dev/Stg/Prod) and ECR Repo
# Reuse logic from start_server_deploy.sh for consistency
# PRIORITY 1: CodeDeploy Runtime Environment Variable (DEPLOYMENT_GROUP_NAME)
if [ -n "$DEPLOYMENT_GROUP_NAME" ]; then
    log "ℹ️  Detected CodeDeploy Deployment Group: $DEPLOYMENT_GROUP_NAME"
    GROUP_LOWER=$(echo "$DEPLOYMENT_GROUP_NAME" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$GROUP_LOWER" == *"prod"* ]]; then
        ENV_TAG="prod"
        ECR_REPO_NAME="devths/ai-prod"
    elif [[ "$GROUP_LOWER" == *"stg"* ]] || [[ "$GROUP_LOWER" == *"staging"* ]]; then
        ENV_TAG="stg"
        ECR_REPO_NAME="devths/ai-stg"
    else
        ENV_TAG="dev"
        ECR_REPO_NAME="devths/ai-dev"
    fi

# PRIORITY 2: Build-time Variable from .deploy-env (CODEDEPLOY_DEPLOYMENT_GROUP)
elif [ -n "$CODEDEPLOY_DEPLOYMENT_GROUP" ]; then
    log "ℹ️  Using build-time CODEDEPLOY_DEPLOYMENT_GROUP: $CODEDEPLOY_DEPLOYMENT_GROUP"
    GROUP_LOWER=$(echo "$CODEDEPLOY_DEPLOYMENT_GROUP" | tr '[:upper:]' '[:lower:]')
    if [[ "$GROUP_LOWER" == *"prod"* ]]; then
        ENV_TAG="prod"
        ECR_REPO_NAME="devths/ai-prod"
    elif [[ "$GROUP_LOWER" == *"stg"* ]] || [[ "$GROUP_LOWER" == *"staging"* ]]; then
        ENV_TAG="stg"
        ECR_REPO_NAME="devths/ai-stg"
    else
        ENV_TAG="dev"
        ECR_REPO_NAME="devths/ai-dev"
    fi

# PRIORITY 3: Branch Name (Fallback)
elif [ -n "$DEPLOY_BRANCH" ]; then
    log "ℹ️  Using branch name: $DEPLOY_BRANCH"
    if [[ "$DEPLOY_BRANCH" == "main" ]]; then
        ENV_TAG="prod"
        ECR_REPO_NAME="devths/ai-prod"
    elif [[ "$DEPLOY_BRANCH" == "release/"* ]]; then
        ENV_TAG="stg"
        ECR_REPO_NAME="devths/ai-stg"
    else
        ENV_TAG="dev"
        ECR_REPO_NAME="devths/ai-dev"
    fi
else
    ENV_TAG="dev"
    ECR_REPO_NAME="devths/ai-dev"
    log "⚠️  Could not determine environment, defaulting to DEV."
fi

log "🌍 Target Environment: $ENV_TAG"
log "📦 ECR Repository: $ECR_REPO_NAME"

# 2. Login to ECR
log "🔐 Logging into AWS ECR..."
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_REGISTRY="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com"
IMAGE_URI="$ECR_REGISTRY/$ECR_REPO_NAME:latest"

aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_REGISTRY"
if [ $? -ne 0 ]; then
    log "❌ ECR Login failed!"
    exit 1
fi
log "✅ ECR Login successful."

# 3. Load vars from Parameter Store & Export to shell (디스크 기록 없음)
# docker compose는 현재 셸의 export된 변수를 자동으로 읽습니다.
log "📥 Loading vars from Parameter Store..."
if [[ "$ENV_TAG" == "prod" ]]; then export PARAMETER_STORE_PATH="/Prod/AI/"; fi
if [[ "$ENV_TAG" == "stg" ]]; then export PARAMETER_STORE_PATH="/Stg/AI/"; fi
if [[ "$ENV_TAG" == "dev" ]]; then export PARAMETER_STORE_PATH="/Dev/AI/"; fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
if [ -f "$SCRIPT_DIR/load_env_from_parameter_store.sh" ]; then
    source "$SCRIPT_DIR/load_env_from_parameter_store.sh" >> "$LOG_FILE" 2>&1
else
    log "⚠️  load_env_from_parameter_store.sh not found at $SCRIPT_DIR. Skipping Parameter Store load."
fi

# IMAGE_URI를 셸 환경변수로 export (docker-compose.cicd.yml의 ${IMAGE_URI} 참조)
export IMAGE_URI="$IMAGE_URI"

# Parameter Store에서 로드된 변수 전체를 셸 환경변수로 export
# 디스크에 기록하지 않음 — 스크립트 종료 시 메모리에서 자동 소멸
if [ -n "$LOADED_PARAM_KEYS" ]; then
    EXPORTED_COUNT=0
    for var_name in $LOADED_PARAM_KEYS; do
        var_value="${!var_name}"
        if [ -n "$var_value" ]; then
            export "$var_name"   # 셸 메모리에만 존재
            EXPORTED_COUNT=$((EXPORTED_COUNT + 1))
        fi
    done
    log "✅ $EXPORTED_COUNT variables exported to shell environment (no disk write)."
fi

# Promtail LOKI_URL: AWS EC2 API로 모니터링 서버 Private IP 조회 후 export
log "🔍 Resolving Loki URL from monitoring instance..."
if [[ "$DEPLOYMENT_GROUP_NAME" == *"Prod"* ]] || [[ "$DEPLOY_BRANCH" == "main" ]]; then
    TARGET_INSTANCE_NAME="devths-v2-prod-monitoring"
else
    TARGET_INSTANCE_NAME="devths-v2-nonprod-monitoring"
fi

MONITORING_PRIVATE_IP=$(aws ec2 describe-instances \
    --region "$AWS_REGION" \
    --filters "Name=tag:Name,Values=$TARGET_INSTANCE_NAME" "Name=instance-state-name,Values=running" \
    --query "Reservations[*].Instances[*].PrivateIpAddress" \
    --output text 2>/dev/null | awk '{print $1}')

if [ -n "$MONITORING_PRIVATE_IP" ]; then
    export LOKI_URL="http://${MONITORING_PRIVATE_IP}:3100/loki/api/v1/push"
    log "✅ LOKI_URL resolved: $LOKI_URL"
else
    log "⚠️  Could not resolve monitoring server IP. Promtail will start without Loki target."
    export LOKI_URL=""
fi

# 4. Pre-flight: 필수 환경변수 존재 여부 확인 (compose up 전에 즉시 감지)
log "🔍 Pre-flight check: Validating required environment variables..."
PREFLIGHT_FAIL=false
REQUIRED_VARS=("REDIS_URL" "CELERY_BROKER_URL" "CELERY_RESULT_BACKEND" "CHROMA_SERVER_HOST" "API_KEY")
for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        log "❌ Missing required variable: $var"
        PREFLIGHT_FAIL=true
    else
        log "   ✅ $var is set."
    fi
done
if [ "$PREFLIGHT_FAIL" = "true" ]; then
    log "❌ Pre-flight failed: Required environment variables are missing. Aborting deployment."
    exit 1
fi
log "✅ Pre-flight check passed."

# 5. Stop existing services (if any)
log "🛑 Stopping existing services..."

# [마이그레이션 정리] 기존 컨테이너 정리 로직 수정
# docker compose down이 이전 배포 버전의 컨테이너를 Graceful(SIGTERM)하게 꺼줍니다.
# 잔여 고아(Orphan) 컨테이너 및 죽은(Exited) 컨테이너만 정리합니다.
log "🧹 Removing legacy or orphaned containers..."
docker container prune -f 2>&1 | tee -a "$LOG_FILE"

if [ -f "$COMPOSE_FILE" ]; then
    docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" down --remove-orphans 2>&1 | tee -a "$LOG_FILE"
    log "✅ Existing services stopped."
fi

# 5. Pull new image
log "⬇️  Pulling new Docker image: $IMAGE_URI"
docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" pull 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    log "❌ Docker image pull failed!"
    exit 1
fi

# 6. Start all services via Docker Compose
log "▶️  Starting all services (ai-endpoint, celery-worker-trend, celery-worker-extract, celery-beat, promtail)..."
COMPOSE_UP_LOG="$LOG_DIR/compose_up.log"
docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" up -d 2>&1 | tee -a "$LOG_FILE" "$COMPOSE_UP_LOG"
UP_EXIT_CODE=${PIPESTATUS[0]}
if [ $UP_EXIT_CODE -ne 0 ]; then
    log "❌ docker compose up failed (exit code: $UP_EXIT_CODE)! Dumping diagnostics..."

    # [1] compose up 자체 출력 (에러 메시지 원문)
    log "=== [1/4] docker compose up output ==="
    cat "$COMPOSE_UP_LOG" >> "$LOG_FILE" 2>/dev/null

    # [2] 전체 컨테이너 상태
    log "=== [2/4] Container Status (docker ps -a) ==="
    docker ps -a 2>&1 | tee -a "$LOG_FILE"

    # [3] compose 서비스별 상태 확인 (exited/created 여부 포함)
    log "=== [3/4] Service-level Status (docker compose ps) ==="
    docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" ps -a 2>&1 | tee -a "$LOG_FILE"

    # [4] 각 서비스별 로그 개별 출력 (Created 상태 포함)
    log "=== [4/4] Per-service Logs (last 30 lines each) ==="
    for SERVICE in ai-endpoint celery-worker-trend celery-worker-extract celery-beat promtail; do
        log "--- [$SERVICE] ---"
        docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" logs --tail=30 "$SERVICE" 2>&1 | tee -a "$LOG_FILE" || \
            log "  (no logs available for $SERVICE — container may not have started)"
    done

    exit 1
fi

log "✅ All services started successfully."

# 7. Health Check (24 × 5s = 120초 대기 — 새 ML 서비스 기동 여유)
log "🏥 Health Checking (up to 120s)..."
for i in {1..24}; do
    sleep 5
    if curl -s "http://localhost:8000/health" > /dev/null; then
        log "✅ Health check passed!"

        # 8. Trigger Auto-Embedding if flagged by CI/CD
        if [ "$TRIGGER_DATA_EMBEDDING" = "true" ]; then
            log "📂 Data directory change detected. Triggering auto-embedding..."
            
            # [수정] docker exec(컨테이너 내부 파일 의존) -> docker run(호스트 디렉토리 마운트 + 일회용 실행)
            # 호스트의 data/ 폴더만 읽기 전용(ro)으로 안전하게 마운트하여 Dockerfile 미포함 문제 해결
            # 이미 로드된 환경변수 중 필수인 것들을 명시적으로 주입
            docker run --rm -v "$(pwd)/data:/app/data:ro" \
                -e CHROMA_SERVER_HOST="${CHROMA_SERVER_HOST}" \
                -e CHROMA_SERVER_PORT="${CHROMA_SERVER_PORT:-8000}" \
                -e GOOGLE_API_KEY="${GOOGLE_API_KEY}" \
                -e INTERVIEW_DATASET_FILE="${INTERVIEW_DATASET_FILE}" \
                "${IMAGE_URI}" \
                poetry run python scripts/auto_embed_data.py 2>&1 | tee -a "$LOG_FILE"
                
            if [ ${PIPESTATUS[0]} -eq 0 ]; then
                log "✅ Auto-embedding completed successfully."
            else
                log "⚠️  Auto-embedding failed! Check the logs above."
            fi
        fi

        log "🚀 Deployment Successful!"
        exit 0
    fi
    log "⏳ Waiting for service to be healthy... ($i/24)"
done

log "❌ Health check timed out! Dumping ai-endpoint logs:"
docker compose -p "$COMPOSE_PROJECT" -f "$COMPOSE_FILE" logs --tail 50 ai-endpoint 2>&1 | tee -a "$LOG_FILE"
exit 1
