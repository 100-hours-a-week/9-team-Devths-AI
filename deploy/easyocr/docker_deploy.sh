#!/bin/bash

# ----------------------------------------------------------------------
# EC2 Docker Deployment Script for GPU EasyOCR Service
# ----------------------------------------------------------------------

APP_DIR="/home/ubuntu/ai/easyocr"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/deploy.log"
CONTAINER_NAME="easyocr_server"
AWS_REGION="ap-northeast-2"

export PATH=$PATH:/usr/local/bin:/usr/bin:/bin:/usr/local/sbin:/usr/sbin:/sbin

mkdir -p "$LOG_DIR"
> "$LOG_FILE"

log() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
log "🚀 Starting EasyOCR GPU Docker Deployment..."
log "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Load Environment Variables (.deploy-env from GitHub Actions)
if [ -f "/home/ubuntu/ai/deploy/easyocr/.deploy-env" ]; then
    log "📄 Loading deployment info from .deploy-env..."
    source "/home/ubuntu/ai/deploy/easyocr/.deploy-env"
else
    log "⚠️  .deploy-env file not found."
fi

# 2. Determine Environment (dev/stg/prod) to select correct ECR Repo
if [ -n "$DEPLOYMENT_GROUP_NAME" ]; then
    log "ℹ️  Detected CodeDeploy Deployment Group: $DEPLOYMENT_GROUP_NAME"
    GROUP_LOWER=$(echo "$DEPLOYMENT_GROUP_NAME" | tr '[:upper:]' '[:lower:]')
    
    if [[ "$GROUP_LOWER" == *"prod"* ]]; then
        ENV_TAG="prod"
        ECR_REPO_NAME="devths/easyocr-prod"
    elif [[ "$GROUP_LOWER" == *"stg"* ]] || [[ "$GROUP_LOWER" == *"staging"* ]]; then
        ENV_TAG="stg"
        ECR_REPO_NAME="devths/easyocr-stg"
    else
        ENV_TAG="dev"
        ECR_REPO_NAME="devths/easyocr-dev"
    fi
elif [ -n "$CODEDEPLOY_DEPLOYMENT_GROUP" ]; then
    GROUP_LOWER=$(echo "$CODEDEPLOY_DEPLOYMENT_GROUP" | tr '[:upper:]' '[:lower:]')
    if [[ "$GROUP_LOWER" == *"prod"* ]]; then
        ENV_TAG="prod"
        ECR_REPO_NAME="devths/easyocr-prod"
    elif [[ "$GROUP_LOWER" == *"stg"* ]] || [[ "$GROUP_LOWER" == *"staging"* ]]; then
        ENV_TAG="stg"
        ECR_REPO_NAME="devths/easyocr-stg"
    else
        ENV_TAG="dev"
        ECR_REPO_NAME="devths/easyocr-dev"
    fi
else
    ENV_TAG="dev"
    ECR_REPO_NAME="devths/easyocr-dev"
    log "⚠️  Could not determine environment, defaulting to DEV."
fi

log "🌍 Target Environment: $ENV_TAG"
log "📦 Target ECR Repository: $ECR_REPO_NAME"

# 3. Login to ECR
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

# 4. Pull Image
log "⬇️  Pulling Docker image: $IMAGE_URI"
docker pull "$IMAGE_URI"
if [ $? -ne 0 ]; then
    log "❌ Docker pull failed!"
    exit 1
fi

# 5. Stop & Remove Old Container
log "🛑 Stopping existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
    log "✅ Removed old container."
else
    log "ℹ️  No existing container found."
fi

# 6. Run New Container (GPU Enabled)
log "▶️  Starting new GPU container..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --log-driver json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    --gpus all \
    -p 8002:8000 \
    "$IMAGE_URI"

if [ $? -ne 0 ]; then
    log "❌ Failed to start GPU container!"
    exit 1
fi

log "✅ GPU Container started successfully."

# 7. Health Check
log "Hz  Health Checking..."
for i in {1..12}; do
    sleep 5
    # EasyOCR internal port is 8000, mapped to 8002 on host
    if curl -s "http://localhost:8002/health" > /dev/null; then
        log "✅ Health check passed!"
        log "🚀 Deployment Successful!"
        exit 0
    fi
    log "⏳ Waiting for service to be healthy... ($i/12)"
done

log "❌ Health check timed out!"
docker logs --tail 20 "$CONTAINER_NAME"
exit 1
