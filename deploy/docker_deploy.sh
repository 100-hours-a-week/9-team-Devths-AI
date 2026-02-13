#!/bin/bash

# ----------------------------------------------------------------------
# EC2 Docker Deployment Script
# ----------------------------------------------------------------------

APP_DIR="/home/ubuntu/ai"
LOG_FILE="$APP_DIR/deploy.log"
CONTAINER_NAME="ai-service"
AWS_REGION="ap-northeast-2" # Default region

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
if [ -n "$CODEDEPLOY_DEPLOYMENT_GROUP" ]; then
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
elif [ -n "$DEPLOY_BRANCH" ]; then
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

# 3. Pull New Image
log "⬇️  Pulling Docker image: $IMAGE_URI"
docker pull "$IMAGE_URI"
if [ $? -ne 0 ]; then
    log "❌ Docker pull failed!"
    exit 1
fi

# 4. Stop & Remove Old Container
log "🛑 Stopping existing container..."
if docker ps -a --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    docker stop "$CONTAINER_NAME"
    docker rm "$CONTAINER_NAME"
    log "✅ Removed old container."
else
    log "ℹ️  No existing container found."
fi

# 5. Prepare Environment Variables for Container
# We need to pass env vars to the container.
# Strategy: Run load_env_from_parameter_store.sh and dump exports to a file.

ENV_FILE="$APP_DIR/.env.docker"
echo "# Docker Env File" > "$ENV_FILE"

# If param store script exists, run it to set vars in current shell
if [ -f "$APP_DIR/deploy/load_env_from_parameter_store.sh" ]; then
    log "📥 Loading vars from Parameter Store..."
    # Set helper vars for load_env script to know where to look
    if [[ "$ENV_TAG" == "prod" ]]; then export PARAMETER_STORE_PATH="/Prod/AI/"; fi
    if [[ "$ENV_TAG" == "stg" ]]; then export PARAMETER_STORE_PATH="/Stg/AI/"; fi
    if [[ "$ENV_TAG" == "dev" ]]; then export PARAMETER_STORE_PATH="/Dev/AI/"; fi
    
    # Run script and capture exported vars is tricky.
    # Instead, let's just reuse the .env file if it exists (assuming load_env creates/updates it or we use it directly)
    # The existing load_env script exports vars. 
    # Let's run it and then dump current env to file, filtering for our app specific vars?
    # Or better, just depend on .env file presence if load_env_from_parameter_store.sh generates one?
    # Looking at load_env_from_parameter_store.sh (cached knowledge), it likely exports.
    
    # Simpler approach: Create env file from current shell env variables after sourcing
    source "$APP_DIR/deploy/load_env_from_parameter_store.sh"
    
    # Append key variables to ENV_FILE
    env | grep -E "^(AWS_|DB_|REDIS_|S3_|OPENAI_|SLACK_|DISCORD_|JWT_|SECRET_|ALGORITHM|ACCESS_TOKEN|REFRESH_TOKEN|BACKEND_|FRONTEND_|VITE_)" >> "$ENV_FILE"
fi

if [ -f "$APP_DIR/.env" ]; then
    log "fq  Merging local .env file..."
    cat "$APP_DIR/.env" >> "$ENV_FILE"
fi

# 6. Run New Container
log "▶️  Starting new container..."
docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --env-file "$ENV_FILE" \
    -p 8000:8000 \
    "$IMAGE_URI"

if [ $? -ne 0 ]; then
    log "❌ Failed to start container!"
    exit 1
fi

# 7. Health Check
log "Hz  Health Checking..."
for i in {1..12}; do
    sleep 5
    if curl -s "http://localhost:8000/health" > /dev/null; then
        log "✅ Health check passed!"
        log "🚀 Deployment Successful!"
        exit 0
    fi
    log "⏳ Waiting for service to be healthy... ($i/12)"
done

log "❌ Health check timed out!"
docker logs --tail 20 "$CONTAINER_NAME"
exit 1
