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

# 5. Prepare Environment Variables for Container (Memory Injection)
# We need to pass env vars to the container without writing them to disk.
# Strategy: Source load_env_from_parameter_store.sh and construct -e flags dynamically.

log "📥 Loading vars from Parameter Store..."
# Set helper vars for load_env script to know where to look
if [[ "$ENV_TAG" == "prod" ]]; then export PARAMETER_STORE_PATH="/Prod/AI/"; fi
if [[ "$ENV_TAG" == "stg" ]]; then export PARAMETER_STORE_PATH="/Stg/AI/"; fi
if [[ "$ENV_TAG" == "dev" ]]; then export PARAMETER_STORE_PATH="/Dev/AI/"; fi

# Source the script to load variables into the current shell session
if [ -f "$APP_DIR/deploy/load_env_from_parameter_store.sh" ]; then
    source "$APP_DIR/deploy/load_env_from_parameter_store.sh"
else
    log "⚠️  load_env_from_parameter_store.sh not found. Skipping Parameter Store load."
fi

# Load local .env if exists (for local testing or overrides)
if [ -f "$APP_DIR/.env" ]; then
    log "📄 Loading local .env file..."
    set -a # Automatically export all variables
    source "$APP_DIR/.env"
    set +a
fi

# Construct Docker Environment Arguments
log "🔨 Constructing Docker environment arguments..."
DOCKER_ENV_ARGS=()

# Filter and add variables to the array
# We use the same grep pattern to identify relevant variables
ENV_VARS=$(env | grep -E "^(AWS_|DB_|REDIS_|S3_|OPENAI_|SLACK_|DISCORD_|JWT_|SECRET_|ALGORITHM|ACCESS_TOKEN|REFRESH_TOKEN|BACKEND_|FRONTEND_|VITE_|GOOGLE_|GEMINI_|GCP_|VLLM_|CLOVA_|EASYOCR_|CHROMA_|CELERY_|LANGFUSE_|RAG_|EVAL_|INTERVIEW_|LLM_|ENVIRONMENT|DEBUG|LOG_LEVEL)" | cut -d= -f1)

for var_name in $ENV_VARS; do
    # Get the value of the variable indirectly
    var_value="${!var_name}"
    if [ -n "$var_value" ]; then
        # Add to array in the format -e KEY=VALUE
        # IMPORTANT: We verify the value is not empty to avoid empty strings if something went wrong
        DOCKER_ENV_ARGS+=("-e" "$var_name=$var_value")
    fi
done

log "✅ Prepared ${#DOCKER_ENV_ARGS[@]} environment variables for injection."

# 6. Run New Container
log "▶️  Starting new container..."
# create a temporary array for the command to handle quoting correctly
cmd=(docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    --log-driver json-file \
    --log-opt max-size=10m \
    --log-opt max-file=3 \
    "${DOCKER_ENV_ARGS[@]}" \
    -p 8000:8000 \
    "$IMAGE_URI")

# Execute the command
"${cmd[@]}"

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
