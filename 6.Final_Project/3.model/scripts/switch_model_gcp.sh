#!/bin/bash

# GCP에서 vLLM 모델 교체 스크립트
# 사용법: ./switch_model_gcp.sh <MODEL_NAME>

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 사용 가능한 모델 목록
AVAILABLE_MODELS=(
    "MLP-KTLim/llama-3-Korean-Bllossom-8B"
    "facebook/opt-125m"
    "gpt2"
    "EleutherAI/polyglot-ko-1.3b"
)

# 사용법 출력
usage() {
    echo -e "${YELLOW}Usage: $0 <MODEL_NAME>${NC}"
    echo -e "${YELLOW}Available models:${NC}"
    for model in "${AVAILABLE_MODELS[@]}"; do
        echo -e "  - ${GREEN}${model}${NC}"
    done
    exit 1
}

# 인자 확인
if [ $# -eq 0 ]; then
    usage
fi

NEW_MODEL=$1
PORT="${PORT:-8000}"

# 모델 검증
VALID_MODEL=false
for model in "${AVAILABLE_MODELS[@]}"; do
    if [ "$model" == "$NEW_MODEL" ]; then
        VALID_MODEL=true
        break
    fi
done

if [ "$VALID_MODEL" = false ]; then
    echo -e "${RED}Error: Invalid model name${NC}"
    usage
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Switching vLLM Model${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "  New Model: ${GREEN}${NEW_MODEL}${NC}"
echo -e "  Port: ${GREEN}${PORT}${NC}"

# 기존 컨테이너 중지
CONTAINER_NAME="vllm-server"
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop ${CONTAINER_NAME}
    docker rm ${CONTAINER_NAME}
else
    echo -e "${YELLOW}No running container found${NC}"
fi

# 새 모델로 시작
echo -e "${GREEN}Starting vLLM with new model...${NC}"

docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    -p ${PORT}:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    vllm/vllm-openai:latest \
    vllm serve ${NEW_MODEL} \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096 \
    --trust-remote-code

echo -e "${GREEN}Waiting for server to start...${NC}"
sleep 15

# 서버 상태 확인
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:${PORT}/v1/models > /dev/null; then
        echo -e "${GREEN}✅ Model switched successfully!${NC}"
        echo -e "${GREEN}========================================${NC}"
        curl -s http://localhost:${PORT}/v1/models | jq '.'
        echo -e "${GREEN}========================================${NC}"
        exit 0
    fi

    echo -e "${YELLOW}Waiting... ($RETRY_COUNT/$MAX_RETRIES)${NC}"
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

echo -e "${RED}❌ Failed to switch model${NC}"
docker logs ${CONTAINER_NAME}
exit 1
