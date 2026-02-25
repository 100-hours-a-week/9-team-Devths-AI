#!/bin/bash

# GCP vLLM 서버 시작 스크립트
# 이 스크립트는 GCP VM에서 실행됩니다.

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GCP vLLM Server Startup Script${NC}"
echo -e "${GREEN}========================================${NC}"

# 환경 변수 (GCP VM에서 수정)
MODEL_NAME="${MODEL_NAME:-MLP-KTLim/llama-3-Korean-Bllossom-8B}"
PORT="${PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

echo -e "${YELLOW}Configuration:${NC}"
echo -e "  Model: ${GREEN}${MODEL_NAME}${NC}"
echo -e "  Port: ${GREEN}${PORT}${NC}"
echo -e "  GPU Memory: ${GREEN}${GPU_MEMORY_UTILIZATION}${NC}"
echo -e "  Max Model Length: ${GREEN}${MAX_MODEL_LEN}${NC}"

# Docker 실행 여부 확인
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    exit 1
fi

# 기존 컨테이너 확인 및 중지
CONTAINER_NAME="vllm-server"
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo -e "${YELLOW}Stopping existing container...${NC}"
    docker stop ${CONTAINER_NAME} || true
    docker rm ${CONTAINER_NAME} || true
fi

# vLLM Docker 컨테이너 시작
echo -e "${GREEN}Starting vLLM Docker container...${NC}"

docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    -p ${PORT}:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --ipc=host \
    vllm/vllm-openai:latest \
    vllm serve ${MODEL_NAME} \
    --host 0.0.0.0 \
    --port 8000 \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --max-model-len ${MAX_MODEL_LEN} \
    --trust-remote-code

echo -e "${GREEN}Waiting for vLLM server to start...${NC}"
sleep 10

# 서버 상태 확인
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:${PORT}/v1/models > /dev/null; then
        echo -e "${GREEN}✅ vLLM server is running!${NC}"
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Server Info:${NC}"
        curl -s http://localhost:${PORT}/v1/models | jq '.'
        echo -e "${GREEN}========================================${NC}"
        echo -e "${GREEN}Access URL: http://localhost:${PORT}${NC}"
        echo -e "${GREEN}Models endpoint: http://localhost:${PORT}/v1/models${NC}"
        echo -e "${GREEN}========================================${NC}"
        exit 0
    fi

    echo -e "${YELLOW}Waiting for server... ($RETRY_COUNT/$MAX_RETRIES)${NC}"
    sleep 5
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

echo -e "${RED}❌ Failed to start vLLM server${NC}"
echo -e "${YELLOW}Check Docker logs:${NC}"
docker logs ${CONTAINER_NAME}
exit 1
