#!/bin/bash

# GCP vLLM 서버 및 FastAPI 테스트 스크립트

set -e

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 설정
FASTAPI_URL="${FASTAPI_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-your-api-key-here}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}GCP vLLM & FastAPI Test Script${NC}"
echo -e "${GREEN}========================================${NC}"
echo -e "FastAPI URL: ${BLUE}${FASTAPI_URL}${NC}"
echo ""

# 1. 헬스 체크
echo -e "${YELLOW}[1/5] Health Check${NC}"
if curl -s "${FASTAPI_URL}/health" | jq '.' > /dev/null 2>&1; then
    echo -e "${GREEN}✅ FastAPI server is healthy${NC}"
else
    echo -e "${RED}❌ FastAPI server is not responding${NC}"
    exit 1
fi
echo ""

# 2. 모델 목록 조회
echo -e "${YELLOW}[2/5] List Available Models${NC}"
MODELS_RESPONSE=$(curl -s -X GET "${FASTAPI_URL}/ai/models" \
    -H "X-API-Key: ${API_KEY}")

if echo "$MODELS_RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Available models:${NC}"
    echo "$MODELS_RESPONSE" | jq '.available_models[]' -r | while read -r model; do
        echo -e "  - ${BLUE}${model}${NC}"
    done
    CURRENT_MODEL=$(echo "$MODELS_RESPONSE" | jq -r '.current_model')
    echo -e "${GREEN}Current model: ${BLUE}${CURRENT_MODEL}${NC}"
else
    echo -e "${RED}❌ Failed to get models${NC}"
    echo "$MODELS_RESPONSE" | jq '.'
    exit 1
fi
echo ""

# 3. 채팅 테스트 (현재 모델)
echo -e "${YELLOW}[3/5] Test Chat with Current Model (${CURRENT_MODEL})${NC}"
CHAT_RESPONSE=$(curl -s -X POST "${FASTAPI_URL}/ai/chat" \
    -H "X-API-Key: ${API_KEY}" \
    -H "Content-Type: application/json" \
    -d '{
        "user_id": "test_user",
        "room_id": "test_room",
        "message": "안녕하세요, 간단히 자기소개 해주세요.",
        "model": "vllm",
        "history": [],
        "context": {
            "mode": "general"
        }
    }')

if echo "$CHAT_RESPONSE" | grep -q "data:"; then
    echo -e "${GREEN}✅ Chat response received${NC}"
    echo -e "${BLUE}Response preview:${NC}"
    echo "$CHAT_RESPONSE" | head -n 5
else
    echo -e "${RED}❌ Chat failed${NC}"
    echo "$CHAT_RESPONSE"
fi
echo ""

# 4. 모델 전환 테스트 (선택적)
echo -e "${YELLOW}[4/5] Test Model Switch (Optional)${NC}"
read -p "Switch to a different model? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${BLUE}Available models:${NC}"
    echo "$MODELS_RESPONSE" | jq '.available_models[]' -r | nl

    read -p "Enter model number to switch: " MODEL_NUM
    NEW_MODEL=$(echo "$MODELS_RESPONSE" | jq -r ".available_models[$((MODEL_NUM-1))]")

    if [ "$NEW_MODEL" != "null" ] && [ -n "$NEW_MODEL" ]; then
        echo -e "${YELLOW}Switching to: ${BLUE}${NEW_MODEL}${NC}"
        SWITCH_RESPONSE=$(curl -s -X POST "${FASTAPI_URL}/ai/models/switch?model_name=${NEW_MODEL}" \
            -H "X-API-Key: ${API_KEY}")

        if echo "$SWITCH_RESPONSE" | jq -e '.success' > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Model switched successfully${NC}"
            echo "$SWITCH_RESPONSE" | jq '.'
        else
            echo -e "${RED}❌ Model switch failed${NC}"
            echo "$SWITCH_RESPONSE" | jq '.'
        fi
    else
        echo -e "${RED}Invalid model number${NC}"
    fi
else
    echo -e "${BLUE}Skipping model switch${NC}"
fi
echo ""

# 5. 최종 상태 확인
echo -e "${YELLOW}[5/5] Final Status Check${NC}"
FINAL_STATUS=$(curl -s -X GET "${FASTAPI_URL}/ai/models" \
    -H "X-API-Key: ${API_KEY}")
echo "$FINAL_STATUS" | jq '.'
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Test Complete!${NC}"
echo -e "${GREEN}========================================${NC}"
