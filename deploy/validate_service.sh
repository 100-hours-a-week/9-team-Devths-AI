#!/bin/bash

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Validating deployed service..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

APP_DIR="/home/ubuntu/ai"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/fastapi-app.log"
PORT="${PORT:-8000}"
MAX_RETRIES=30
RETRY_INTERVAL=2

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 프로세스 확인 (Docker Container)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "🔎 Checking if Docker container is running..."
CONTAINER_NAME="ai-service"
if docker ps --format '{{.Names}}' | grep -q "^$CONTAINER_NAME$"; then
    echo "✅ Docker container '$CONTAINER_NAME' is running."
else
    echo "❌ Docker container '$CONTAINER_NAME' is NOT running!"
    docker ps -a | grep "$CONTAINER_NAME"
    exit 1
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 헬스 체크 엔드포인트 확인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "🏥 Checking /health endpoint..."
echo "📍 URL: http://localhost:$PORT/health"
echo "⏳ Max retries: $MAX_RETRIES (interval: ${RETRY_INTERVAL}s)"

for i in $(seq 1 $MAX_RETRIES); do
    echo -n "   Attempt $i/$MAX_RETRIES... "

    # HTTP 상태 코드와 응답 본문 가져오기
    HTTP_CODE=$(curl -s -o /tmp/health_response.txt -w "%{http_code}" http://localhost:$PORT/health 2>/dev/null)
    RESPONSE=$(cat /tmp/health_response.txt 2>/dev/null)

    if [ "$HTTP_CODE" = "200" ]; then
        echo "✅ Success (HTTP $HTTP_CODE)"
        echo "📄 Response: $RESPONSE"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "✅ Service validation passed!"
        echo "🎉 Deployment successful!"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        rm -f /tmp/health_response.txt
        exit 0
    else
        echo "❌ Failed (HTTP $HTTP_CODE)"
        if [ $i -lt $MAX_RETRIES ]; then
            echo "   Retrying in ${RETRY_INTERVAL}s..."
            sleep $RETRY_INTERVAL
        fi
    fi
done

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 헬스 체크 실패 시 디버깅 정보 출력
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "❌ Health check failed after $MAX_RETRIES attempts"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "📋 Debugging Information:"
echo ""

# 포트 확인
echo "1. Port status:"
netstat -tuln | grep $PORT || echo "   Port $PORT is not listening"
echo ""

# 프로세스 확인
echo "2. Running processes:"
docker ps -f name=ai-service || echo "   No ai-service container found"
echo ""

# 최근 로그
echo "3. Last 50 lines of application log:"
docker logs --tail 50 ai-service 2>/dev/null || echo "   Could not retrieve docker logs for ai-service"
echo ""

# 환경변수 확인 (민감 정보 제외)
echo "4. Environment check:"
if [ -n "$GOOGLE_API_KEY" ]; then
    echo "   ✅ GOOGLE_API_KEY is set"
else
    echo "   ❌ GOOGLE_API_KEY is not set"
fi

if [ -n "$GEMINI_API_KEY" ]; then
    echo "   ✅ GEMINI_API_KEY is set"
else
    echo "   ❌ GEMINI_API_KEY is not set"
fi

if [ -n "$API_KEY" ]; then
    echo "   ✅ API_KEY is set"
else
    echo "   ❌ API_KEY is not set"
fi
echo ""

# 디스크 용량 확인
echo "5. Disk space:"
df -h "$APP_DIR" | tail -1
echo ""

rm -f /tmp/health_response.txt
exit 1
