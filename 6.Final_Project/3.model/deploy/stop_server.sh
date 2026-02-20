#!/bin/bash

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛑 Stopping FastAPI server..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. uvicorn 프로세스 찾기
PIDS=$(pgrep -f "uvicorn app.main:app")

if [ -z "$PIDS" ]; then
    echo "ℹ️  No running uvicorn processes found."
    exit 0
fi

# 2. 프로세스 종료
echo "Found running processes: $PIDS"
for PID in $PIDS; do
    echo "Killing process $PID..."
    kill -15 $PID 2>/dev/null || true
done

# 3. 정상 종료 대기 (최대 10초)
echo "⏳ Waiting for graceful shutdown..."
for i in {1..10}; do
    if ! pgrep -f "uvicorn app.main:app" > /dev/null; then
        echo "✅ Server stopped successfully"
        exit 0
    fi
    sleep 1
done

# 4. 강제 종료 (여전히 실행 중이면)
REMAINING=$(pgrep -f "uvicorn app.main:app")
if [ ! -z "$REMAINING" ]; then
    echo "⚠️  Force killing remaining processes..."
    pkill -9 -f "uvicorn app.main:app"
    sleep 1
fi

echo "✅ Server stop completed"
exit 0
