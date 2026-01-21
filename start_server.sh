#!/bin/bash

# Masking API Server Starter
# --reload 없이 실행하여 worker 프로세스 분리 문제 해결

echo "🚀 Starting FastAPI server without reload mode..."
echo "   (This fixes the 404 task_id error)"
echo ""

cd "$(dirname "$0")"

# 기존 프로세스 종료
echo "Stopping existing uvicorn processes..."
pkill -f "uvicorn app.main:app" 2>/dev/null || true
sleep 1

# 서버 시작 (--reload 제거)
echo "Starting server on http://0.0.0.0:8000"
# Poetry를 사용하여 실행 (PYTHONPATH를 현재 디렉토리로 설정)
PYTHONPATH=. poetry run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000

# 또는 개발 중이면 --reload 대신 수동으로 재시작하세요
