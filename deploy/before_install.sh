#!/bin/bash

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🧹 Running pre-installation cleanup..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

APP_DIR="/home/ubuntu/ai"

# 1. 이전 배포 백업 (옵션)
if [ -d "$APP_DIR" ]; then
    BACKUP_DIR="/home/ubuntu/backups/ai-$(date +%Y%m%d-%H%M%S)"
    echo "📦 Backing up existing deployment to $BACKUP_DIR"
    mkdir -p /home/ubuntu/backups
    cp -r "$APP_DIR" "$BACKUP_DIR" || true

    # 오래된 백업 삭제 (7일 이상)
    find /home/ubuntu/backups -name "ai-*" -type d -mtime +7 -exec rm -rf {} + 2>/dev/null || true
fi

# 2. 임시 파일 정리
echo "🗑️  Cleaning up temporary files..."
rm -rf "$APP_DIR/__pycache__" 2>/dev/null || true
rm -rf "$APP_DIR/app/__pycache__" 2>/dev/null || true
find "$APP_DIR" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find "$APP_DIR" -type f -name "*.pyc" -delete 2>/dev/null || true

# 3. Poetry 캐시 정리 (옵션)
echo "🗑️  Cleaning Poetry cache..."
rm -rf "$APP_DIR/.venv" 2>/dev/null || true

# 4. 로그 디렉토리 생성
echo "📁 Creating log directory..."
mkdir -p "$APP_DIR/logs"
chown ubuntu:ubuntu "$APP_DIR/logs"

# 5. 태스크 스토리지 디렉토리 유지 (기존 데이터 보존)
echo "💾 Preserving task storage..."
if [ -d "/tmp/masking_tasks" ]; then
    echo "ℹ️  Task storage exists, keeping it"
else
    mkdir -p /tmp/masking_tasks
    chown ubuntu:ubuntu /tmp/masking_tasks
fi

echo "✅ Pre-installation cleanup completed"
exit 0
