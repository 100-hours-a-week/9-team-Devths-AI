#!/bin/bash

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔄 Log Rotation and S3 Upload"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

APP_DIR="/home/ubuntu/ai"
LOG_DIR="$APP_DIR/logs"
LOG_FILE="$LOG_DIR/fastapi-app.log"
ARCHIVE_DIR="$LOG_DIR/archive"

# 타임스탬프 생성 (KST 기준)
TIMESTAMP=$(TZ=Asia/Seoul date +'%Y%m%d_%H%M%S')
DATE=$(TZ=Asia/Seoul date +'%Y-%m-%d')

# 환경 감지 (.deploy-env 파일에서 읽기)
if [ -f "$APP_DIR/.deploy-env" ]; then
    source "$APP_DIR/.deploy-env"
    ENVIRONMENT=$(echo "$DEPLOY_BRANCH" | sed 's/develop/dev/; s/release/stg/; s/main/prod/')
else
    ENVIRONMENT="unknown"
fi

echo "📅 Timestamp: $TIMESTAMP"
echo "🌍 Environment: $ENVIRONMENT"
echo "📂 Log File: $LOG_FILE"

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 로그 파일 존재 확인
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if [ ! -f "$LOG_FILE" ]; then
    echo "⚠️  Log file does not exist: $LOG_FILE"
    echo "✅ Nothing to rotate"
    exit 0
fi

# 로그 파일 크기 확인
LOG_SIZE=$(stat -f%z "$LOG_FILE" 2>/dev/null || stat -c%s "$LOG_FILE" 2>/dev/null)
LOG_SIZE_MB=$(echo "scale=2; $LOG_SIZE/1024/1024" | bc)

echo "📊 Current log size: ${LOG_SIZE_MB}MB"

# 로그 파일이 비어있거나 너무 작으면 스킵 (1MB 미만)
if [ "$LOG_SIZE" -lt 1048576 ]; then
    echo "⚠️  Log file is too small (< 1MB), skipping rotation"
    exit 0
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. 아카이브 디렉토리 생성
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

mkdir -p "$ARCHIVE_DIR"
chown ubuntu:ubuntu "$ARCHIVE_DIR" 2>/dev/null || true

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 로그 파일 로테이트 (복사 후 truncate)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ROTATED_LOG="$ARCHIVE_DIR/fastapi-app-${TIMESTAMP}.log"

echo "🔄 Rotating log file..."
echo "   Source: $LOG_FILE"
echo "   Destination: $ROTATED_LOG"

# 로그 파일 복사 (원본 유지)
cp "$LOG_FILE" "$ROTATED_LOG"

# 원본 파일 truncate (서버 재시작 없이 로그 초기화)
> "$LOG_FILE"

echo "✅ Log rotated successfully"

# 로테이트된 로그 압축
echo "🗜️  Compressing rotated log..."
gzip "$ROTATED_LOG"
COMPRESSED_LOG="${ROTATED_LOG}.gz"

if [ -f "$COMPRESSED_LOG" ]; then
    COMPRESSED_SIZE=$(stat -f%z "$COMPRESSED_LOG" 2>/dev/null || stat -c%s "$COMPRESSED_LOG" 2>/dev/null)
    COMPRESSED_SIZE_MB=$(echo "scale=2; $COMPRESSED_SIZE/1024/1024" | bc)
    echo "✅ Compressed: ${COMPRESSED_SIZE_MB}MB"
else
    echo "❌ Compression failed"
    exit 1
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. S3로 업로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# S3 버킷 결정
case "$ENVIRONMENT" in
    prod)
        S3_BUCKET="${S3_BUCKET_LOGS_PROD:-${S3_BUCKET_PROD}}"
        ;;
    stg)
        S3_BUCKET="${S3_BUCKET_LOGS:-${S3_BUCKET}}"
        ;;
    dev)
        S3_BUCKET="${S3_BUCKET_LOGS:-${S3_BUCKET}}"
        ;;
    *)
        S3_BUCKET="${S3_BUCKET_LOGS:-devths-ai-logs}"
        ;;
esac

# S3 키 생성: logs/{environment}/{date}/fastapi-app-{timestamp}.log.gz
S3_KEY="logs/${ENVIRONMENT}/${DATE}/fastapi-app-${TIMESTAMP}.log.gz"

echo "☁️  Uploading to S3..."
echo "   Bucket: s3://${S3_BUCKET}"
echo "   Key: $S3_KEY"

if aws s3 cp "$COMPRESSED_LOG" "s3://${S3_BUCKET}/${S3_KEY}" 2>&1; then
    echo "✅ Uploaded to S3: s3://${S3_BUCKET}/${S3_KEY}"

    # 업로드 성공 시 로컬 압축 파일 삭제 (선택)
    # 로컬에도 보관하려면 이 부분 주석 처리
    rm -f "$COMPRESSED_LOG"
    echo "🗑️  Local compressed file removed"
else
    echo "⚠️  S3 upload failed, keeping local file: $COMPRESSED_LOG"
    echo "💡 Check AWS credentials and S3 bucket permissions"
fi

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 오래된 로컬 아카이브 정리 (7일 이상)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "🗑️  Cleaning old local archives (older than 7 days)..."
find "$ARCHIVE_DIR" -name "*.log.gz" -type f -mtime +7 -delete 2>/dev/null || true
echo "✅ Cleanup completed"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Log rotation completed successfully"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit 0
