#!/bin/bash

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AWS Systems Manager Parameter Store에서 환경변수 로드
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# 사용법:
#   source deploy/load_env_from_parameter_store.sh

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# AWS Parameter Store에서 환경변수 가져오기
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Parameter Store 경로 설정
# develop 브랜치: /Dev/AI/
# release 브랜치: /Stg/AI/
# main 브랜치: /Prod/AI/
PARAMETER_PATH="${PARAMETER_STORE_PATH:-/Prod/AI/}"

echo "☁️  Loading environment variables from AWS Parameter Store..."
echo "📍 Path: $PARAMETER_PATH"

# AWS CLI 설치 확인
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI not found! Please install AWS CLI first."
    return 1 2>/dev/null || exit 1
fi

# Parameter Store에서 모든 파라미터 가져오기
PARAMS=$(aws ssm get-parameters-by-path \
    --path "$PARAMETER_PATH" \
    --with-decryption \
    --query 'Parameters[*].[Name,Value]' \
    --output text 2>/dev/null)

if [ -z "$PARAMS" ]; then
    echo "⚠️  No parameters found at $PARAMETER_PATH"
    echo "💡 Tip: Set parameters in AWS Systems Manager Parameter Store first"
    echo "   Example parameters:"
    echo "   - ${PARAMETER_PATH}GOOGLE_API_KEY"
    echo "   - ${PARAMETER_PATH}API_KEY"
    echo "   - ${PARAMETER_PATH}GCP_VLLM_BASE_URL"
    echo "   - ${PARAMETER_PATH}HOST"
    echo "   - ${PARAMETER_PATH}PORT"
    return 1 2>/dev/null || exit 1
fi

# 파라미터를 환경변수로 export
echo "📥 Exporting parameters as environment variables..."
while IFS=$'\t' read -r name value; do
    # 파라미터 이름에서 경로 제거 (예: /devths/ai/prod/API_KEY -> API_KEY)
    var_name=$(echo "$name" | sed "s|${PARAMETER_PATH}||")
    export "$var_name=$value"
    echo "   ✓ $var_name"
done <<< "$PARAMS"

echo "✅ Environment variables loaded from Parameter Store"

# 필수 환경변수 검증
REQUIRED_VARS=("GOOGLE_API_KEY" "API_KEY")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -gt 0 ]; then
    echo "⚠️  Warning: Missing required environment variables:"
    for var in "${MISSING_VARS[@]}"; do
        echo "   - $var"
    done
    echo "💡 Please set these in Parameter Store at: $PARAMETER_PATH"
fi

return 0 2>/dev/null || exit 0
