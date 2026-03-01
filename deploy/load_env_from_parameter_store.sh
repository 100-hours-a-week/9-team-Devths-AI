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

# AWS Region 자동 감지 (IMDSv2 지원) 및 Fallback
if [ -z "$AWS_REGION" ] && [ -z "$AWS_DEFAULT_REGION" ]; then
    TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" -s --max-time 2)
    if [ -n "$TOKEN" ]; then
        REGION=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" -s --max-time 2 http://169.254.169.254/latest/meta-data/placement/region)
    else
        # Fallback for IMDSv1
        REGION=$(curl -s --max-time 2 http://169.254.169.254/latest/meta-data/placement/region)
    fi
    
    # 만약 Metadata 서버 접근에 실패했다면 기본값 주입
    if [ -z "$REGION" ]; then
        REGION="ap-northeast-2"
        echo "⚠️  Failed to detect region from IMDS. Falling back to default: $REGION"
    fi

    export AWS_REGION="$REGION"
    export AWS_DEFAULT_REGION="$REGION"
    echo "🌍 AWS Region set to: $REGION"
fi

# Parameter Store에서 모든 파라미터 가져오기
# --no-paginate: NextToken 페이지네이션 없이 전체 결과 한 번에 반환 (--max-items 제거)
PARAMS=$(aws ssm get-parameters-by-path \
    --region "${AWS_REGION:-ap-northeast-2}" \
    --path "$PARAMETER_PATH" \
    --recursive \
    --with-decryption \
    --no-paginate \
    --query 'Parameters[*].[Name,Value]' \
    --output text)

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
PARAM_KEYS=()

# herestring(<<<) 대신 process substitution(<())으로 변경
# → trailing newline으로 인한 마지막 줄 파싱 오류 방지
while IFS=$'\t' read -r name value; do
    # bash parameter expansion으로 경로 제거 (sed 특수문자 문제 방지)
    # 예: /Dev/AI/GOOGLE_API_KEY → GOOGLE_API_KEY
    var_name="${name#"${PARAMETER_PATH}"}"

    # 변수명이 비어있거나 경로 제거가 안 된 경우 건너뜀
    if [ -z "$var_name" ] || [ "$var_name" = "$name" ]; then
        echo "   ⏭️  Skipping invalid parameter: $name"
        continue
    fi

    # export 시 값에 공백·특수문자 포함돼도 안전하도록 printf + declare 활용
    export "$var_name"
    printf -v "$var_name" '%s' "$value"
    declare -gx "$var_name"

    PARAM_KEYS+=("$var_name")
    echo "   ✓ $var_name"
done < <(echo "$PARAMS")

export LOADED_PARAM_KEYS="${PARAM_KEYS[*]}"
echo "✅ Loaded ${#PARAM_KEYS[@]} variables from Parameter Store (Path: $PARAMETER_PATH)"

# 필수 환경변수 검증
REQUIRED_VARS=("API_KEY")
MISSING_VARS=()

# Google/Gemini API Key는 둘 중 하나만 있어도 됨
if [ -z "$GOOGLE_API_KEY" ] && [ -z "$GEMINI_API_KEY" ]; then
    MISSING_VARS+=("GOOGLE_API_KEY or GEMINI_API_KEY")
fi

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
