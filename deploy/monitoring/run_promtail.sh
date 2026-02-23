#!/bin/bash
# ----------------------------------------------------------------------
# AI 서비스 로그 수집용 (Standalone Docker Run)
# 용도: FastAPI, Celery 등의 컨테이너 로그(stdout)를 Loki로 전송
# ----------------------------------------------------------------------

# 스크립트 실행 경로 기준 설정 디렉토리
CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CONFIG_FILE="${CURRENT_DIR}/promtail-config.yaml"
CONTAINER_NAME="ai-promtail"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 설정 파일을 찾을 수 없습니다: $CONFIG_FILE"
    echo "이 스크립트와 동일한 디렉토리에 promtail-config.yaml 파일이 있어야 합니다."
    exit 1
fi

# 1. 모니터링 서버 대상 인스턴스 이름 결정
TARGET_INSTANCE_NAME="devths-v2-nonprod-monitoring" # 기본값

if [[ "$DEPLOYMENT_GROUP_NAME" == *"Prod"* ]] || [[ "$DEPLOY_BRANCH" == "main" ]]; then
    TARGET_INSTANCE_NAME="devths-v2-prod-monitoring"
elif [[ "$DEPLOYMENT_GROUP_NAME" == *"Stg"* ]] || [[ "$DEPLOY_BRANCH" == *"release"* ]]; then
    TARGET_INSTANCE_NAME="devths-v2-nonprod-monitoring"
fi

echo "🔍 AWS EC2 리소스에서 모니터링 서버($TARGET_INSTANCE_NAME)의 Private IP를 조회합니다..."

# AWS CLI 설치 여부 확인
if ! command -v aws &> /dev/null; then
    echo "❌ AWS CLI가 설치되어 있지 않습니다. 모니터링 서버 IP를 조회할 수 없습니다."
else
    # 상태가 'running'인 지정된 이름의 인스턴스 프라이빗 IP 추출
    MONITORING_PRIVATE_IP=$(aws ec2 describe-instances \
        --region ap-northeast-2 \
        --filters "Name=tag:Name,Values=$TARGET_INSTANCE_NAME" "Name=instance-state-name,Values=running" \
        --query "Reservations[*].Instances[*].PrivateIpAddress" \
        --output text)

    if [ -n "$MONITORING_PRIVATE_IP" ]; then
        # 공백이나 멀티 값이 나올 수 있으므로 첫번째 값만 추출
        MONITORING_PRIVATE_IP=$(echo "$MONITORING_PRIVATE_IP" | awk '{print $1}')
        export LOKI_URL="http://${MONITORING_PRIVATE_IP}:3100/loki/api/v1/push"
        echo "✅ 모니터링 서버 IP 획득 성공: $MONITORING_PRIVATE_IP"
    else
        echo "⚠️  실행 중인 모니터링 서버($TARGET_INSTANCE_NAME)를 찾을 수 없거나 Private IP를 조회하지 못했습니다."
    fi
fi

# 조회 실패 시 Fallback 로컬 .env 시도
if [ -z "$LOKI_URL" ]; then
    if [ -f "$CURRENT_DIR/.env" ]; then
        echo "📄 Fallback: .env 파일에서 LOKI_URL 환경변수를 로드합니다..."
        set -a
        source "$CURRENT_DIR/.env"
        set +a
    fi
    
    if [ -z "$LOKI_URL" ]; then
        echo "❌ LOKI_URL 마저 설정되지 않아 실행을 중지합니다."
        exit 1
    fi
fi

echo "🔗 설정된 Loki 전송 주소: $LOKI_URL"


# 2. 컨테이너 실행
echo "🚀 기존의 ${CONTAINER_NAME} 컨테이너가 있다면 중지 및 삭제합니다..."
docker stop $CONTAINER_NAME 2>/dev/null
docker rm $CONTAINER_NAME 2>/dev/null

echo "📦 새로운 $CONTAINER_NAME 컨테이너를 실행합니다..."

# 핵심: 
# 1. -e LOKI_URL 로 컨테이너 내부에 환경변수를 전달합니다.
# 2. -config.expand-env=true 플래그를 주어 YAML 속 ${LOKI_URL} 이 치환되게 합니다.
docker run -d \
    --name $CONTAINER_NAME \
    --restart unless-stopped \
    -e LOKI_URL="$LOKI_URL" \
    -v ai_promtail_data:/promtail_data \
    -v ${CONFIG_FILE}:/etc/promtail/config.yml:ro \
    -v /var/lib/docker/containers:/var/lib/docker/containers:ro \
    -v /var/run/docker.sock:/var/run/docker.sock:ro \
    -v /var/log:/var/log:ro \
    -p 9080:9080 \
    grafana/promtail:3.2.1 \
    -config.file=/etc/promtail/config.yml \
    -config.expand-env=true

if [ $? -eq 0 ]; then
    echo "✅ Promtail 컨테이너가 성공적으로 실행되었습니다!"
    echo "   - 실행 상태 확인: docker ps | grep $CONTAINER_NAME"
    echo "   - 로그 확인: docker logs -f $CONTAINER_NAME"
else
    echo "❌ 컨테이너 실행에 실패했습니다. (Docker 권한 문제 등 확인 필요)"
    exit 1
fi
