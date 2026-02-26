#!/bin/bash
# =============================================================================
# ASG Launch Template User Data
# AI EC2 인스턴스 초기화 스크립트 (Blue/Green 배포용)
#
# 대상: Stg/Prod AI EC2 인스턴스 (Ubuntu 22.04 LTS)
# 역할:
#   - Docker, Docker Compose, AWS CLI, CodeDeploy Agent 설치
#   - AI 서비스 디렉토리 구조 생성
#   - CodeDeploy Agent 실행
#
# [주의] 이 스크립트는 인스턴스 최초 부팅 시 1회만 실행됩니다.
#        이후 배포는 CodeDeploy → appspec.yml → docker_deploy.sh 로 처리됩니다.
# =============================================================================

set -e
exec > /var/log/user-data.log 2>&1

echo "============================================================"
echo "🚀 AI EC2 User Data 초기화 시작 ($(date))"
echo "============================================================"

# ────────────────────────────────────────────
# 1. 시스템 패키지 업데이트
# ────────────────────────────────────────────
echo "📦 시스템 패키지 업데이트 중..."
apt-get update -y
apt-get upgrade -y
apt-get install -y \
    curl \
    wget \
    unzip \
    git \
    jq \
    ca-certificates \
    gnupg \
    lsb-release \
    ruby-full \
    python3-pip

# ────────────────────────────────────────────
# 2. Docker 설치
# ────────────────────────────────────────────
echo "🐳 Docker 설치 중..."

# Docker GPG 키 추가
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg
chmod a+r /etc/apt/keyrings/docker.gpg

# Docker 저장소 추가
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null

# Docker 설치 (compose plugin 포함)
apt-get update -y
apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# Docker 서비스 시작 및 자동 시작 설정
systemctl enable docker
systemctl start docker

# ubuntu 유저에게 Docker 권한 부여 (sudo 없이 docker 명령 사용)
usermod -aG docker ubuntu

echo "✅ Docker 설치 완료: $(docker --version)"
echo "✅ Docker Compose 설치 완료: $(docker compose version)"

# ────────────────────────────────────────────
# 3. AWS CLI v2 설치
# ────────────────────────────────────────────
echo "☁️ AWS CLI v2 설치 중..."

# 기존 v1 제거 (있을 경우)
apt-get remove -y awscli 2>/dev/null || true

curl -sSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
/tmp/aws/install
rm -rf /tmp/awscliv2.zip /tmp/aws

echo "✅ AWS CLI 설치 완료: $(aws --version)"

# ────────────────────────────────────────────
# 4. SSM Agent 설치 (Session Manager 접속용)
# ────────────────────────────────────────────
echo "🔌 SSM Agent 설치 중..."
snap install amazon-ssm-agent --classic
systemctl enable amazon-ssm-agent
systemctl start amazon-ssm-agent
echo "✅ SSM Agent 설치 완료: $(systemctl is-active amazon-ssm-agent)"

# ────────────────────────────────────────────
# 5. CodeDeploy Agent 설치
# ────────────────────────────────────────────
echo "📋 CodeDeploy Agent 설치 중..."

# AWS 리전 자동 감지 (IMDSv2 사용)
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")
AWS_REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/dynamic/instance-identity/document | jq -r '.region')

# CodeDeploy Agent 설치
wget -q "https://aws-codedeploy-${AWS_REGION}.s3.${AWS_REGION}.amazonaws.com/latest/install" \
    -O /tmp/codedeploy-install
chmod +x /tmp/codedeploy-install
/tmp/codedeploy-install auto
rm -f /tmp/codedeploy-install

# CodeDeploy Agent 서비스 시작 및 자동 시작 설정
systemctl enable codedeploy-agent
systemctl start codedeploy-agent

echo "✅ CodeDeploy Agent 설치 완료"
systemctl status codedeploy-agent --no-pager

# ────────────────────────────────────────────
# 5. AI 서비스 디렉토리 구조 생성
# ────────────────────────────────────────────
echo "📁 디렉토리 구조 생성 중..."

# CodeDeploy가 배포하는 기본 경로
AI_DIR="/home/ubuntu/ai"
mkdir -p "${AI_DIR}/logs"
mkdir -p "${AI_DIR}/deploy"

# 디렉토리 소유권 설정 (ubuntu 유저로 배포 스크립트 실행)
chown -R ubuntu:ubuntu /home/ubuntu/ai

echo "✅ 디렉토리 구조 생성 완료"
ls -la /home/ubuntu/ai

# ────────────────────────────────────────────
# 6. 시스템 설정 최적화
# ────────────────────────────────────────────
echo "⚙️ 시스템 설정 최적화 중..."

# Docker 데몬 설정 (로그 드라이버, 이미지 정리 등)
cat > /etc/docker/daemon.json << 'EOF'
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "live-restore": true
}
EOF

# live-restore: Docker 데몬 재시작 시 컨테이너 유지 (OS 레벨 예외 상황 대비)
systemctl restart docker

# 스왑 메모리 설정 (메모리 80% 점유 상황 완화)
# 2GB 스왑 추가 (0에서 메모리 부족 시 완충재 역할)
if [ ! -f /swapfile ]; then
    fallocate -l 2G /swapfile
    chmod 600 /swapfile
    mkswap /swapfile
    swapon /swapfile
    echo '/swapfile none swap sw 0 0' >> /etc/fstab
    echo "✅ 2GB 스왑 메모리 추가 완료"
fi

# ────────────────────────────────────────────
# 7. 초기화 완료 마킹
# ────────────────────────────────────────────
echo "============================================================"
echo "✅ AI EC2 초기화 완료 ($(date))"
echo "   - Docker: $(docker --version)"
echo "   - Docker Compose: $(docker compose version)"
echo "   - AWS CLI: $(aws --version)"
echo "   - CodeDeploy Agent: $(systemctl is-active codedeploy-agent)"
echo "   - 디렉토리: $(ls /home/ubuntu/ai)"
echo "============================================================"
