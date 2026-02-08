#!/bin/bash

# 테스트 환경 정보 수집 스크립트
# GCP L4 GPU 환경 정보 수집

echo "=================================="
echo "테스트 환경 정보 수집"
echo "수집 시간: $(date)"
echo "=================================="
echo ""

# 1. GPU 정보
echo "========== 1. GPU 정보 =========="
echo ""
echo "--- nvidia-smi 기본 정보 ---"
nvidia-smi
echo ""

echo "--- GPU 상세 정보 ---"
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.free,memory.used,compute_cap --format=csv
echo ""

echo "--- GPU 프로세스 정보 ---"
nvidia-smi pmon -c 1
echo ""

# 2. CPU 정보
echo "========== 2. CPU 정보 =========="
echo ""
echo "--- CPU 모델 및 코어 수 ---"
lscpu | grep -E "Model name|CPU\(s\)|Thread|Core|Socket"
echo ""

echo "--- CPU 상세 정보 ---"
cat /proc/cpuinfo | grep -E "model name|cpu MHz|cache size" | head -n 10
echo ""

# 3. 메모리 정보
echo "========== 3. 메모리 정보 =========="
echo ""
echo "--- 메모리 용량 ---"
free -h
echo ""

echo "--- 메모리 상세 정보 ---"
cat /proc/meminfo | grep -E "MemTotal|MemFree|MemAvailable"
echo ""

# 4. 디스크 정보
echo "========== 4. 디스크 정보 =========="
echo ""
df -h
echo ""

# 5. OS 정보
echo "========== 5. OS 정보 =========="
echo ""
echo "--- OS 버전 ---"
cat /etc/os-release
echo ""

echo "--- 커널 버전 ---"
uname -a
echo ""

# 6. Python 환경
echo "========== 6. Python 환경 =========="
echo ""
echo "--- Python 버전 ---"
python3 --version
python --version 2>/dev/null || echo "python (2.x) not found"
echo ""

echo "--- pip 버전 ---"
pip3 --version
echo ""

echo "--- 주요 패키지 버전 ---"
pip3 list | grep -E "torch|vllm|transformers|fastapi|langchain"
echo ""

# 7. Docker 정보
echo "========== 7. Docker 정보 =========="
echo ""
echo "--- Docker 버전 ---"
docker --version
echo ""

echo "--- 실행 중인 컨테이너 ---"
docker ps
echo ""

echo "--- Docker 이미지 ---"
docker images | grep vllm
echo ""

# 8. CUDA 정보
echo "========== 8. CUDA 정보 =========="
echo ""
echo "--- CUDA 버전 ---"
nvcc --version 2>/dev/null || echo "nvcc not found in PATH"
echo ""

echo "--- CUDA 라이브러리 경로 ---"
ls -la /usr/local/ | grep cuda
echo ""

# 9. PCI 장치 정보 (GPU 확인)
echo "========== 9. PCI 장치 정보 =========="
echo ""
lspci | grep -i nvidia
echo ""

# 10. 네트워크 정보
echo "========== 10. 네트워크 정보 =========="
echo ""
echo "--- 네트워크 인터페이스 ---"
ip addr show | grep -E "inet |UP"
echo ""

# 11. GCP 인스턴스 메타데이터 (GCP 환경인 경우)
echo "========== 11. GCP 인스턴스 정보 =========="
echo ""
echo "--- 인스턴스 타입 ---"
curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/machine-type 2>/dev/null | awk -F'/' '{print $NF}' || echo "Not GCP instance"
echo ""

echo "--- 인스턴스 존 ---"
curl -s -H "Metadata-Flavor: Google" http://metadata.google.internal/computeMetadata/v1/instance/zone 2>/dev/null | awk -F'/' '{print $NF}' || echo "Not GCP instance"
echo ""

# 12. vLLM 컨테이너 로그 (실행 중인 경우)
echo "========== 12. vLLM 컨테이너 정보 =========="
echo ""
CONTAINER_NAME=$(docker ps --filter "ancestor=vllm/vllm-openai:latest" --format "{{.Names}}" | head -n 1)
if [ -n "$CONTAINER_NAME" ]; then
    echo "--- 컨테이너 이름: $CONTAINER_NAME ---"
    echo ""
    echo "--- 컨테이너 상태 ---"
    docker inspect $CONTAINER_NAME | grep -A 5 "State"
    echo ""
    echo "--- 컨테이너 리소스 사용량 ---"
    docker stats --no-stream $CONTAINER_NAME
    echo ""
else
    echo "실행 중인 vLLM 컨테이너가 없습니다."
fi
echo ""

echo "=================================="
echo "정보 수집 완료!"
echo "=================================="
