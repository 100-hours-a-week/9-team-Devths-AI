# Python 버전 관리 가이드

## 현재 설정

- **Python 버전**: 3.10.19
- **패키지 관리**: pip + conda
- **가상환경**: 프로젝트별 독립 환경 권장

## 권장 설정 (프로덕션)

### 옵션 1: Python 3.10 (안정성 우선) ⭐ 추천

```bash
# Backend용 환경
conda create -n env_python310 python=3.10.19 -y
conda activate env_python310
cd 1.Wedding_OS_Project/2.Wedding_OS_back
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8101 --reload

# Model용 환경
conda create -n env_python310_model python=3.10.19 -y
conda activate env_python310_model
cd 1.Wedding_OS_Project/3.Wedding_OS_model
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8102 --reload
```

**장점**:
- 대부분의 라이브러리 완벽 지원
- Docker 이미지 풍부 (python:3.10-slim)
- 배포 환경 안정적
- TensorFlow/Keras 완전 지원
- 현재 프로젝트에서 검증된 버전

### 옵션 2: Python 3.11 (대안)

```bash
# Python 3.11 사용 시
conda create -n env_python311_model python=3.11 -y
conda activate env_python311_model
cd 1.Wedding_OS_Project/3.Wedding_OS_model
pip install -r requirements.txt
python -m uvicorn app.main:app --host 0.0.0.0 --port 8102 --reload
```

**주의사항**:
- 일부 패키지 버전 호환성 확인 필요
- 프로젝트에서 테스트 필요

## Docker 배포 시

### Dockerfile 예시 (Python 3.10) ⭐ 추천

```dockerfile
FROM python:3.10.19-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8102"]
```

### Dockerfile 예시 (Python 3.11)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8102"]
```

## 버전 확인 방법

```bash
# Python 버전 확인
python --version

# 설치된 패키지 확인
pip list

# 가상환경 목록
conda env list
```

## 문제 발생 시

### 패키지 호환성 문제
```bash
# 특정 버전으로 재설치
pip install 패키지명==버전

# 캐시 삭제 후 재설치
pip install --no-cache-dir -r requirements.txt
```

### 가상환경 재생성
```bash
conda deactivate
conda remove -n env_python310_model --all
conda create -n env_python310_model python=3.10.19 -y
conda activate env_python310_model
cd 1.Wedding_OS_Project/3.Wedding_OS_model
pip install -r requirements.txt
```

## 협업 시 권장사항

1. **Python 버전 통일**: 팀원 모두 같은 버전 사용 (3.10.19 권장) ⭐
2. **requirements.txt 항상 최신화**: `pip freeze > requirements.txt`
3. **가상환경 필수 사용**: base 환경 사용 금지
4. **`.python-version` 파일 커밋**: pyenv 등에서 자동 인식
5. **Docker 사용 권장**: 환경 차이 최소화

## 현재 프로젝트 환경 확인

```bash
# Python 버전 확인
python --version
# Python 3.10.19

# 현재 활성화된 conda 환경 확인
conda env list
# * 표시가 있는 환경이 현재 활성화된 환경

# 가상환경 활성화
conda activate env_python310_model
```

