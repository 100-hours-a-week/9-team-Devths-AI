# Exaone vLLM 전용 서버 구축 메뉴얼

다른 서버에 Exaone(vLLM)만 올릴 때 따라 할 수 있는 단계별 가이드입니다.  
**전체 FastAPI 앱(RAG, OCR, Presidio 등)을 돌리는 서버가 아닌, Exaone 추론 전용 서버**를 세팅할 때 사용하세요.

---

## 1. 서버 용도 구분

| 용도 | 설치할 것 | 비고 |
|------|------------|------|
| **Exaone만 서빙** (이 메뉴얼) | `vllm_requirements.txt` 또는 Docker `vllm/vllm-openai` 또는 **Poetry 환경 + vllm** | GPU 필요, 전체 `requirements.txt` **사용하지 않음** |
| 전체 모델 서비스(FastAPI + RAG + OCR 등) | `requirements.txt` 또는 `requirements-serving.txt` | 별도 문서 참고 |

Exaone 전용 서버에서는 **`requirements.txt`를 설치하지 마세요.** 의존성 충돌·디스크 사용이 커질 수 있습니다.

---

## 2. 사전 요구사항

- **OS**: Amazon Linux, Ubuntu 등 (GCP EC2, AWS EC2 등)
- **Python**: 3.10 이상 (venv/Poetry 사용 시)
- **GPU**: NVIDIA GPU + 드라이버 + CUDA (vLLM GPU 모드용)
- **Docker 사용 시**: Docker + NVIDIA Container Toolkit 설치됨
- **Poetry 사용 시**: [Poetry 설치](https://python-poetry.org/docs/#installation) 완료
- **모델**: Hugging Face gated 모델 사용 시 `HUGGING_FACE_HUB_TOKEN` 필요

---

## 3. 방법 A: Docker로 vLLM 실행 (권장)

다른 서버에서도 그대로 쓸 수 있는 방식입니다. 레포의 스크립트를 그대로 쓰거나, 아래 명령만 복사해 사용하면 됩니다.

### 3.1 레포 클론 및 이동

```bash
git clone https://github.com/100-hours-a-week/9-team-Devths-AI-develop.git
cd 9-team-Devths-AI-develop/3.model
```

### 3.2 스크립트로 실행 (Exaone 8B)

```bash
export MODEL_NAME="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
export PORT=8000
export HUGGING_FACE_HUB_TOKEN="hf_xxxx"   # gated 모델일 때

./scripts/start_vllm_gcp.sh
```

### 3.3 Docker만 직접 실행 (스크립트 없이)

```bash
export HUGGING_FACE_HUB_TOKEN="hf_xxxx"   # gated 모델일 때만

docker run -d \
  --name exaone-8b \
  --gpus all \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN \
  --ipc=host \
  vllm/vllm-openai:latest \
  vllm serve LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --trust-remote-code
```

Exaone 32B를 별도 서버에 올릴 때는 `--name`과 모델명을 `EXAONE-3.5-32B-Instruct`로 바꾸고 포트(예: 8001)만 다르게 하면 됩니다.

### 3.4 동작 확인

```bash
curl -s http://localhost:8000/v1/models | jq .
```

---

## 4. 방법 B: venv + vllm_requirements.txt (Docker 미사용)

Poetry 없이, 해당 서버에서 vLLM만 pip으로 실행할 때 사용합니다.

### 4.1 레포 클론 및 이동

```bash
cd /home/ssm-user   # 또는 실제 작업 디렉터리
git clone https://github.com/100-hours-a-week/9-team-Devths-AI-develop.git
cd 9-team-Devths-AI-develop/3.model
```

### 4.2 가상환경 생성 및 활성화

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 4.3 vLLM 전용 패키지만 설치

**반드시 `vllm_requirements.txt`만 사용하세요.**

```bash
pip install --upgrade pip
pip install -r vllm_requirements.txt
```

### 4.4 vLLM 서버 실행

```bash
export HUGGING_FACE_HUB_TOKEN="hf_xxxx"   # gated 모델이면

vllm serve LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --trust-remote-code
```

백그라운드 실행은 `nohup ... &` 또는 `screen`/`tmux` 사용을 권장합니다.

### 4.5 동작 확인

```bash
curl -s http://localhost:8000/v1/models | jq .
```

---

## 5. 방법 C: Poetry 환경에서 vLLM(Exaone) 실행

이미 Poetry를 쓰는 서버에서, Poetry가 만든 가상환경 안에서 vLLM만 실행하고 싶을 때 사용합니다.  
(`pyproject.toml`에는 vLLM이 없으므로, Poetry 환경에 vLLM을 pip으로 추가한 뒤 실행합니다.)

### 5.1 레포 클론 및 이동

```bash
cd /home/ssm-user   # 또는 실제 작업 디렉터리
git clone https://github.com/100-hours-a-week/9-team-Devths-AI-develop.git
cd 9-team-Devths-AI-develop/3.model
```

### 5.2 Poetry 가상환경 준비

```bash
# 가상환경을 프로젝트 내부에 생성 (선택, 이미 있으면 생략)
poetry config virtualenvs.in-project true
poetry env use python3
```

### 5.3 Poetry 셸 진입 후 vLLM 전용 설치

**전체 `poetry install`은 하지 마세요.** Exaone 전용 서버에서는 vLLM만 필요합니다.

```bash
poetry shell
pip install --upgrade pip
pip install -r vllm_requirements.txt
```

### 5.4 vLLM 서버 실행 (Poetry 환경 안에서)

```bash
export HUGGING_FACE_HUB_TOKEN="hf_xxxx"   # gated 모델이면

vllm serve LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  --trust-remote-code
```

셸을 닫아도 계속 돌리려면 `nohup` 또는 `poetry run`으로 실행할 수 있습니다.

```bash
# poetry run으로 실행 (poetry shell 없이)
export HUGGING_FACE_HUB_TOKEN="hf_xxxx"
poetry run vllm serve LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct \
  --host 0.0.0.0 --port 8000 --max-model-len 4096 --trust-remote-code
```

(이때 `poetry run`이 동작하려면 위 5.3에서 `poetry shell` 안에서 `pip install -r vllm_requirements.txt`로 같은 Poetry 환경에 vLLM이 설치되어 있어야 합니다.)

### 5.5 동작 확인

```bash
curl -s http://localhost:8000/v1/models | jq .
```

---

## 6. 메인 앱(FastAPI) 쪽 설정

Exaone 전용 서버를 띄운 뒤, 모델 서비스(FastAPI)가 돌아가는 서버에서:

- `GCP_VLLM_BASE_URL=http://<Exaone서버IP 또는 호스트>:8000`
- (32B 별도 서버 시) `VLLM_32B_BASE_URL=http://<32B서버>:8000`

Docker Compose 예시:

```yaml
environment:
  - GCP_VLLM_BASE_URL=http://exaone-8b:8000
  - VLLM_32B_BASE_URL=http://exaone-32b:8000
```

### 6.1 메인 앱을 Poetry로 실행할 때 (다른 서버에 전체 앱 올리는 경우)

Exaone은 위 URL로 원격 연결하고, **메인 FastAPI 앱**은 Poetry로 띄우는 경우:

```bash
cd 9-team-Devths-AI-develop/3.model
poetry install   # 또는 poetry install --only main
PYTHONPATH=. poetry run python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- `start_server.sh` 참고. `GCP_VLLM_BASE_URL` 등은 해당 서버의 환경 변수 또는 `.env`에 설정합니다.

---

## 7. 다른 서버 올릴 때 체크리스트

- [ ] NVIDIA 드라이버 + CUDA 설치 확인
- [ ] (Docker 시) Docker + NVIDIA Container Toolkit 설치
- [ ] (venv 시) `vllm_requirements.txt`만 설치, `requirements.txt` 사용 금지
- [ ] gated 모델이면 `HUGGING_FACE_HUB_TOKEN` 설정
- [ ] `vllm serve LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct ...` 실행
- [ ] `curl http://localhost:8000/v1/models` 로 확인
- [ ] 메인 앱에 `GCP_VLLM_BASE_URL`(및 필요 시 `VLLM_32B_BASE_URL`) 설정
- [ ] (Poetry 사용 시) `poetry shell` 후 `pip install -r vllm_requirements.txt` 또는 메인 앱은 `poetry install` 후 `poetry run uvicorn ...`

---

## 8. 참고 파일 위치 (이 레포 기준)

| 파일 | 용도 |
|------|------|
| `3.model/vllm_requirements.txt` | Exaone(vLLM) 전용 pip 의존성 |
| `3.model/requirements.txt` | 전체 FastAPI 앱용 (Exaone 전용 서버에서는 사용하지 않음) |
| `3.model/scripts/start_vllm_gcp.sh` | Docker 기반 vLLM 서버 시작 스크립트 |
| `3.model/terraform/README.md` | GCP Exaone 8B VM + Docker 실행 요약 |
| `3.model/docker-compose.yml` | 로컬/테스트용 Exaone 8B·32B 컨테이너 |
| `3.model/app/config/settings.py` | `gcp_vllm_base_url`, `vllm_model_name` 등 |
| `3.model/pyproject.toml` | Poetry 의존성 (vLLM은 없음, 메인 앱용) |
| `3.model/start_server.sh` | Poetry로 메인 앱(uvicorn) 실행 예시 |

---

## 9. 트러블슈팅

- **Poetry로 Exaone만 돌릴 때**: 전체 `poetry install`은 하지 말고, `poetry shell` 후 `pip install -r vllm_requirements.txt`만 수행. vLLM은 `pyproject.toml`에 없습니다.
- **venv만 쓸 때**: Exaone 전용 서버는 **venv + pip**만으로도 가능. `pip install -r vllm_requirements.txt`만 수행.
- **CUDA/GPU 오류**: `nvidia-smi` 확인. Docker 시 `--gpus all` 및 NVIDIA Container Toolkit 확인.
- **gated 모델 401**: `HUGGING_FACE_HUB_TOKEN` 설정 및 Docker 시 `-e HUGGING_FACE_HUB_TOKEN` 전달 확인.
- **메인 앱에서 vLLM 미연결**: `GCP_VLLM_BASE_URL`이 Exaone 서버 IP/호스트:8000과 일치하는지, 방화벽 8000 포트 개방 여부 확인.
