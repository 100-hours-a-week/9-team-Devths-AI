# GCP AI GPU VM (Terraform)

v2 인프라 설계 기준 **Exaone 8B**·**EasyOCR** 전용 GCP VM을 프로비저닝합니다.  
각 VM은 L4 GPU 1장, Docker + NVIDIA Container Toolkit이 설치됩니다.

## 디렉터리

| 디렉터리 | 용도 |
|----------|------|
| `gcp-exaone8b/` | Exaone 8B (vLLM) VM — Docker+NVIDIA만 설치, 컨테이너는 SSH 후 수동 실행 (HF 토큰 필요) |
| `gcp-easyocr/`  | EasyOCR VM — Docker+NVIDIA 설치, 옵션으로 시작 시 GitHub에서 코드 받아 빌드·실행 |

## 공통 사용법

```bash
cd terraform/gcp-exaone8b   # 또는 gcp-easyocr
# terraform.tfvars 가 없으면: cp terraform.tfvars.example terraform.tfvars 후 project_id 수정

terraform init
terraform plan
terraform apply
```

## 기존 VM이 이미 있을 때 (Import)

콘솔에서 이미 만든 `easyocr` / `exaone8b` VM을 이 Terraform이 관리하도록 넣을 때 사용합니다.  
**apply 하지 말고** 아래만 실행하세요.

**EasyOCR:**

```bash
cd terraform/gcp-easyocr
terraform init
terraform import google_compute_instance.easyocr projects/gen-lang-client-0124970927/zones/asia-northeast3-a/instances/easyocr
terraform plan   # No changes 또는 변경분만 나오면 성공
```

**Exaone 8B:**

```bash
cd terraform/gcp-exaone8b
terraform init
terraform import google_compute_instance.exaone8b projects/gen-lang-client-0124970927/zones/asia-northeast3-a/instances/exaone8b
terraform plan   # No changes 또는 변경분만 나오면 성공
```

- 프로젝트/존이 다르면 위 `projects/.../zones/.../instances/이름` 부분을 실제 값으로 바꾸면 됩니다.

## Exaone 8B

- VM 기동 후 SSH 접속하여 아래 실행 (Hugging Face gated 모델용 토큰 필요).

```bash
export HUGGING_FACE_HUB_TOKEN="hf_..."

docker run -d --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HUGGING_FACE_HUB_TOKEN \
  -p 8000:8000 \
  --ipc=host \
  --name exaone-8b \
  vllm/vllm-openai:latest \
  vllm serve LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct \
  --host 0.0.0.0 \
  --port 8000
```

- 모델: `EXAONE-3.5-7.8B-Instruct` 사용 시 위 모델명만 교체하면 됨.

## EasyOCR

- `startup_fetch_and_run_easyocr = true` (기본): VM 기동 시 GitHub에서 zip 다운로드 후 `docker build` & `docker run` 자동 수행.
- `startup_fetch_and_run_easyocr = false`: Docker+NVIDIA만 설치, SSH 접속 후 수동으로 코드 받아서 빌드·실행.

수동 실행 예시:

```bash
wget https://github.com/100-hours-a-week/9-team-Devths-AI/archive/refs/heads/chore/docker-v2-setup.zip
unzip docker-v2-setup.zip
cd 9-team-Devths-AI-chore-docker-v2-setup/easyocr_server/
docker build -t easyocr-server:gpu .
docker run -d --gpus all -p 8000:8000 --name easyocr-server easyocr-server:gpu
```

확인 (이미지 파일로 테스트):

```bash
curl http://localhost:8000/health
curl -X POST -F "file=@./이미지.png" http://localhost:8000/ocr/extract
```

## 요구사항

- Terraform >= 1.0
- Google Provider ~> 5.0
- GCP 프로젝트에 Compute Engine API 활성화, L4 GPU 할당량

## 참고 문서

- [AI] v2_인프라_설계.md
- [AI] 인프라_설계_클라우드_비용_최적화.md
