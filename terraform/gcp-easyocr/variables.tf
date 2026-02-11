# GCP EasyOCR VM (L4 GPU) - Terraform variables

variable "project_id" {
  description = "GCP 프로젝트 ID"
  type        = string
}

variable "zone" {
  description = "GCP 존"
  type        = string
  default     = "asia-northeast3-a"
}

variable "region" {
  description = "GCP 리전"
  type        = string
  default     = "asia-northeast3"
}

variable "instance_name" {
  description = "VM 인스턴스 이름"
  type        = string
  default     = "easyocr"
}

variable "machine_type" {
  description = "머신 타입 (L4 1장: g2-standard-4)"
  type        = string
  default     = "g2-standard-4"
}

variable "boot_disk_size_gb" {
  description = "부팅 디스크 크기(GB)"
  type        = number
  default     = 100
}

variable "boot_disk_image" {
  description = "부팅 디스크 이미지 (CUDA 포함)"
  type        = string
  default     = "projects/ml-images/global/images/c0-deeplearning-common-cu124-v20250325-debian-11-py310-conda"
}

variable "subnet_self_link" {
  description = "서브넷 self_link (비우면 default)"
  type        = string
  default     = ""
}

variable "install_docker_nvidia" {
  description = "시작 시 Docker + NVIDIA Container Toolkit 설치 여부"
  type        = bool
  default     = true
}

variable "startup_fetch_and_run_easyocr" {
  description = "시작 시 GitHub에서 코드 받아서 docker build & run 까지 수행"
  type        = bool
  default     = true
}

variable "easyocr_repo_zip_url" {
  description = "EasyOCR 소스 zip URL (chore/docker-v2-setup 브랜치)"
  type        = string
  default     = "https://github.com/100-hours-a-week/9-team-Devths-AI/archive/refs/heads/chore/docker-v2-setup.zip"
}

variable "labels" {
  description = "VM 라벨"
  type        = map(string)
  default = {
    purpose = "easyocr-gpu"
  }
}
