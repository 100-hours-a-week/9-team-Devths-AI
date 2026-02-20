# GCP Exaone 8B VM (vLLM) - Terraform variables
# Terraform 4.25.0+ / Google provider 호환

variable "project_id" {
  description = "GCP 프로젝트 ID"
  type        = string
}

variable "zone" {
  description = "GCP 존 (예: asia-northeast3-a)"
  type        = string
  default     = "asia-northeast3-a"
}

variable "region" {
  description = "GCP 리전 (서브넷 등에 사용)"
  type        = string
  default     = "asia-northeast3"
}

variable "instance_name" {
  description = "VM 인스턴스 이름"
  type        = string
  default     = "exaone8b"
}

variable "machine_type" {
  description = "머신 타입 (L4 1장 권장: g2-standard-4)"
  type        = string
  default     = "g2-standard-4"
}

variable "boot_disk_size_gb" {
  description = "부팅 디스크 크기(GB)"
  type        = number
  default     = 256
}

variable "boot_disk_image" {
  description = "부팅 디스크 이미지 (CUDA 포함 Deep Learning 이미지 권장)"
  type        = string
  default     = "projects/ml-images/global/images/c0-deeplearning-common-cu124-v20250325-debian-11-py310-conda"
}

variable "subnet_self_link" {
  description = "서브넷 self_link (비우면 default 사용)"
  type        = string
  default     = ""
}

variable "install_docker_nvidia" {
  description = "시작 시 Docker + NVIDIA Container Toolkit 설치 스크립트 실행 여부"
  type        = bool
  default     = true
}

variable "labels" {
  description = "VM 라벨"
  type        = map(string)
  default = {
    purpose = "exaone-8b-vllm"
  }
}
