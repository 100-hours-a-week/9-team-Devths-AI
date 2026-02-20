# EasyOCR VM (L4 GPU) - Docker + NVIDIA Container Toolkit
# 옵션: 시작 시 GitHub에서 코드 받아서 docker build & run (도커파일 빌드 방식)

locals {
  subnet = var.subnet_self_link != "" ? var.subnet_self_link : "projects/${var.project_id}/regions/${var.region}/subnetworks/default"

  script_docker_nvidia_only = <<-EOT
    #!/bin/bash
    set -e
    export DEBIAN_FRONTEND=noninteractive
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker ubuntu 2>/dev/null || usermod -aG docker $(logname) 2>/dev/null || true
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker
    echo "Docker and NVIDIA Container Toolkit ready. Run EasyOCR manually."
  EOT

  script_docker_nvidia_and_easyocr = <<-EOT
    #!/bin/bash
    set -e
    export DEBIAN_FRONTEND=noninteractive

    # 1) Docker + NVIDIA Container Toolkit
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker ubuntu 2>/dev/null || usermod -aG docker $(logname) 2>/dev/null || true
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker

    # 2) EasyOCR 코드 받아서 빌드 & 실행 (도커파일 빌드)
    cd /tmp
    wget -q -O docker-v2-setup.zip "${var.easyocr_repo_zip_url}"
    unzip -o docker-v2-setup.zip
    cd 9-team-Devths-AI-chore-docker-v2-setup/easyocr_server
    docker build -t easyocr-server:gpu .
    docker run -d --gpus all -p 8000:8000 --restart unless-stopped --name easyocr-server easyocr-server:gpu
    echo "EasyOCR server started on port 8000."
  EOT

  startup_script = var.install_docker_nvidia ? (var.startup_fetch_and_run_easyocr ? local.script_docker_nvidia_and_easyocr : local.script_docker_nvidia_only) : null
}

resource "google_compute_instance" "easyocr" {
  project      = var.project_id
  zone         = var.zone
  name         = var.instance_name
  machine_type = var.machine_type
  labels       = var.labels

  boot_disk {
    auto_delete  = true
    device_name  = var.instance_name
    initialize_params {
      image = var.boot_disk_image
      size  = var.boot_disk_size_gb
      type  = "pd-balanced"
    }
    mode = "READ_WRITE"
  }

  guest_accelerator {
    type  = "nvidia-l4"
    count = 1
  }

  network_interface {
    subnetwork = local.subnet
    access_config {
      network_tier = "PREMIUM"
    }
  }

  scheduling {
    automatic_restart   = true
    on_host_maintenance = "TERMINATE"
    preemptible         = false
    provisioning_model  = "STANDARD"
  }

  metadata = {
    enable-osconfig = "TRUE"
  }

  metadata_startup_script = local.startup_script
}

output "easyocr_name" {
  value = google_compute_instance.easyocr.name
}

output "easyocr_zone" {
  value = google_compute_instance.easyocr.zone
}

output "easyocr_network_ip" {
  value = google_compute_instance.easyocr.network_interface[0].network_ip
}

output "easyocr_access_config" {
  value       = try(google_compute_instance.easyocr.network_interface[0].access_config[0], null)
  description = "외부 IP 등 (SSH / 헬스체크용)"
}
