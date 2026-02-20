# Exaone 8B VM (L4 GPU) - Docker + NVIDIA Container Toolkit 설치
# 컨테이너 실행은 SSH 접속 후 수동 (HUGGING_FACE_HUB_TOKEN 필요)

locals {
  subnet = var.subnet_self_link != "" ? var.subnet_self_link : "projects/${var.project_id}/regions/${var.region}/subnetworks/default"
  startup_script = <<-EOT
    #!/bin/bash
    set -e
    export DEBIAN_FRONTEND=noninteractive

    # Docker 설치
    curl -fsSL https://get.docker.com | sh
    usermod -aG docker ubuntu 2>/dev/null || usermod -aG docker $(logname) 2>/dev/null || true

    # NVIDIA Container Toolkit
    distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
    curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | apt-key add -
    curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update && apt-get install -y nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
    systemctl restart docker

    echo "Docker and NVIDIA Container Toolkit ready. Run Exaone 8B manually with HUGGING_FACE_HUB_TOKEN."
  EOT
}

resource "google_compute_instance" "exaone8b" {
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

  metadata_startup_script = var.install_docker_nvidia ? local.startup_script : null
}

output "exaone8b_name" {
  value = google_compute_instance.exaone8b.name
}

output "exaone8b_zone" {
  value = google_compute_instance.exaone8b.zone
}

output "exaone8b_network_ip" {
  value = google_compute_instance.exaone8b.network_interface[0].network_ip
}

output "exaone8b_access_config" {
  value       = try(google_compute_instance.exaone8b.network_interface[0].access_config[0], null)
  description = "외부 IP 등 access_config (SSH 접속용)"
}
