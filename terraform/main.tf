resource "google_cloud_run_v2_service" "bps_classifier" {
  name     = var.service_name
  location = var.region
  ingress  = "INGRESS_TRAFFIC_ALL"

  deletion_protection = var.deletion_protection

  invoker_iam_disabled = true

  lifecycle {
    ignore_changes = [
        client,
        scaling,
        template[0].labels,
        template[0].containers[0].name,
        template[0].containers[0].resources[0].cpu_idle,
      ]
    }

  template {
    service_account                  = var.service_account_email
    timeout                          = var.timeout
    max_instance_request_concurrency = var.container_concurrency

    scaling {
      min_instance_count = var.min_instance_count
      max_instance_count = var.max_instance_count
    }

    containers {
      image = var.container_image

      ports {
        container_port = var.container_port
      }

      resources {
        startup_cpu_boost = true

        limits = {
          cpu    = var.cpu_limit
          memory = var.memory_limit
        }
      }

      startup_probe {
        timeout_seconds   = 240
        period_seconds    = 240
        failure_threshold = 1

        tcp_socket {
          port = var.container_port
        }
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}