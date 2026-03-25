variable "project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "region" {
  description = "Cloud Run region"
  type        = string
  default     = "us-central1"
}

variable "service_name" {
  description = "Cloud Run service name"
  type        = string
  default     = "bps-classifier"
}

variable "container_image" {
  description = "Artifact Registry image URL"
  type        = string
  default     = "us-central1-docker.pkg.dev/stoked-monitor-375412/cloud-run-source-deploy/bps-img-classifier-cloudrun/bps-classifier:d62c20c1e063660ef6e0e9d372353d1a13d06f1d"
}

variable "service_account_email" {
  description = "Service account used by Cloud Run"
  type        = string
  default     = "bps-svc-acct@stoked-monitor-375412.iam.gserviceaccount.com"
}

variable "min_instance_count" {
  type    = number
  default = 1
}

variable "max_instance_count" {
  type    = number
  default = 20
}

variable "container_concurrency" {
  type    = number
  default = 80
}

variable "timeout" {
  description = "Request timeout"
  type        = string
  default     = "300s"
}

variable "container_port" {
  type    = number
  default = 8080
}

variable "cpu_limit" {
  type    = string
  default = "1000m"
}

variable "memory_limit" {
  type    = string
  default = "2Gi"
}

variable "deletion_protection" {
  description = "Set true in production"
  type        = bool
  default     = true
}

variable "allow_public_access" {
  description = "Whether to allow unauthenticated public access"
  type        = bool
  default     = true
}