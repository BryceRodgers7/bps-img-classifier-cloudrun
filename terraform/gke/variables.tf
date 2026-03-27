variable "project_id" {
  type        = string
  description = "GCP project ID"
  default     = "stoked-monitor-375412"
}

variable "region" {
  type        = string
  description = "GCP region for the GKE cluster"
  default     = "us-central1"
}

variable "cluster_name" {
  type        = string
  description = "Name of the GKE cluster"
  default     = "classifier-gke-cluster"
}

variable "node_pool_name" {
  type        = string
  description = "Name of the primary node pool"
  default     = "primary-node-pool"
}

variable "node_count" {
  type        = number
  description = "Initial node count"
  default     = 1
}

variable "machine_type" {
  type        = string
  description = "GCE machine type for GKE nodes"
  default     = "e2-standard-2"
}

variable "disk_size_gb" {
  type        = number
  description = "Boot disk size in GB"
  default     = 50
}

variable "environment" {
  type        = string
  description = "Environment label"
  default     = "dev"
}

variable "artifact_registry_repo" {
  type        = string
  description = "Artifact Registry repository name"
  default     = "ml-apps"
}

variable "k8s_namespace" {
  type        = string
  description = "Kubernetes namespace for the classifier workload"
  default     = "default"
}

variable "k8s_service_account_name" {
  type        = string
  description = "Kubernetes service account name for classifier-api"
  default     = "classifier-api-sa"
}

variable "model_bucket_name" {
  type        = string
  description = "GCS bucket containing model artifacts"
  default     = "bps-model"
}