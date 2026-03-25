output "service_name" {
  value = google_cloud_run_v2_service.bps_classifier.name
}

output "service_uri" {
  value = google_cloud_run_v2_service.bps_classifier.uri
}