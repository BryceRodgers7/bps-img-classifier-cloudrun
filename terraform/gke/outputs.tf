output "cluster_name" {
  value = google_container_cluster.classifier_cluster.name
}

output "cluster_location" {
  value = google_container_cluster.classifier_cluster.location
}

output "node_pool_name" {
  value = google_container_node_pool.primary_nodes.name
}