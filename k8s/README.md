## Kubernetes Deployment

This project uses Kubernetes to run and scale the classifier API. The
configuration is structured using Kustomize to separate
environment-agnostic resources from environment-specific overrides.

### Structure

    k8s/
      base/
        deployment.yaml
        service.yaml
      overlays/
        local/
        gke/

-   **base/** Contains shared Kubernetes resources that are
    environment-independent:

    -   Deployment
    -   Service
    -   Health probes
    -   Core environment variables

-   **overlays/local/** Local development configuration for Minikube:

    -   Uses locally built Docker image (`bps-classifier-api:local`)
    -   Sets `imagePullPolicy: Never`
    -   Injects GCP credentials via a Kubernetes Secret
    -   Mounts credentials into the container for model download from
        GCS

-   **overlays/gke/** Production configuration for Google Kubernetes
    Engine (GKE):

    -   Uses Artifact Registry image
    -   Enables Horizontal Pod Autoscaling (HPA)
    -   Removes local credential mounts
    -   Uses Workload Identity to securely access GCP resources without
        credential files
    -   Exposes the service externally using GKE Ingress

------------------------------------------------------------------------

## Local Development (Minikube)

### Prerequisites

-   Docker Desktop
-   kubectl
-   Minikube
-   GCP SDK (`gcloud`) authenticated locally

------------------------------------------------------------------------

### Create GCP Credentials Secret

This project requires access to a model stored in Google Cloud Storage.

Create the Kubernetes secret from your local ADC credentials:

    kubectl create secret generic gcp-adc \
      --from-file=application_default_credentials.json="C:\Users\<your-user>\AppData\Roaming\gcloud\application_default_credentials.json"

> Note: This secret is **not committed to the repository** and must be
> created locally.

------------------------------------------------------------------------

### Build and Load Image into Minikube

    docker build -t bps-classifier-api:local .
    minikube image load bps-classifier-api:local

------------------------------------------------------------------------

### Deploy to Kubernetes

    kubectl apply -k k8s/overlays/local

------------------------------------------------------------------------

### Test the Service

    kubectl port-forward service/classifier-api 8080:8080

Then in another terminal:

    curl http://localhost:8080/health

    curl -X POST "http://localhost:8080/predict" -F "file=@test-image.jpg"

------------------------------------------------------------------------

## Google Kubernetes Engine (GKE)

### Infrastructure (Terraform)

GKE infrastructure is provisioned using Terraform:

-   GKE cluster with Workload Identity enabled
-   Dedicated node pool with configurable machine type and size
-   Google service accounts for:
    -   Node access (Artifact Registry)
    -   Workload identity (GCS access)
-   IAM bindings:
    -   Kubernetes Service Account → Google Service Account (Workload
        Identity)
    -   Google Service Account → GCS bucket
        (`roles/storage.objectViewer`)

------------------------------------------------------------------------

### Deploy to GKE

    kubectl apply -k k8s/overlays/gke

------------------------------------------------------------------------

### Verify Deployment

    kubectl get pods
    kubectl get deployment
    kubectl get hpa
    kubectl get ingress

------------------------------------------------------------------------

### External Access (Ingress)

The GKE deployment uses a Kubernetes Ingress resource to expose the
service publicly.

-   Ingress is defined in `k8s/overlays/gke/ingress.yaml`
-   GKE automatically provisions an external HTTP load balancer
-   A public IP address is assigned to the Ingress

To retrieve the external IP:

    kubectl get ingress

Example output:

    NAME                     ADDRESS
    classifier-api-ingress   34.x.x.x

------------------------------------------------------------------------

### Test via Public Endpoint

    curl http://<EXTERNAL_IP>/health

    curl -X POST "http://<EXTERNAL_IP>/predict" -F "file=@test-image.jpg"

------------------------------------------------------------------------

### Test via Port Forwarding (Optional)

    kubectl port-forward deployment/classifier-api 8080:8080

Then:

    curl http://localhost:8080/health

------------------------------------------------------------------------

## Authentication (Workload Identity)

This deployment does **not** use credential files.

Instead:

1.  A Kubernetes Service Account (`classifier-api-sa`) is created
2.  It is annotated with a Google Service Account
3.  IAM binding allows the KSA to impersonate the GSA
4.  The pod runs using that KSA

This enables secure, keyless access to GCP services such as Cloud
Storage.

------------------------------------------------------------------------

## Model Loading

The classifier loads a trained model from Google Cloud Storage at
startup.

-   Bucket: `bps-model`
-   Path: `models/best_model.pth`

### Local (Minikube)

-   Uses mounted ADC credentials via Kubernetes Secret
-   Model is downloaded at container startup

### Production (GKE)

-   Uses Workload Identity (no credential files)
-   Kubernetes Service Account is mapped to a Google Service Account
-   Google Service Account has `roles/storage.objectViewer` on the model
    bucket
-   Model is downloaded securely from GCS at container startup

Authentication flow:

    Pod → Kubernetes Service Account → Workload Identity → Google Service Account → GCS

------------------------------------------------------------------------

## Scaling

The classifier API supports horizontal scaling:

-   Local: manually set replica count in overlay
-   GKE: uses Horizontal Pod Autoscaler (HPA)

Example:

    kubectl get pods

Multiple pods will be created and managed automatically.

------------------------------------------------------------------------

## Architecture Summary

    Frontend
       │
       ├── Cloud Run (current production)
       │
       └── GKE (Kubernetes deployment)
             │
             ├── Ingress (external load balancer)
             │
             ├── Service (internal networking)
             │
             ├── Deployment (classifier-api)
             │     └── Pods (replicated)
             │
             ├── HPA (auto-scaling)
             │
             └── Workload Identity
                   └── GCS (model storage)

------------------------------------------------------------------------

## Deployment Tradeoffs

This project includes both Cloud Run and GKE deployments of the same
API.

### Cloud Run

-   Fully managed, serverless
-   Scales to zero
-   Lower operational overhead
-   Cost-efficient for low or bursty traffic

### GKE (Standard)

-   Full Kubernetes control
-   Supports complex workloads and scaling patterns
-   Requires always-on compute (nodes)
-   Higher operational overhead and cost

For this project, Cloud Run is used as the primary serving layer, while
GKE is implemented for learning, flexibility, and demonstrating
Kubernetes-based deployment patterns.