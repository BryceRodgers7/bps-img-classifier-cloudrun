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

    kubectl create secret generic gcp-adc   --from-file=application_default_credentials.json="C:\Users\<your-user>\AppData\Roaming\gcloud\application_default_credentials.json"

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

------------------------------------------------------------------------

### Test via Port Forwarding

    kubectl port-forward deployment/classifier-api 8080:8080

Then:

    curl http://localhost:8080/health

    curl -X POST "http://localhost:8080/predict" -F "file=@test-image.jpg"

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
             ├── Deployment (classifier-api)
             │     └── Pods (replicated)
             │
             ├── Service (internal networking)
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































## Architecture Overview

- Frontend: Streamlit app hosted on Fly.io
- Backend:
  - classifier-api: deployed on Kubernetes (Minikube locally)
  - gpt-api: separate service (future Kubernetes deployment)
- Model storage:
  - ResNet model stored in Google Cloud Storage
- Model generation:
  - training & data collection pipeline at https://github.com/BryceRodgers7/img-classifier-birdplanesuper
- Kubernetes:
  - Base manifests define shared deployment + service
  - Local overlay injects credentials and uses local Docker image



## Local Kubernetes Setup

Before deploying locally, create the GCP credentials secret:

kubectl create secret generic gcp-adc \
  --from-file=application_default_credentials.json="C:\Users\<your-user>\AppData\Roaming\gcloud\application_default_credentials.json"

Then deploy:

kubectl apply -k k8s/overlays/local


















# Local Kubernetes Deployment

This directory contains manifests for running the classifier locally with Kubernetes
(Docker Desktop, minikube, or kind).

## Prerequisites

- A local Kubernetes cluster (Docker Desktop k8s, minikube, or kind)
- `kubectl` configured to point at it
- A GCP service-account key JSON with read access to the `bps-model` GCS bucket

## 1 — Create the GCP credentials secret

This is a one-time step.  The secret is never committed to Git.

```bash
kubectl create secret generic gcp-credentials \
  --from-file=key.json=/path/to/your/service-account-key.json
```

The deployment mounts this secret at `/var/secrets/gcp/key.json` and sets
`GOOGLE_APPLICATION_CREDENTIALS` to that path so the GCS client picks it up
automatically.

## 2 — Build the Docker image

```bash
# From the repo root
docker build -t bps-classifier:latest .
```

> **minikube users:** run `eval $(minikube docker-env)` first so the image is
> built inside minikube's Docker daemon and is available without a registry push.

## 3 — Apply the manifests

```bash
kubectl apply -f k8s/deployment.yaml
```

This creates:
- A `PersistentVolumeClaim` (`bps-model-cache`) that stores the downloaded model
  file so pod restarts skip the GCS download.
- A `Deployment` with startup / readiness / liveness probes all wired to `/health`.
- A `LoadBalancer` Service on port 8080.

## 4 — Watch the pod start up

```bash
kubectl get pods -w
```

The pod will stay in `0/1 Running` (not ready) while the model is downloading from
GCS.  Once `/health` returns 200 the readiness probe passes and the pod flips to
`1/1 Running`.

Check logs if you want to watch the download progress:

```bash
kubectl logs -f deployment/bps-classifier
```

## 5 — Call the API

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict -F "file=@your_image.jpg"
```

## Probe summary

| Probe | Endpoint | Purpose |
|---|---|---|
| `startupProbe` | `GET /health` | Allows up to **5 minutes** (30 × 10 s) for the model to download before marking the pod failed |
| `readinessProbe` | `GET /health` | Keeps pod out of the load-balancer until the model is loaded; re-checks every 10 s |
| `livenessProbe` | `GET /health` | Restarts the pod if it becomes unhealthy after startup |
