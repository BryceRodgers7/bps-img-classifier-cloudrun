## Kubernetes Deployment

This project uses Kubernetes to run and scale the classifier API. The configuration is structured using Kustomize to separate environment-agnostic resources from environment-specific overrides.

### Structure

```
k8s/
  base/
    deployment.yaml
    service.yaml
  overlays/
    local/
    gke/
```

* **base/**
  Contains shared Kubernetes resources that are environment-independent:

  * Deployment
  * Service
  * Health probes
  * Core environment variables

* **overlays/local/**
  Local development configuration for Minikube:

  * Uses locally built Docker image (`bps-classifier-api:local`)
  * Sets `imagePullPolicy: Never`
  * Injects GCP credentials via a Kubernetes Secret
  * Mounts credentials into the container for model download from GCS

* **overlays/gke/**
  Production configuration for Google Kubernetes Engine (GKE):

  * Uses Artifact Registry image
  * Enables Horizontal Pod Autoscaling (HPA)
  * Removes local credential mounts
  * Designed to use Workload Identity instead of credential files

---

## Local Development (Minikube)

### Prerequisites

* Docker Desktop
* kubectl
* Minikube
* GCP SDK (`gcloud`) authenticated locally

### Create GCP Credentials Secret

This project requires access to a model stored in Google Cloud Storage.

Create the Kubernetes secret from your local ADC credentials:

```
kubectl create secret generic gcp-adc \
  --from-file=application_default_credentials.json="C:\Users\<your-user>\AppData\Roaming\gcloud\application_default_credentials.json"
```

> Note: This secret is **not committed to the repository** and must be created locally.

---

### Build and Load Image into Minikube

```
docker build -t bps-classifier-api:local .
minikube image load bps-classifier-api:local
```

---

### Deploy to Kubernetes

```
kubectl apply -k k8s/overlays/local
```

---

### Test the Service

```
kubectl port-forward service/classifier-api 8080:8080
```

Then in another terminal:

```
curl http://localhost:8080/health
```

```
curl -X POST "http://localhost:8080/predict" -F "file=@test-image.jpg"
```

---

## Scaling

The classifier API supports horizontal scaling:

* Local: manually set replica count in overlay
* GKE: uses Horizontal Pod Autoscaler (HPA)

Example:

```
kubectl get pods
```

Multiple pods will be created and managed automatically.

---

## Model Loading

The classifier loads a trained model from Google Cloud Storage at startup.

* Bucket: `bps-model`
* Path: `models/best_model.pth`

### Local (Minikube)

* Uses mounted ADC credentials via Kubernetes Secret
* Model is downloaded at container startup

### Production (GKE – planned)

* Will use Workload Identity (no credential files)
* Model download handled via init container
* Model stored in versioned path in GCS

---





































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
