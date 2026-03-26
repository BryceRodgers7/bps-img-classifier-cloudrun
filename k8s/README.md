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
