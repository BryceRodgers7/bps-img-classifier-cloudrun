# Bird/Plane/Superman Classifier API

A FastAPI-based image classification service for identifying birds, planes, superman, and other objects. Designed for deployment on Google Cloud Run.

## Project Structure

```
.
├── main.py              # FastAPI application with /predict endpoint
├── classifier.py        # Image classifier implementation
├── models/              # Directory for downloaded model files
│   └── best_model.pth   # Trained model weights (downloaded from GCS)
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration for Cloud Run
└── .dockerignore       # Files to exclude from Docker build
```

## Model Management

The application automatically downloads the model file from Google Cloud Storage on startup:
- **Bucket**: `bps-model` (configurable via `GCS_BUCKET_NAME` env var)
- **Model file**: `best_model.pth` (configurable via `GCS_MODEL_PATH` env var)
- **Local path**: `models/best_model.pth`

The model is downloaded only if it doesn't already exist locally, making it efficient for container restarts.

## API Endpoints

### POST /predict
Upload an image for classification.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `file` field containing the image

**Supported image formats:**
- JPEG/JPG
- PNG
- GIF
- WebP

**Response:**
```json
{
  "predicted_class": "bird",
  "confidence": 0.9523,
  "probabilities": {
    "bird": 0.9523,
    "plane": 0.0321,
    "superman": 0.0098,
    "other": 0.0058
  },
  "threshold_applied": false
}
```

### GET /health
Health check endpoint for Cloud Run monitoring.

**Response:**
```json
{
  "status": "healthy"
}
```

### GET /info
Get model configuration and metadata.

**Response:**
```json
{
  "classes": ["bird", "plane", "superman", "other"],
  "confidence_threshold": 0.7,
  "device": "cpu",
  "config": {}
}
```

### GET /
API information and available endpoints.

## Local Development

### Prerequisites
- Python 3.11+
- Docker (for containerized deployment)

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Locally
```bash
python main.py
```

The API will be available at `http://localhost:8080`

### Test the API
```bash
# Using curl
curl -X POST -F "file=@path/to/image.jpg" http://localhost:8080/predict

# Using Python
import requests

with open("test_image.jpg", "rb") as f:
    response = requests.post(
        "http://localhost:8080/predict",
        files={"file": f}
    )
    print(response.json())
```

## Docker Deployment

### Build Image
```bash
docker build -t bird-classifier .
```

### Run Container
```bash
docker run -p 8080:8080 bird-classifier
```

### Test Container
```bash
curl -X POST -F "file=@test_image.jpg" http://localhost:8080/predict
```

## Google Cloud Run Deployment

### Prerequisites
- Google Cloud SDK installed and configured
- Project ID set up in Google Cloud Platform
- Cloud Run API enabled
- Model file uploaded to GCS bucket `bps-model`

### Upload Model to Google Cloud Storage

If you haven't already uploaded your model:

```bash
# Create the bucket (if it doesn't exist)
gsutil mb gs://bps-model

# Upload the model file
gsutil cp best_model.pth gs://bps-model/best_model.pth

# Verify upload
gsutil ls -lh gs://bps-model/
```

### Deploy to Cloud Run

1. **Build and push to Google Container Registry:**
```bash
gcloud builds submit --tag gcr.io/PROJECT_ID/bird-classifier
```

2. **Deploy to Cloud Run:**
```bash
gcloud run deploy bird-classifier \
  --image gcr.io/PROJECT_ID/bird-classifier \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --timeout 300s
```

**Note**: Cloud Run will automatically have access to your GCS bucket if the service account has the necessary permissions. The default Cloud Run service account typically has Storage Object Viewer role.

3. **Get the service URL:**
```bash
gcloud run services describe bird-classifier --region us-central1 --format 'value(status.url)'
```

### Environment Variables

The following environment variables can be configured:

- `PORT` - Server port (default: 8080, automatically set by Cloud Run)
- `CONFIDENCE_THRESHOLD` - Minimum confidence for predictions (default: 0.7)
- `GCS_BUCKET_NAME` - GCS bucket name for model storage (default: bps-model)
- `GCS_MODEL_PATH` - Path to model file in bucket (default: best_model.pth)

To set environment variables on Cloud Run:
```bash
gcloud run services update bird-classifier \
  --region us-central1 \
  --set-env-vars CONFIDENCE_THRESHOLD=0.8,GCS_BUCKET_NAME=my-custom-bucket
```

## Model Information

- **Architecture:** ResNet50
- **Classes:** bird, plane, superman, other
- **Input:** RGB images (automatically resized to 224x224)
- **Confidence Threshold:** 0.7 (configurable)

When the model's confidence is below the threshold, it predicts "other" as a safety mechanism.

## API Features

- **In-memory processing:** Images are processed directly from upload without disk I/O
- **File validation:** Content type and size validation (10MB max)
- **CORS enabled:** Cross-origin requests allowed
- **Error handling:** Comprehensive error messages for debugging
- **Health checks:** Built-in endpoint for monitoring
- **Auto-documentation:** Interactive API docs at `/docs`

## Performance Considerations

- Model is loaded once at startup for efficiency
- Single worker recommended due to large model size
- Images processed in-memory to reduce latency
- CPU inference (GPU optional if available)

## Security

- File size limits enforced (10MB)
- Content type validation
- Image format validation
- CORS configured (modify in `main.py` for production)

## Troubleshooting

### Model not loading
- Verify `best_model.pth` is in the same directory as `main.py`
- Check the model file is not corrupted (should be ~214MB)

### Out of memory errors
- Increase Cloud Run memory allocation to 2Gi or 4Gi
- Use single worker configuration

### Slow predictions
- First prediction may be slower due to model warmup
- Consider using GPU-enabled Cloud Run for better performance

## License

See project repository for license information.
