"""
FastAPI Backend for Bird/Plane/Superman Image Classifier

Provides a REST API for image classification running on Google Cloud Run.
"""

import os
import io
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn
from google.cloud import storage

from classifier import BirdPlaneSupermanClassifier


# Global classifier instance
classifier = None


def download_model_from_gcs(bucket_name: str, source_blob_name: str, destination_path: str):
    """
    Download model file from Google Cloud Storage if it doesn't exist locally
    
    Args:
        bucket_name: GCS bucket name
        source_blob_name: Path to the model file in the bucket
        destination_path: Local path where the model should be saved
    """
    # Check if file already exists
    if os.path.exists(destination_path):
        file_size = os.path.getsize(destination_path)
        print(f"Model file already exists at {destination_path} ({file_size / (1024*1024):.1f} MB)")
        return
    
    print(f"Model file not found locally. Downloading from gs://{bucket_name}/{source_blob_name}...")
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        
        # Initialize GCS client
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(source_blob_name)
        
        # Download the file
        blob.download_to_filename(destination_path)
        
        file_size = os.path.getsize(destination_path)
        print(f"Model downloaded successfully! ({file_size / (1024*1024):.1f} MB)")
        
    except Exception as e:
        print(f"Failed to download model from GCS: {e}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown"""
    global classifier
    
    # Configuration
    bucket_name = os.getenv("GCS_BUCKET_NAME", "bps-model")
    model_blob_name = os.getenv("GCS_MODEL_PATH", "models/best-model.pth")
    model_local_path = "models/best-model.pth"
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    
    # Download model from GCS if needed
    print("Checking for model file...")
    try:
        download_model_from_gcs(bucket_name, model_blob_name, model_local_path)
    except Exception as e:
        print(f"Failed to download model: {e}")
        raise
    
    # Load the classifier
    print("Loading classifier model...")
    try:
        classifier = BirdPlaneSupermanClassifier(
            model_path=model_local_path,
            confidence_threshold=confidence_threshold
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {e}")
        raise
    
    yield
    
    # Shutdown: Cleanup if needed
    print("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="Bird/Plane/Superman Classifier API",
    description="Image classification API for Bird, Plane, Superman, and Other categories",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with API information"""
    return {
        "name": "Bird/Plane/Superman Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict - Upload an image for classification",
            "health": "GET /health - Health check endpoint",
            "info": "GET /info - Model information"
        },
        "docs": "/docs"
    }


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint for Cloud Run"""
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return {"status": "healthy"}


@app.get("/info")
async def model_info() -> Dict[str, Any]:
    """Get model configuration and metadata"""
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    return classifier.get_model_info()


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Predict the class of an uploaded image
    
    Args:
        file: Image file (JPEG, PNG, GIF, WebP)
        
    Returns:
        JSON with predicted_class, confidence, and probabilities for all classes
    """
    # Validate model is loaded
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Validate file is provided
    if not file:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided"
        )
    
    # Validate content type
    allowed_content_types = [
        "image/jpeg",
        "image/png",
        "image/gif",
        "image/webp",
        "image/jpg"
    ]
    
    if file.content_type not in allowed_content_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type. Allowed types: {', '.join(allowed_content_types)}"
        )
    
    # Read file content
    try:
        contents = await file.read()
        
        # Check file size (10MB limit)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(contents) > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File too large. Maximum size: {max_size / (1024*1024)}MB"
            )
        
        # Open image with PIL
        image = Image.open(io.BytesIO(contents))
        
        # Ensure RGB format
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image file: {str(e)}"
        )
    
    # Run prediction
    try:
        pred_class, confidence, probabilities = classifier.predict_from_image(image)
        
        # Determine if threshold was applied
        max_prob = max(probabilities.values())
        threshold_applied = max_prob < classifier.confidence_threshold and pred_class == 'other'
        
        return {
            "predicted_class": pred_class,
            "confidence": round(confidence, 4),
            "probabilities": {k: round(v, 4) for k, v in probabilities.items()},
            "threshold_applied": threshold_applied
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


if __name__ == "__main__":
    # Get port from environment variable (Cloud Run sets this)
    port = int(os.getenv("PORT", "8080"))
    
    # Run with uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
