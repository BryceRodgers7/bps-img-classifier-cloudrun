"""
FastAPI Backend for Bird/Plane/Superman Image Classifier

Provides a REST API for image classification running on Google Cloud Run.
"""

import os
import io
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

from classifier import BirdPlaneSupermanClassifier


# Global classifier instance
classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup and cleanup on shutdown"""
    global classifier
    
    # Startup: Load the model
    print("Loading classifier model...")
    model_path = "best_model.pth"
    confidence_threshold = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
    
    try:
        classifier = BirdPlaneSupermanClassifier(
            model_path=model_path,
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
