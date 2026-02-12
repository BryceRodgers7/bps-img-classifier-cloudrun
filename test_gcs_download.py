"""
Test script to verify GCS model download functionality
"""

import os
from pathlib import Path

# Set environment variables for testing
os.environ["GCS_BUCKET_NAME"] = "bps-model"
os.environ["GCS_MODEL_PATH"] = "best_model.pth"

def test_model_download():
    """Test the model download from GCS"""
    from main import download_model_from_gcs
    
    bucket_name = os.getenv("GCS_BUCKET_NAME", "bps-model")
    model_blob_name = os.getenv("GCS_MODEL_PATH", "best_model.pth")
    model_local_path = "models/best_model.pth"
    
    print(f"Testing model download from gs://{bucket_name}/{model_blob_name}")
    print(f"Target local path: {model_local_path}")
    
    try:
        download_model_from_gcs(bucket_name, model_blob_name, model_local_path)
        
        if os.path.exists(model_local_path):
            file_size = os.path.getsize(model_local_path)
            print(f"✓ Success! Model file exists: {file_size / (1024*1024):.1f} MB")
            return True
        else:
            print("✗ Failed: Model file does not exist after download")
            return False
            
    except Exception as e:
        print(f"✗ Failed with error: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("GCS Model Download Test")
    print("=" * 60)
    
    success = test_model_download()
    
    print("=" * 60)
    if success:
        print("Test PASSED")
    else:
        print("Test FAILED")
    print("=" * 60)
