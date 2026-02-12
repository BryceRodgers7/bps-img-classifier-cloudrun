"""
Sample Python client for the Bird/Plane/Superman Classifier API
"""

import requests
import sys
from pathlib import Path


def predict_image(image_path: str, api_url: str = "http://localhost:8080"):
    """Send image to prediction API and return results"""
    
    # Open and send the image
    with open(image_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{api_url}/predict", files=files)
    
    # Check for errors
    response.raise_for_status()
    
    return response.json()


def predict_multiple_images(image_paths: list, api_url: str = "http://localhost:8080"):
    """Predict multiple images"""
    results = []
    
    for img_path in image_paths:
        try:
            result = predict_image(img_path, api_url)
            results.append({
                'image': img_path,
                'prediction': result
            })
            print(f"✓ {Path(img_path).name}: {result['predicted_class']} ({result['confidence']:.2%})")
        except Exception as e:
            print(f"✗ {Path(img_path).name}: Error - {e}")
            results.append({
                'image': img_path,
                'error': str(e)
            })
    
    return results


if __name__ == "__main__":
    # Example 1: Single image
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        api_url = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:8080"
        
        print(f"Predicting: {image_path}")
        result = predict_image(image_path, api_url)
        
        print(f"\nPredicted Class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nAll Probabilities:")
        for cls, prob in sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {cls:10s}: {prob:.2%}")
    else:
        print("Usage: python example_predict.py <image_path> [api_url]")
        print("\nExamples:")
        print("  python example_predict.py test.jpg")
        print("  python example_predict.py test.jpg http://your-service.run.app")
