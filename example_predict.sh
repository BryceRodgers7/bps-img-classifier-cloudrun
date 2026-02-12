#!/bin/bash

# Sample shell script for testing the Bird/Plane/Superman Classifier API

API_URL="${1:-http://localhost:8080}"
IMAGE_FILE="${2:-test_image.jpg}"

echo "Testing API at: $API_URL"
echo "Image file: $IMAGE_FILE"
echo ""

# Health check
echo "=== Health Check ==="
curl -s "$API_URL/health" | python -m json.tool
echo ""

# Get model info
echo "=== Model Info ==="
curl -s "$API_URL/info" | python -m json.tool
echo ""

# Predict image
echo "=== Prediction ==="
curl -s -X POST -F "file=@$IMAGE_FILE" "$API_URL/predict" | python -m json.tool
