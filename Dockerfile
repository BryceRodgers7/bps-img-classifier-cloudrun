# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model
COPY classifier.py .
COPY main.py .
COPY best_model.pth .

# Expose port (Cloud Run will set PORT env variable)
EXPOSE 8080

# Set environment variables for production
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Run the application with uvicorn
CMD exec uvicorn main:app --host 0.0.0.0 --port ${PORT}
