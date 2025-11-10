# Use Python 3.10 slim image as base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for audio processing and build tools
# Note: gcc and python3-dev are required to build webrtcvad from source
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    gcc \
    g++ \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p ecapa audio_samples

# Set environment variables
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_SERVER_PORT=7860

# Expose the port Gradio runs on
EXPOSE 7860

# Run the application
CMD ["python", "app.py"]
