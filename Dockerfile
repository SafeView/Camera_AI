# Python FastAPI + Ultralytics/OpenCV
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps for opencv/ultralytics and ffmpeg
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for caching
COPY requirements.txt ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .

# Create runtime dirs (will be mounted as volumes in compose too)
RUN mkdir -p /app/uploads /app/api_results

EXPOSE 8000

# Default env (can be overridden by compose)
ENV API_HOST=0.0.0.0 \
    API_PORT=8000

# Start server
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]

