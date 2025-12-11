FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p ./CAM_project

# Expose ports
EXPOSE 8080

# Environment variables
ENV CHROMA_PATH=/app/CAM_project/chroma_db
ENV CHROMA_PORT=8000
ENV PROXY_PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/v1/memory/debug || exit 1

# Default command - start the CAM proxy
CMD ["uvicorn", "proxy_api.app:app", "--host", "0.0.0.0", "--port", "8080"]