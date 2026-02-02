FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY checkpoints/ ./checkpoints/

# Create directories for outputs
RUN mkdir -p outputs models

# Set environment variables
ENV MODEL_PATH=/app/checkpoints/best.pth
ENV PYTHONPATH=/app

# Expose API port
EXPOSE 8000

# Run the API server
CMD ["python", "src/api/main.py", "--host", "0.0.0.0", "--port", "8000"]
