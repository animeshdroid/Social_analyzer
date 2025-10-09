# Use official slim Python image for reduced size and fewer system dependencies
FROM python:3.9-slim

# Set working directory in the container
WORKDIR /app
RUN mkdir -p /app/models_cache && chmod -R 777 /app/models_cache
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV PYTHONPATH=/app
ENV MPLCONFIGDIR=/tmp

# Install necessary system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first for build cache optimization
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Create models cache and ensure it is writable (for HuggingFace/transformers)
RUN mkdir -p /app/models_cache && chmod -R 777 /app/models_cache

# Set environment variables for model and matplotlib cache
ENV TRANSFORMERS_CACHE=/app/models_cache
ENV MPLCONFIGDIR=/tmp
ENV PYTHONPATH=/app  

# Copy the rest of your code into the container
COPY . .

# Create a non-root user for security and use it
RUN adduser --disabled-password --gecos '' appuser
USER appuser

# Expose Streamlit default port
EXPOSE 8501

# Health check endpoint for container orchestration (optional)
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Start your app with Streamlit, targeting app/main.py
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true", "--server.fileWatcherType=none"]
