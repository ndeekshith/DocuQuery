FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN dnf-get update && dnf-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create necessary directories
RUN mkdir -p data/vectorstore data/temp_docs logs models/embeddings

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run the application
ENTRYPOINT ["streamlit", "run", "app_new.py", "--server.port=8501", "--server.address=0.0.0.0"]
