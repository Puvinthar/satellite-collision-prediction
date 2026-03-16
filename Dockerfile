FROM python:3.12-slim

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application
COPY . .

# Expose port
EXPOSE 8051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8051/')" || exit 1

# Run with gunicorn for production
CMD ["gunicorn", "src.app:server", \
     "--bind", "0.0.0.0:8051", \
     "--workers", "2", \
     "--timeout", "120", \
     "--access-logfile", "-"]
