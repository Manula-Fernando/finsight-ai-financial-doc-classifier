# Base image
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 tesseract-ocr && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt --no-cache-dir

# Copy source
COPY src ./src
COPY app ./app
COPY api ./api
COPY models ./models
COPY outputs ./outputs
COPY README.md ./

# Expose ports: 8501 Streamlit, 8000 FastAPI
EXPOSE 8501 8000

# Default command launches both API and Streamlit via a simple process manager
RUN pip install fastapi uvicorn[standard] supervisor

COPY <<'SUPERVISOR' /etc/supervisord.conf
[supervisord]
nodaemon=true

[program:api]
command=uvicorn api.main:app --host 0.0.0.0 --port 8000
directory=/app
autostart=true
autorestart=true

[program:ui]
command=streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
directory=/app
autostart=true
autorestart=true
SUPERVISOR

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisord.conf"]
