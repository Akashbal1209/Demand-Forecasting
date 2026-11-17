# Docker image for the forecast API (FastAPI + LightGBM).
#
# Build and run from the project root:
#     docker build -t demand-forecast .
#     docker run -p 8000:8000 demand-forecast
# then open http://localhost:8000

FROM python:3.13.9-slim

# LightGBM needs the OpenMP runtime, which the slim image does not include
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# install packages first, so this layer is reused when only the code changes
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# only what the API needs at runtime (see .dockerignore)
COPY app/ app/
COPY src/ src/
COPY models/ models/
COPY data/app.db data/app.db

# do not run the server as root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(f'http://localhost:{os.getenv(\"PORT\", \"8000\")}/health')"

# PORT is set by hosting platforms such as Render; 8000 when run locally
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
