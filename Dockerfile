# Multi-stage build for smaller final image
# Stage 1: Base image with system dependencies
FROM python:3.10-slim as base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: Builder - install dependencies
FROM base as builder

COPY pyproject.toml requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Production - copy source and run
FROM base as production

COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY src/ src/
COPY configs/ configs/

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

ENTRYPOINT ["python", "src/train.py"]

# Stage 4: Development - includes dev tools
FROM builder as development

RUN pip install --no-cache-dir -e ".[dev]"

COPY . .

ENTRYPOINT ["python", "src/train.py"]