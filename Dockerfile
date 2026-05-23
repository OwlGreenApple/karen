# ── Stage 1: build React dashboard ────────────────────────────────────────────
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend
COPY karen/dashboard/frontend/package.json karen/dashboard/frontend/package-lock.json* ./
RUN npm ci --prefer-offline 2>/dev/null || npm install

COPY karen/dashboard/frontend/ ./
RUN npm run build

# ── Stage 2: Python runtime ────────────────────────────────────────────────────
FROM python:3.11-slim

# System deps for ccxt / aiohttp
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY karen/ ./karen/
COPY pyproject.toml ruff.toml ./

# Copy built frontend from stage 1
COPY --from=frontend-builder /frontend/dist ./karen/dashboard/static/

# Data directory (will be overridden by volume mount)
RUN mkdir -p /app/data /app/logs

# Default env file location — mount your real .env here
COPY .env.example .env

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000

CMD ["python", "-m", "karen.main"]
