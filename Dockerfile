
# ==========================================
# 1. Builder Stage
# ==========================================
FROM python:3.10-slim as builder

WORKDIR /app

# Install system dependencies for build
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"

# Copy dependency definition
COPY pyproject.toml poetry.lock ./

# Export dependencies to requirements.txt (to install without poetry in runtime)
RUN poetry export --without-hashes --format=requirements.txt > requirements.txt

# ==========================================
# 2. Runtime Stage
# ==========================================
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (runtime only)
# Install system dependencies (runtime only)
# poppler-utils: PDF processing dependency (pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements from builder
COPY --from=builder /app/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run application
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
