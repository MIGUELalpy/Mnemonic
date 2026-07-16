# Mnemonic — Dockerfile
#
# Multi-stage build:
#   Stage 1 (builder): installs all Python dependencies into an isolated venv.
#   Stage 2 (runtime): copies only the venv and source code into a slim image.
#
# The same image is used for both services (app and worker).
# The entry command is overridden in docker-compose.yml:
#   app    → uvicorn app.main:app ...
#   worker → celery -A app.worker.celery_app worker ...

#Stage 1: builder 
FROM python:3.12-slim AS builder

# Install build dependencies needed to compile psycopg2 and other C extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment inside the builder stage. Copying the venv to the runtime stage is cleaner than copying site-packages directly.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies.
# Copying requirements first (before source code) lets Docker cache this layer — the expensive pip install step is only re-run when requirements.txt changes.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


#  Stage 2: runtime 
FROM python:3.12-slim AS runtime

# Install only the runtime system libraries (not build tools). libpq-dev is the PostgreSQL client library required by psycopg2 at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage.
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for the application process. Running as root inside a container is a security risk even on a local network.
RUN groupadd --gid 1001 mnemonic \
    && useradd --uid 1001 --gid mnemonic --shell /bin/bash --create-home mnemonic

# Set the working directory. All paths in the app are relative to this.
WORKDIR /app

# Copy the application source code. Ownership is set to the non-root user at copy time.
COPY --chown=mnemonic:mnemonic . .

# Switch to the non-root user before the entrypoint.
USER mnemonic

# Expose the FastAPI port (used by the `app` service only; ignored by `worker`).
EXPOSE 8000

# Default command runs the FastAPI server. Overridden in docker-compose.yml for the Celery worker service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]