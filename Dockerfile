# Multi-stage build. The "builder" stage installs Python dependencies
# into a self-contained venv; the runtime stage copies ONLY that venv
# plus the actual application code, not the pip cache, build tooling, or
# anything else pip/apt pulled in along the way. Concretely: nemoguardrails
# and its own transitive tree are large, and none of that installation-
# time weight needs to exist twice or linger in the final image layer.

FROM python:3.12-slim AS builder

WORKDIR /code

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy only the dependency manifest first — Docker's layer cache means
# this expensive `pip install` step is skipped entirely on rebuilds where
# app code changed but requirements.txt didn't, which is most rebuilds
# during normal development.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


FROM python:3.12-slim AS runtime

# Runs as a non-root user rather than the container default root —
# standard hardening: if the app process is ever compromised, it isn't
# running as root inside the container.
RUN useradd --create-home --shell /bin/bash appuser

WORKDIR /code

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY app/ ./app/
COPY frontend/ ./frontend/

# app/data/ holds the seeded SQLite commerce DB and the embedded Qdrant
# index (see vector_store.py / mock_db.py) — created at runtime, owned by
# appuser so it isn't accidentally root-owned inside a mounted volume.
RUN mkdir -p /code/app/data && chown -R appuser:appuser /code

USER appuser

EXPOSE 8000

# Reuses the app's own GET /health (checks real Mongo connectivity, not
# just "is the process alive") rather than a separate synthetic check —
# one source of truth for what "healthy" means, shared with anything
# else (a load balancer, docker-compose's own depends_on: condition)
# that wants to ask the same question.
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
