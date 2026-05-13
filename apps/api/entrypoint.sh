#!/usr/bin/env bash
set -euo pipefail

echo "[entrypoint] running alembic migrations"
alembic upgrade head

echo "[entrypoint] starting uvicorn on :${API_PORT:-8000}"
exec uvicorn finsight.main:app \
    --host 0.0.0.0 \
    --port "${API_PORT:-8000}" \
    --proxy-headers \
    --forwarded-allow-ips='*'
