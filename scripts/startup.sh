#!/bin/sh
set -e

HOST="${API_HOST:-0.0.0.0}"
PORT="${PORT:-${API_PORT:-8000}}"

exec uvicorn api.main:app --host "$HOST" --port "$PORT"
