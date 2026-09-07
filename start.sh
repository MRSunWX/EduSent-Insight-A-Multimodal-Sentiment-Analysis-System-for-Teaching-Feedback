#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -x "$project_dir/backend/.venv/bin/uvicorn" ]]; then
  echo "Backend environment is missing. Run: cd backend && uv sync or install requirements.txt"
  exit 1
fi

cleanup() {
  [[ -n "${backend_pid:-}" ]] && kill "$backend_pid" 2>/dev/null || true
  [[ -n "${ollama_pid:-}" ]] && kill "$ollama_pid" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

env_file="$project_dir/backend/.env"
local_qwen="false"
ollama_models_path="${OLLAMA_MODELS:-}"
ollama_model="qwen3.5:9b"
local_qwen_host="127.0.0.1:11435"
if [[ -f "$env_file" ]]; then
  local_qwen="$(sed -n 's/^LOCAL_QWEN=//p' "$env_file" | tail -n 1)"
  configured_models_path="$(sed -n 's/^OLLAMA_MODELS_PATH=//p' "$env_file" | tail -n 1)"
  configured_model="$(sed -n 's/^OPENAI_MODEL=//p' "$env_file" | tail -n 1)"
  configured_qwen_host="$(sed -n 's/^LOCAL_QWEN_HOST=//p' "$env_file" | tail -n 1)"
  [[ -n "$configured_models_path" ]] && ollama_models_path="$configured_models_path"
  [[ -n "$configured_model" ]] && ollama_model="$configured_model"
  [[ -n "$configured_qwen_host" ]] && local_qwen_host="$configured_qwen_host"
fi

if [[ "$local_qwen" == "true" ]]; then
  if ! command -v ollama >/dev/null 2>&1; then
    echo "Ollama is required for local Qwen3.5 but was not found."
    exit 1
  fi
  if ! curl -fsS "http://$local_qwen_host/api/version" >/dev/null 2>&1; then
    if [[ -n "$ollama_models_path" ]]; then
      OLLAMA_MODELS="$ollama_models_path" OLLAMA_HOST="$local_qwen_host" ollama serve &
    else
      OLLAMA_HOST="$local_qwen_host" ollama serve &
    fi
    ollama_pid=$!
    for _ in {1..40}; do
      curl -fsS "http://$local_qwen_host/api/version" >/dev/null 2>&1 && break
      sleep 0.25
    done
  fi
  if ! OLLAMA_HOST="$local_qwen_host" ollama show "$ollama_model" >/dev/null 2>&1; then
    visible_models="$(curl -fsS "http://$local_qwen_host/api/tags" 2>/dev/null || true)"
    echo "Local model $ollama_model is not available from Ollama at $local_qwen_host."
    echo "Visible models: $visible_models"
    exit 1
  fi
fi

cd "$project_dir/backend"
if [[ -f "$env_file" ]]; then
  "$project_dir/backend/.venv/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000 --env-file "$env_file" &
else
  "$project_dir/backend/.venv/bin/uvicorn" app.main:app --host 0.0.0.0 --port 8000 &
fi
backend_pid=$!

cd "$project_dir/frontend"
npm run dev
