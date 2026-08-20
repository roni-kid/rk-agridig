#!/bin/bash
# docker-entrypoint.sh — RK AgriDig
#
# Starts the Ollama server in the background, waits for it to be ready,
# registers the mounted GGUF model (if not already registered), then
# launches the Gradio UI in the foreground.

set -euo pipefail

MODEL_PATH="/app/models/phi3_mini_4k_instruct.gguf"
MODEL_NAME="phi3-agridig"
OLLAMA_READY_TIMEOUT=60

echo "[entrypoint] Starting Ollama server..."
ollama serve &
OLLAMA_PID=$!

echo "[entrypoint] Waiting for Ollama server to be ready..."
elapsed=0
until curl -sf http://localhost:11434/api/tags >/dev/null 2>&1; do
    sleep 2
    elapsed=$((elapsed + 2))
    if [ "$elapsed" -ge "$OLLAMA_READY_TIMEOUT" ]; then
        echo "[entrypoint] ERROR: Ollama server did not become ready within ${OLLAMA_READY_TIMEOUT}s"
        exit 1
    fi
done
echo "[entrypoint] Ollama server is ready."

if [ ! -f "$MODEL_PATH" ]; then
    echo "[entrypoint] WARNING: Model file not found at $MODEL_PATH"
    echo "[entrypoint]          Mount it via the 'models' volume in docker-compose.yml."
    echo "[entrypoint]          Continuing — the UI will launch but inference will fail"
    echo "[entrypoint]          until the model is available and the container is restarted."
else
    if ollama list | grep -q "^${MODEL_NAME}"; then
        echo "[entrypoint] Model '${MODEL_NAME}' already registered."
    else
        echo "[entrypoint] Registering model '${MODEL_NAME}' from ${MODEL_PATH}..."
        ollama create "${MODEL_NAME}" -f /app/Modelfile
        echo "[entrypoint] Model registered."
    fi
fi

echo "[entrypoint] Launching Gradio UI..."
python3 /app/ui/app.py &
APP_PID=$!

# Forward termination signals to both child processes for clean shutdown.
trap 'echo "[entrypoint] Shutting down..."; kill -TERM "$OLLAMA_PID" "$APP_PID" 2>/dev/null; wait' TERM INT

wait -n "$OLLAMA_PID" "$APP_PID"