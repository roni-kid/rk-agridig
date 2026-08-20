# RK AgriDig — offline crop disease diagnostic system
# Base: Ubuntu 22.04 LTS (matches ADTC target hardware reference OS)

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST=0.0.0.0:11434

# -- System dependencies -----------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
        python3-pip \
        curl \
        ca-certificates \
        gosu \
        zstd \
    && rm -rf /var/lib/apt/lists/*

# -- Ollama (bundles its own llama.cpp-based runtime) ------------------------
RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

# -- Python dependencies (cached layer — only rebuilds when requirements.txt changes)
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# -- Application code ---------------------------------------------------------
COPY src/ ./src/
COPY ui/ ./ui/
COPY Modelfile ./Modelfile

# NOTE: The GGUF model itself is NOT copied into the image (it's large and
# gitignored). Mount it at runtime via the volume defined in docker-compose.yml,
# at /app/models/phi3_mini_4k_instruct.gguf — the entrypoint below registers
# it with Ollama on first boot using `ollama create`.

COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 7860 11434

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:11434/api/tags || exit 1

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]