#!/bin/bash

################################################################################
# RK AgriDig — Performance Profiler Script
#
# This script benchmarks the Phi-3-mini GGUF model using llama-bench.
# It measures:
# - Tokens per second (TPS) — throughput performance (Sperf)
# - Peak RAM usage — memory efficiency (Seff)
# - Latency (prompt processing + generation)
# - CPU utilization
#
# Output: JSON file with metrics for ADTC submission
#
# Requirements:
# - llama.cpp compiled (from setup.sh)
# - Phi-3-mini GGUF model downloaded
#
# Usage:
#     bash benchmarks/run_profiler.sh
#     bash benchmarks/run_profiler.sh --quick    # Faster benchmark
#     bash benchmarks/run_profiler.sh --full     # Comprehensive benchmark
#
################################################################################

set -e

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/benchmarks/results"
MODEL_PATH="${PROJECT_ROOT}/models/phi3_mini_4k_instruct.gguf"
LLAMABENCH_BIN="${PROJECT_ROOT}/llama.cpp/build/llama-bench"
LLAMA_MAIN="${PROJECT_ROOT}/llama.cpp/build/bin/main"

# If llama-bench not found, try alternative paths
if [[ ! -f "$LLAMABENCH_BIN" ]]; then
    LLAMABENCH_BIN="${PROJECT_ROOT}/llama.cpp/build/bin/llama-bench"
fi
if [[ ! -f "$LLAMABENCH_BIN" ]]; then
    LLAMABENCH_BIN=$(which llama-bench || echo "")
fi

# Benchmark parameters
BENCHMARK_MODE="${1:-standard}"  # quick, standard, full
CONTEXT_SIZE=2048               # Context window size
NUM_PREDICT=128                 # Number of tokens to generate
BATCH_SIZES=(1 8 32)            # Batch sizes to test
NUM_THREADS=$(nproc)            # Number of CPU threads
WARMUP_RUNS=1                   # Warmup iterations
BENCHMARK_RUNS=5                # Benchmark iterations

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m''
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ============================================================================
# Pre-flight Checks
# ============================================================================

log_info "========================================"
log_info "RK AgriDig — Performance Profiler"
log_info "========================================"

# Check if model exists
if [[ ! -f "$MODEL_PATH" ]]; then
    log_error "Model not found: $MODEL_PATH"
    log_info "Download it with: python models/download_model.py"
    exit 1
fi

MODEL_SIZE=$(du -sh "$MODEL_PATH" | awk '{print $1}')
log_success "Model found ($MODEL_SIZE): $MODEL_PATH"

# Check if llama-bench exists
if [[ ! -f "$LLAMABENCH_BIN" ]]; then
    log_error "llama-bench not found"
    log_info "Build it with: bash setup.sh"
    exit 1
fi

log_success "llama-bench found: $LLAMABENCH_BIN"

# Create results directory
mkdir -p "$RESULTS_DIR"
log_info "Results will be saved to: $RESULTS_DIR"

# ============================================================================
# Adjust Benchmark Parameters by Mode
# ============================================================================

case "$BENCHMARK_MODE" in
    quick)
        log_info "Running QUICK benchmark (fast, less accurate)"
        BATCH_SIZES=(1 8)
        BENCHMARK_RUNS=2
        NUM_PREDICT=64
        ;;
    standard)
        log_info "Running STANDARD benchmark"
        BATCH_SIZES=(1 8 32)
        BENCHMARK_RUNS=5
        NUM_PREDICT=128
        ;;
    full)
        log_info "Running FULL comprehensive benchmark"
        BATCH_SIZES=(1 8 32 64)
        BENCHMARK_RUNS=10
        NUM_PREDICT=256
        CONTEXT_SIZE=4096
        ;;
    *)
        log_warn "Unknown mode: $BENCHMARK_MODE. Using standard."
        ;;
esac

log_info "Benchmark parameters:"
log_info "  Context size: $CONTEXT_SIZE"
log_info "  Tokens to generate: $NUM_PREDICT"
log_info "  Batch sizes: ${BATCH_SIZES[@]}"
log_info "  CPU threads: $NUM_THREADS"
log_info "  Warmup runs: $WARMUP_RUNS"
log_info "  Benchmark runs: $BENCHMARK_RUNS"

# ============================================================================
# System Information
# ============================================================================

log_info ""
log_info "System Information:"

# CPU info
if [[ -f /proc/cpuinfo ]]; then
    CPU_MODEL=$(grep -m 1 "model name" /proc/cpuinfo | cut -d: -f2 | xargs)
    log_info "  CPU: $CPU_MODEL"
fi

# RAM info
if command -v free &> /dev/null; then
    TOTAL_RAM=$(free -h | awk 'NR==2 {print $2}')
    AVAILABLE_RAM=$(free -h | awk 'NR==2 {print $7}')
    log_info "  RAM: $TOTAL_RAM (Available: $AVAILABLE_RAM)"
fi

# ============================================================================
# Run Benchmarks
# ============================================================================

log_info ""
log_info "Starting benchmarks..."

RESULTS_JSON="${RESULTS_DIR}/performance_metrics.json"
RESULTS_TEXT="${RESULTS_DIR}/benchmark_results.txt"

# Initialize JSON output
cat > "$RESULTS_JSON" << EOF
{
  "benchmark_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "model": "Phi-3-mini-4k-instruct-Q4_K_M",
  "model_path": "$MODEL_PATH",
  "model_size_gb": $(du -b "$MODEL_PATH" | awk '{printf "%.2f", $1 / (1024^3)}'),
  "benchmark_mode": "$BENCHMARK_MODE",
  "context_size": $CONTEXT_SIZE,
  "num_predict": $NUM_PREDICT,
  "num_threads": $NUM_THREADS,
  "warmup_runs": $WARMUP_RUNS,
  "benchmark_runs": $BENCHMARK_RUNS,
  "results": []
}
EOF

# Track peak RAM usage
PEAK_RAM=0
START_TIME=$(date +%s)

# Run for each batch size
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    log_info ""
    log_info "Benchmarking with batch size: $BATCH_SIZE"
    
    BATCH_RESULT_FILE="${RESULTS_DIR}/benchmark_batch_${BATCH_SIZE}.txt"
    
    # Run llama-bench
    # Note: llama-bench typically outputs results to stdout
    # We'll capture and parse them
    {
        timeout 600 "$LLAMABENCH_BIN" \
            --model "$MODEL_PATH" \
            --n-threads $NUM_THREADS \
            --n-predict $NUM_PREDICT \
            --n-batch $BATCH_SIZE \
            --n-warmup $WARMUP_RUNS \
            --n-runs $BENCHMARK_RUNS \
            2>&1 || log_warn "llama-bench timeout or error for batch size $BATCH_SIZE"
    } | tee "$BATCH_RESULT_FILE"
    
    # Parse results (llama-bench output format varies; this is a template)
    # Extract key metrics
    if [[ -f "$BATCH_RESULT_FILE" ]]; then
        # Try to extract TPS (tokens per second)
        TPS=$(grep -i "tokens per second" "$BATCH_RESULT_FILE" | tail -1 | awk '{print $NF}' || echo "N/A")
        
        # Try to extract latency
        PROMPT_MS=$(grep -i "prompt processing" "$BATCH_RESULT_FILE" | tail -1 | awk '{print $NF}' || echo "N/A")
        GEN_MS=$(grep -i "generation" "$BATCH_RESULT_FILE" | tail -1 | awk '{print $NF}' || echo "N/A")
        
        log_info "  Batch $BATCH_SIZE: TPS=$TPS tokens/sec"
        log_info "  Prompt latency: ${PROMPT_MS}ms, Gen latency: ${GEN_MS}ms"
    fi
done

# ============================================================================
# Monitor RAM Usage
# ============================================================================

log_info ""
log_info "Measuring peak RAM usage..."

# Run a single inference and monitor RAM
if [[ -f "$LLAMA_MAIN" ]]; then
    # Create a simple prompt file
    PROMPT_FILE="${RESULTS_DIR}/test_prompt.txt"
    echo "What disease is affecting my maize plant?" > "$PROMPT_FILE"
    
    # Start inference in background and monitor RAM
    (
        "$LLAMA_MAIN" \
            --model "$MODEL_PATH" \
            --threads $NUM_THREADS \
            --n-predict 100 \
            --prompt-cache-all \
            --file "$PROMPT_FILE" \
            > /dev/null 2>&1
    ) &
    
    INFERENCE_PID=$!
    
    # Monitor RAM while inference runs
    while kill -0 $INFERENCE_PID 2>/dev/null; do
        if command -v ps &> /dev/null; then
            CURRENT_RAM=$(ps aux | grep $INFERENCE_PID | grep -v grep | awk '{print int($6)}')
            if [[ ! -z "$CURRENT_RAM" ]]; then
                CURRENT_RAM_GB=$(echo "scale=2; $CURRENT_RAM / 1024 / 1024" | bc)
                if (( $(echo "$CURRENT_RAM_GB > $PEAK_RAM" | bc -l) )); then
                    PEAK_RAM=$CURRENT_RAM_GB
                fi
            fi
        fi
        sleep 0.1
    done
    
    wait $INFERENCE_PID
    
    log_success "Peak RAM usage: ${PEAK_RAM} GB"
else
    log_warn "Could not find llama-main binary for RAM monitoring"
fi

# ============================================================================
# Calculate Efficiency Score
# ============================================================================

log_info ""
log_info "Calculating ADTC scores..."

# Seff = 100 × ((7 GB − Peak_RAM) / 7 GB)
if (( $(echo "$PEAK_RAM > 0" | bc -l) )); then
    SEFF=$(echo "scale=2; 100 * ((7 - $PEAK_RAM) / 7)" | bc)
    log_info "Seff (Efficiency): $SEFF"
else
    SEFF="N/A"
    log_warn "Could not measure peak RAM"
fi

# ============================================================================
# Compile Final Results
# ============================================================================

log_info ""
log_info "Compiling results..."

# Append results to JSON
cat >> "$RESULTS_JSON" << EOF
  "peak_ram_gb": $PEAK_RAM,
  "seff_score": $SEFF,
  "duration_seconds": $(($(date +%s) - START_TIME)),
  "notes": "Benchmark completed successfully"
}
EOF

log_success "Results saved to: $RESULTS_JSON"

# Print summary
log_info ""
log_info "=========================================="
log_success "BENCHMARK COMPLETE"
log_info "=========================================="

echo ""
echo "Results Summary:"
echo "  Model: Phi-3-mini Q4_K_M"
echo "  Peak RAM: ${PEAK_RAM} GB"
echo "  Seff Score: $SEFF"
echo ""
echo "Detailed results:"
echo "  JSON: $RESULTS_JSON"
echo "  Text: $RESULTS_TEXT"
echo ""
echo "Next steps:"
echo "  1. Review the performance metrics"
echo "  2. If TPS < 15 tokens/sec, consider smaller model or optimizations"
echo "  3. If RAM > 7GB, consider lower quantization (Q3 or Q2)"
echo "  4. Proceed to Phase 2: Prompt Engineering"
echo ""

# ============================================================================
# Clean Up
# ============================================================================

# Remove test prompt file
rm -f "${RESULTS_DIR}/test_prompt.txt"

log_success "Profiler ready for next phase!"
