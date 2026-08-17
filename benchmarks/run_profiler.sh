#!/usr/bin/env bash
#
# run_profiler.sh — RK AgriDig Phase 1, Task 1.3
#
# Benchmarks the Phi-3-mini GGUF using llama-bench: measures prompt-processing
# (pp) and text-generation (tg) throughput separately, at batch sizes 1/8/32,
# and tracks peak RAM usage of the benchmark process alongside it.
#
# llama-bench does not report RAM itself, so this script launches it as a
# background process and polls its RSS via /proc/<pid>/status while it runs,
# which is the standard way to get real peak resident memory on Linux without
# extra dependencies (no psutil needed here — that's used in thermal_monitor.py
# instead, where it interleaves with lm-sensors reads).
#
# Usage:
#   bash benchmarks/run_profiler.sh
#   bash benchmarks/run_profiler.sh --threads 4
#   bash benchmarks/run_profiler.sh --model models/phi3_mini_4k_instruct.gguf
#
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${REPO_ROOT}/models/phi3_mini_4k_instruct.gguf"
LLAMA_BENCH="${REPO_ROOT}/llama.cpp/build/bin/llama-bench"
RESULTS_DIR="${REPO_ROOT}/benchmarks/results"
RAW_JSON="${RESULTS_DIR}/llama_bench_raw.json"
RAW_TXT="${RESULTS_DIR}/../llama_bench_results.txt"  # matches project structure spec (benchmarks/llama_bench_results.txt)
FINAL_JSON="${RESULTS_DIR}/performance_metrics.json"
RAM_SAMPLE_LOG="${RESULTS_DIR}/.ram_samples.tmp"

PHYSICAL_CORES="$(lscpu 2>/dev/null | awk -F: '/^Core\(s\) per socket/{c=$2} /^Socket\(s\)/{s=$2} END{if(c&&s) print c*s}')"
THREADS="${PHYSICAL_CORES:-$(nproc 2>/dev/null || echo 4)}"
# NOTE: deliberately using physical core count, not nproc (logical/hyperthreaded
# count). On hyperthreaded CPUs, llama.cpp's CPU inference is typically FASTER
# with threads == physical cores; oversubscribing to logical thread count causes
# contention and can roughly HALVE throughput. Confirmed empirically on this
# project: 16 threads (logical) = 10.9 TPS, 8 threads (physical) = 20.1 TPS.
BATCH_SIZES="1,8,32"
N_PROMPT=512   # tokens for prompt-processing test
N_GEN=128      # tokens for text-generation test
REPETITIONS=3

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log()     { echo -e "${BLUE}[profiler]${NC} $*"; }
success() { echo -e "${GREEN}✓${NC} $*"; }
warn()    { echo -e "${YELLOW}⚠${NC} $*"; }
fail()    { echo -e "${RED}✗ $*${NC}" >&2; }
die()     { fail "$1"; exit 1; }

# ---------------------------------------------------------------------------
# Parse args
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --threads) THREADS="$2"; shift 2 ;;
    --model)   MODEL_PATH="$2"; shift 2 ;;
    --batch-sizes) BATCH_SIZES="$2"; shift 2 ;;
    *) warn "Unknown argument: $1"; shift ;;
  esac
done

mkdir -p "$RESULTS_DIR"

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
echo "======================================================================"
echo "  RK AgriDig — Performance Profiler (Task 1.3)"
echo "======================================================================"

if [[ ! -f "$MODEL_PATH" ]]; then
  die "Model not found at $MODEL_PATH. Run: python3 models/download_model.py"
fi

if [[ ! -x "$LLAMA_BENCH" ]]; then
  die "llama-bench not found/executable at $LLAMA_BENCH. Run setup.sh first (Task 1.2)."
fi

MODEL_SIZE_BYTES=$(stat -c%s "$MODEL_PATH" 2>/dev/null || stat -f%z "$MODEL_PATH")
MODEL_SIZE_GB=$(awk "BEGIN { printf \"%.3f\", $MODEL_SIZE_BYTES / 1073741824 }")

log "Model:      $MODEL_PATH (${MODEL_SIZE_GB} GB)"
log "Threads:    $THREADS"
log "Batch sizes: $BATCH_SIZES"
log "pp tokens:  $N_PROMPT | tg tokens: $N_GEN | repetitions: $REPETITIONS"
echo ""

# ---------------------------------------------------------------------------
# RAM sampling helper
# ---------------------------------------------------------------------------
# Polls RSS (in KB) of a given PID every 0.5s until the PID exits.
# Writes one KB value per line to $1 (a file path).
sample_ram() {
  local pid="$1"
  local outfile="$2"
  : > "$outfile"
  while kill -0 "$pid" 2>/dev/null; do
    if [[ -r "/proc/$pid/status" ]]; then
      awk '/VmRSS/ {print $2}' "/proc/$pid/status" 2>/dev/null >> "$outfile"
    fi
    sleep 0.5
  done
}

peak_ram_kb_from_log() {
  local logfile="$1"
  if [[ -s "$logfile" ]]; then
    sort -n "$logfile" | tail -1
  else
    echo "0"
  fi
}

# ---------------------------------------------------------------------------
# Run llama-bench with RAM sampling wrapped around it
# ---------------------------------------------------------------------------
log "Running llama-bench (this loads the model fresh, may take ~30s to start)..."

LOAD_START=$(date +%s.%N)

# Launch llama-bench in the background so we can sample its RSS while it runs.
"$LLAMA_BENCH" \
  -m "$MODEL_PATH" \
  -p "$N_PROMPT" \
  -n "$N_GEN" \
  -b "$BATCH_SIZES" \
  -t "$THREADS" \
  -r "$REPETITIONS" \
  -o json \
  > "$RAW_JSON" 2>"${RESULTS_DIR}/.llama_bench_stderr.log" &
BENCH_PID=$!

sample_ram "$BENCH_PID" "$RAM_SAMPLE_LOG" &
SAMPLER_PID=$!

wait "$BENCH_PID"
BENCH_EXIT=$?
wait "$SAMPLER_PID" 2>/dev/null

LOAD_END=$(date +%s.%N)
TOTAL_ELAPSED=$(awk "BEGIN { printf \"%.2f\", $LOAD_END - $LOAD_START }")

if [[ "$BENCH_EXIT" -ne 0 ]]; then
  fail "llama-bench exited with code $BENCH_EXIT."
  echo "--- stderr ---" >&2
  cat "${RESULTS_DIR}/.llama_bench_stderr.log" >&2
  exit 1
fi

if [[ ! -s "$RAW_JSON" ]]; then
  die "llama-bench produced no output. Check ${RESULTS_DIR}/.llama_bench_stderr.log"
fi

success "llama-bench completed in ${TOTAL_ELAPSED}s."

# --- Also produce a human-readable markdown table, per project structure spec ---
# (benchmarks/llama_bench_results.txt). This re-runs llama-bench with -o md;
# the model file is already OS-page-cached from the run above, so this is
# fast and does not meaningfully add to total profiling time.
log "Generating human-readable results table (benchmarks/llama_bench_results.txt)..."
{
  echo "# RK AgriDig — llama-bench Results"
  echo "# Generated: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# Model: $MODEL_PATH"
  echo "# Threads: $THREADS | Batch sizes: $BATCH_SIZES | pp=$N_PROMPT tg=$N_GEN reps=$REPETITIONS"
  echo ""
  "$LLAMA_BENCH" \
    -m "$MODEL_PATH" \
    -p "$N_PROMPT" \
    -n "$N_GEN" \
    -b "$BATCH_SIZES" \
    -t "$THREADS" \
    -r "$REPETITIONS" \
    -o md
} > "$RAW_TXT" 2>>"${RESULTS_DIR}/.llama_bench_stderr.log"

if [[ -s "$RAW_TXT" ]]; then
  success "Wrote $RAW_TXT"
else
  warn "Could not generate $RAW_TXT (non-fatal — JSON results are still complete)"
fi

PEAK_RAM_KB=$(peak_ram_kb_from_log "$RAM_SAMPLE_LOG")
PEAK_RAM_GB=$(awk "BEGIN { printf \"%.3f\", $PEAK_RAM_KB / 1048576 }")
log "Peak RSS sampled during benchmark: ${PEAK_RAM_GB} GB"

if (( $(awk "BEGIN { print ($PEAK_RAM_GB > 7.0) }") )); then
  warn "Peak RAM (${PEAK_RAM_GB} GB) EXCEEDS the ADTC 7GB budget!"
elif (( $(awk "BEGIN { print ($PEAK_RAM_GB > 5.0) }") )); then
  warn "Peak RAM (${PEAK_RAM_GB} GB) is within budget but above the 5GB comfort margin."
else
  success "Peak RAM (${PEAK_RAM_GB} GB) comfortably within budget."
fi

# ---------------------------------------------------------------------------
# Parse llama-bench's raw JSON and reshape into the plan's target schema
# ---------------------------------------------------------------------------
log "Parsing results..."

python3 - "$RAW_JSON" "$FINAL_JSON" "$MODEL_PATH" "$MODEL_SIZE_GB" "$PEAK_RAM_GB" "$THREADS" <<'PYEOF'
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

raw_path, final_path, model_path, model_size_gb, peak_ram_gb, threads = sys.argv[1:7]

with open(raw_path) as f:
    raw = json.load(f)

if not raw:
    print("ERROR: llama-bench JSON output was empty.", file=sys.stderr)
    sys.exit(1)

# llama-bench JSON: one row per (test type, batch size) combination.
# "n_prompt" > 0 and "n_gen" == 0  -> a prompt-processing (pp) test
# "n_gen" > 0    and "n_prompt" == 0 -> a text-generation (tg) test
pp_rows = [r for r in raw if r.get("n_prompt", 0) > 0 and r.get("n_gen", 0) == 0]
tg_rows = [r for r in raw if r.get("n_gen", 0) > 0 and r.get("n_prompt", 0) == 0]

def rows_by_batch(rows):
    out = {}
    for r in rows:
        b = r.get("n_batch", r.get("batch_size", "unknown"))
        # avg_ts = average tokens/sec, per llama-bench's own JSON schema
        ts = r.get("avg_ts")
        ns_per_token = (1e9 / ts) if ts else None
        out[str(b)] = {
            "tokens_per_second": round(ts, 3) if ts is not None else None,
            "per_token_ms": round(ns_per_token / 1e6, 4) if ns_per_token else None,
            "stddev_ts": round(r.get("stddev_ts", 0), 3),
        }
    return out

pp_by_batch = rows_by_batch(pp_rows)
tg_by_batch = rows_by_batch(tg_rows)

# Overall throughput figure: use tg (text generation) at batch size 1,
# since that's the realistic single-user farmer-facing scenario the
# ADTC throughput target (15+ TPS) is meant to reflect.
overall_tps = None
if "1" in tg_by_batch and tg_by_batch["1"]["tokens_per_second"]:
    overall_tps = tg_by_batch["1"]["tokens_per_second"]
elif tg_by_batch:
    overall_tps = next(iter(tg_by_batch.values()))["tokens_per_second"]

first_row = raw[0]

result = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "model_path": model_path,
    "model_size_gb": round(float(model_size_gb), 3),
    "threads": int(threads),
    "cpu_info": first_row.get("cpu_info", "unknown"),
    "build_commit": first_row.get("build_commit", "unknown"),
    "prompt_processing_per_token_ms": (
        pp_by_batch.get("1", {}).get("per_token_ms")
        if pp_by_batch else None
    ),
    "generation_per_token_ms": (
        tg_by_batch.get("1", {}).get("per_token_ms")
        if tg_by_batch else None
    ),
    "peak_ram_usage_gb": round(float(peak_ram_gb), 3),
    "throughput_tokens_per_second": overall_tps,
    "prompt_processing_by_batch_size": pp_by_batch,
    "text_generation_by_batch_size": tg_by_batch,
    "adtc_targets": {
        "min_tps": 15,
        "max_ram_gb": 7,
        "meets_tps_target": bool(overall_tps and overall_tps >= 15),
        "meets_ram_target": float(peak_ram_gb) <= 7.0,
    },
}

with open(final_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"Wrote {final_path}")
print(f"  Overall TPS (tg, batch=1): {overall_tps}")
print(f"  Peak RAM: {peak_ram_gb} GB")
print(f"  Meets ADTC TPS target (>=15): {result['adtc_targets']['meets_tps_target']}")
print(f"  Meets ADTC RAM target (<=7GB): {result['adtc_targets']['meets_ram_target']}")
PYEOF

PARSE_EXIT=$?

# Clean up temp files
rm -f "$RAM_SAMPLE_LOG" "${RESULTS_DIR}/.llama_bench_stderr.log"

if [[ "$PARSE_EXIT" -ne 0 ]]; then
  die "Failed to parse llama-bench output into final schema. Raw output preserved at $RAW_JSON"
fi

echo ""
echo "======================================================================"
success "Profiling complete."
echo "  Raw llama-bench JSON: $RAW_JSON"
echo "  Final metrics JSON:   $FINAL_JSON"
echo "======================================================================"