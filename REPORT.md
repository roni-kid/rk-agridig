# RK AgriDig — Technical Report

*Africa Deep Tech Challenge 2026*

---

## Phase 1: Model Selection & Profiling

**Status:** ✅ Complete

### Summary

Phi-3-mini-4k-instruct (Q4_K_M quantization, ~2.23GB) was selected and validated
against all ADTC hardware constraints. Throughput and memory targets are met with
comfortable margin. A genuine thermal risk was identified under sustained
continuous load and is addressed as a UI-level design constraint in Phase 3
rather than a reason to change models.

### Model & Environment

| | |
|---|---|
| Model | `microsoft/Phi-3-mini-4k-instruct-gguf`, file `Phi-3-mini-4k-instruct-q4.gguf` |
| Quantization | Q4_K_M (per model card; filename says "q4" but is confirmed Q4_K_M) |
| Size on disk | 2.23 GB |
| Checksum | SHA256 verified against HuggingFace Hub API (build plan originally specified MD5; HF does not publish MD5 for this file, so SHA256 was used instead) |
| Inference engine | llama.cpp (build 10454, commit `4df29be4f`), CPU-only |
| Dev/test hardware | AMD Ryzen AI 7 350 (8 physical cores / 16 logical threads), WSL2 Ubuntu on Windows 11, 7.3GB RAM visible to WSL2 |

### Task 1.1 — Model Download

`models/download_model.py` downloads the model via `huggingface_hub`, verifies
its SHA256 checksum against Hub metadata (fetched dynamically rather than
hardcoded, to stay resilient to upstream changes), and reports timing/size.
Download completed in 10m 46s at an average 3.53 MB/s.

### Task 1.2 — llama.cpp Compilation

`setup.sh` clones and compiles llama.cpp with CMake, CPU-only, targeting
Ubuntu 22.04-compatible toolchains. `llama-cli`, `llama-bench`, and
`llama-server` all built and verified working via a sanity inference call.

### Task 1.3 — Performance Profiling

`benchmarks/run_profiler.sh` wraps `llama-bench` (prompt-processing and
text-generation tested separately, batch sizes 1/8/32) with peak-RSS RAM
sampling.

**Important tuning finding:** the CPU has 8 physical cores but reports 16
logical (hyperthreaded) threads. An initial profiling run defaulted to
`nproc` (16 threads) and produced only **12.16 TPS** — below the 15 TPS ADTC
target. A manual thread-count sweep isolated the cause:

| Threads | TPS (tg, 128 tok) |
|---|---|
| 4 | 17.41 |
| 6 | 19.29 |
| **8 (physical core count)** | **20.09 (peak)** |
| 12 | 17.25 |
| 16 (logical/hyperthreaded count) | 10.93 |

Thread oversubscription past the physical core count caused contention that
nearly **halved throughput**. `run_profiler.sh` was corrected to default to
physical core count (via `lscpu`) rather than `nproc`. The corrected full
profiler run — with concurrent RAM-sampling overhead included — produced:

| Metric | Target | Result | Status |
|---|---|---|---|
| Throughput | ≥15 TPS | **17.69 TPS** | ✅ Pass (+18%) |
| Peak RAM | ≤7GB | **3.65 GB** | ✅ Pass (3.35GB headroom) |

Full results: `benchmarks/results/performance_metrics.json`.

### Task 1.4 — Thermal Monitoring

**Platform note:** `benchmarks/thermal_monitor.py` (the ADTC-spec Linux tool,
using `lm-sensors`) is correctly implemented for Ubuntu, but WSL2's kernel has
no hardware sensor access — confirmed via `sensors-detect`, which fails with
`Module cpuid not found in directory /lib/modules/<wsl-kernel>`. This is a
WSL2 platform limitation, not a script defect; the script correctly detects
and reports "sensors unavailable" rather than fabricating data.

**Workaround used:** a companion script, `benchmarks/thermal_monitor_windows.ps1`,
polls the real CPU package temperature from the Windows host via WMI
(`MSAcpi_ThermalZoneTemperature`) while inference load runs inside WSL2. This
requires an elevated (Administrator) PowerShell session.

**Test conditions:** 20 consecutive back-to-back `llama-cli` calls (~128
tokens generated each, `-t 8`, zero gap between calls) over ~220 seconds.
CPU was not at a cold baseline at test start (91°C initial reading).

**Results:**

| Metric | Target | Result | Status |
|---|---|---|---|
| Avg temp | — | 63.4°C | — |
| Max temp | <80°C (target), <85°C (disqualification) | **98°C** | ❌ Exceeds both |
| Sustained >80°C | avoid | ~200 seconds | ❌ |

The load-correlated climb (91°C → 98°C plateau → steady cooldown to 66°C
after the loop ended) confirms this is a genuine thermal response to
sustained CPU-bound inference, not sensor noise.

**Caveats on this result:**
1. The test used unrealistic continuous back-to-back load with zero
   inter-query gaps — real farmer usage (one question, read the answer,
   ask another) will not sustain load this way.
2. The CPU started already warm, not at a genuine cold baseline.
3. This reflects one specific laptop's cooling solution and may not match
   ADTC's actual judged hardware.

**Decision:** This is treated as a real design constraint, not a reason to
reconsider the model. Mitigation planned for Phase 3: rate-limit / minimum
cooldown between consecutive UI-triggered inference calls in `ui/app.py`.
**Before final ADTC submission, this test should be repeated on real
bare-metal Ubuntu 22.04 hardware** using `benchmarks/thermal_monitor.py`
directly (real `lm-sensors` access, no Windows workaround needed), ideally
from a genuine cold start and with realistic query spacing.

### Phase 1 Decision

**Proceed with Phi-3-mini Q4_K_M.** Throughput and memory both clear ADTC
targets with meaningful margin. The thermal finding is real and worth
designing around (query rate-limiting in the UI layer) but does not on its
own justify pivoting to a smaller model — the same sustained-load stress
test would likely produce similar heat on any CPU-only model of comparable
size running continuously with no pauses.

---

## Phase 2: Prompt Engineering & Evaluation

*Not yet started.*

## Phase 3: UI & Deployment

*Not yet started.*
