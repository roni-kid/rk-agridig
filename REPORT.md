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

**Correction — OpenBLAS was likely never actually active.** `setup.sh`
configures CMake with `-DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS`, but the
real benchmark output (`llama_bench_raw.json`) reports `"backends": "CPU"`
with no BLAS designation. Per llama.cpp's own behavior, an active BLAS
backend reports as `"BLAS,..."`, not bare `"CPU"` (confirmed via
[ggml-org/llama.cpp#25547](https://github.com/ggml-org/llama.cpp/issues/25547)).
The setup run also hit real `apt`/dpkg lock permission errors while
installing OpenBLAS dev headers. Taken together, this strongly suggests
OpenBLAS was configured but never actually linked in — meaning the
Phase 1 performance numbers below were achieved on llama.cpp's native
CPU backend alone. This isn't necessarily bad news (the result already
clears the ADTC target without acceleration), but any earlier claim of
OpenBLAS providing a speedup should be treated as unverified.

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
| Throughput (text-generation, batch=1) | ≥15 TPS | **17.69 TPS** (mean of 3 runs) | ✅ Pass (+18%) |
| Peak RAM | ≤7GB | **3.65 GB** | ✅ Pass (3.35GB headroom) |

**Variance note:** the 17.69 TPS figure is a mean across 3 benchmark runs
with meaningful spread (stddev 3.00; individual runs measured 20.16, 18.56,
and 14.36 TPS). The lowest individual run (14.36 TPS) falls *below* the 15
TPS target — the mean passes, but not every run would. Separately,
prompt-processing throughput at batch=1 measured 15.54 TPS (stddev 0.81) —
also passing, but with a much smaller margin (+3.6%) than the
text-generation figure typically cited. Both are legitimate readings of
"throughput" depending on which ADTC scores against; worth flagging both
figures rather than only the more favorable one.

Full results: `benchmarks/results/performance_metrics.json`,
`benchmarks/llama_bench_raw.json` (per-run data).

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
| Avg temp | — | **87.1°C** (corrected — see note below) | — |
| Max temp | <80°C (target), <85°C (disqualification) | **98°C** | ❌ Exceeds both |
| Sustained >80°C | avoid | ~211-217 seconds (out of ~302s total) | ❌ |

**Correction:** an earlier version of this report incorrectly stated the
average temperature as 63.4°C (a transcription error made while summarizing
results). The correct figure, independently re-verified against the raw
`thermal_logs_windows.txt` data and matching `thermal_summary_windows.json`
exactly, is **87.08°C** — meaning the CPU spent the *majority* of the
5-minute test above both the 80°C target and the 85°C disqualification
threshold, not just briefly spiking above it. This is a materially more
serious result than originally reported and should be treated accordingly.

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

**Status:** 🔶 In progress (Tasks 2.1–2.3 complete, Task 2.4 outstanding)

### Task 2.1 — System Prompt Variants

Three system prompt variants were drafted — **conservative**, **balanced**,
and **aggressive** — trading off strictness of the structured-output format
against flexibility for edge cases (e.g. symptoms that don't map cleanly to
a known disease). All three were verified to stay under the 500-token
budget using the real Phi-3 tokenizer (`llama-tokenize`), not an estimate.

### Task 2.2 — Evaluation Dataset

The evaluation set was extracted from `toufiqmusah/GhanaAgricVQA-Dataset`.
The dataset viewer's preview initially suggested `disease_labels` was a
nested dict, which turned out to be a misrepresentation — the real schema,
confirmed empirically by loading actual rows, is:

- `answer_text`: plain string
- `disease_labels`: a flat list (not nested)

This matters because `src/evaluator.py`'s parsing logic is written against
the real flat-list schema; code written against the dataset-viewer's
apparent structure would have silently mismatched at evaluation time.

### Task 2.3 — Evaluator

`src/evaluator.py` is complete and implements:

- **BLEU-4** via `sacrebleu`, normalized to a 0–1 range
- **Semantic similarity** via `all-MiniLM-L6-v2` sentence embeddings
- **Disease name normalization** — strips crop-name prefixes (e.g.
  `Corn_`, `Pepper_`) before comparing predicted vs. reference disease
  labels, so a correct diagnosis isn't penalized for prefix mismatch
- **Question-type-aware structure compliance** — checks that
  Identification/Treatment/Prevention questions get responses matching
  the expected section structure for that question type

### Task 2.4 — Batch Evaluation Runner

**Not yet complete.** The `PrompEngine` batch evaluation runner (which will
run all three prompt variants from Task 2.1 against the Task 2.2 dataset
using the Task 2.3 evaluator, and report comparative scores) is the next
piece of work, along with finalizing `requirements.txt` for the evaluation
dependencies it introduces.

### Outstanding thermal risk (carried over from Phase 1)

The Task 1.4 sustained-load thermal finding (87.1°C avg, 98°C peak,
exceeding both the 80°C target and 85°C disqualification threshold) has
not yet been mitigated or re-tested on bare-metal hardware. This remains
open and should not be considered resolved by Phase 2 progress.

## Phase 3: UI & Deployment

**Status:** 🔶 In progress

The Gradio interface (`ui/app.py`) has been visually restyled to a dark
navy/lime aesthetic (matching an internal design reference,
`index.html`), replacing an earlier slate/emerald theme. This was a
visual-only pass — the diagnosis pipeline, bilingual (English/Twi)
toggle, and Ollama integration in `src/ollama_client.py` were not
modified as part of the restyle.

The Phase 1 thermal mitigation (rate-limiting/cooldown between
consecutive UI-triggered inference calls) discussed above has **not**
yet been implemented in `ui/app.py` and remains outstanding.