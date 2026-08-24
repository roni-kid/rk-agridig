# PROMPTS.md — AI Usage Log

**Project:** RK AgriDig — Offline Crop Disease AI for Ghanaian Farmers
**Competition:** Africa Deep Tech Challenge (ADTC) 2026
**Author:** Aaron Baidoo (RoniKid), GCTU Ghana
**Collaborator:** Firdaus Kudus

This file documents all AI-assisted components of RK AgriDig per ADTC transparency requirements. All AI outputs were reviewed, tested, and modified before integration.

---

## Phase 1: Model Selection & Profiling

---

### 1.1 — Model Download Script

**Date:** 2026-08-14
**Tool:** Claude (Anthropic)
**Purpose:** Create `models/download_model.py` to download Phi-3-mini GGUF from HuggingFace with checksum verification and retry logic.

**Prompt Given:**
```
Create a Python script models/download_model.py that:
1. Downloads Phi-3-mini GGUF Q4_K_M from HuggingFace (microsoft/Phi-3-mini-4k-instruct)
2. Saves to models/phi3_mini_4k_instruct.gguf
3. Verifies SHA256 checksum against HuggingFace Hub API metadata
4. Reports file size and download time
5. Handles network interruptions with retry logic
Use the huggingface_hub library. Output should be production-ready Python.
```

**Output Used:** Full script adopted with modifications.

**Modifications Made:**
- Changed checksum from MD5 to SHA256 (HuggingFace does not publish MD5 for this file)
- Added dynamic checksum fetching from Hub API rather than hardcoded value, to stay resilient to upstream changes
- Added progress reporting at 10% intervals

---

### 1.2 — llama.cpp Compilation Script

**Date:** 2026-08-14
**Tool:** Claude (Anthropic)
**Purpose:** Create `setup.sh` to clone, compile, and verify llama.cpp on Ubuntu 22.04.

**Prompt Given:**
```
Create a bash script setup.sh that:
1. Clones llama.cpp from GitHub
2. Compiles with CMake for CPU-only optimization (OpenBLAS for matrix ops)
3. Installs llama-bench tool for benchmarking
4. Tests compilation with a simple inference command
5. Outputs compilation success/errors clearly
Target: Ubuntu 22.04 LTS with GCC 11+. Include comments for debugging.
```

**Output Used:** Script structure adopted; CMake flags adjusted.

**Modifications Made:**
- OpenBLAS was configured but confirmed non-functional in practice (WSL2 dpkg/apt lock errors during header install, and llama-bench output confirmed bare "CPU" backend with no BLAS designation). Script retained the flags but this is documented as unverified in REPORT.md.
- Added physical-core detection via `lscpu` for thread-count optimization (16 logical threads caused severe throughput regression; 8 physical cores produced peak TPS)

---

### 1.3 — Performance Profiling Script

**Date:** 2026-08-15
**Tool:** Claude (Anthropic)
**Purpose:** Create `benchmarks/run_profiler.sh` to benchmark TPS, RAM, and latency across batch sizes.

**Prompt Given:**
```
Create benchmarks/run_profiler.sh that:
1. Runs llama-bench on the Phi-3-mini GGUF model
2. Tests prompt_processing and text_generation separately
3. Measures TPS at batch sizes 1, 8, 32
4. Measures peak RAM usage during inference
5. Outputs results to benchmarks/results/performance_metrics.json
6. Include CPU thread count optimization
```

**Output Used:** Adopted with thread-count correction.

**Modifications Made:**
- Replaced hardcoded `-t $(nproc)` with `-t $(lscpu | grep "^Core(s) per socket" | awk '{print $NF}')` after discovering 16-thread setting halved throughput vs. 8 physical cores
- Added 3-run averaging with stddev reporting after observing high variance (14.36–20.16 TPS range across runs)
- Added concurrent RAM sampling via `/proc/self/status` polling

---

### 1.4 — Thermal Monitor

**Date:** 2026-08-15
**Tool:** Claude (Anthropic)
**Purpose:** Create `benchmarks/thermal_monitor.py` for CPU temperature logging during sustained inference.

**Prompt Given:**
```
Create benchmarks/thermal_monitor.py that:
1. Uses lm-sensors to read CPU temperature
2. Runs inference in a loop for 5+ minutes
3. Logs temperature every 5 seconds
4. Alerts if temperature exceeds 85°C (ADTC threshold)
5. Outputs average, max, min at end
```

**Output Used:** Core structure adopted.

**Modifications Made:**
- Added WSL2 detection and graceful degradation (WSL2 kernel has no hardware sensor access; script reports "sensors unavailable" rather than crashing or fabricating data)
- Wrote companion `benchmarks/thermal_monitor_windows.ps1` to poll real CPU temperature from Windows host via WMI (`MSAcpi_ThermalZoneTemperature`) while inference runs in WSL2
- Test revealed 98°C peak under continuous back-to-back load; documented in REPORT.md with caveats on test conditions

---

## Phase 2: Prompt Engineering & Evaluation

---

### 2.1 — System Prompt Variants

**Date:** 2026-08-16
**Tool:** Claude (Anthropic)
**Purpose:** Design three system prompt variants (conservative, balanced, aggressive) for crop disease diagnosis.

**Prompt Given:**
```
Design a system prompt for Phi-3-mini that optimizes for crop disease diagnosis accuracy.
The model will answer farmer questions about maize, pepper, and tomato diseases.
Requirements:
1. Expert agricultural advisor persona
2. Grounded in GhanaAgricVQA knowledge base
3. Enforce structured output: Disease Name → Symptoms → Treatment → Prevention
4. Farmer-friendly language (avoid jargon)
5. Safety guardrails referencing Ghana crop data
6. Keep prompts under 500 tokens
Provide 3 variations (conservative, balanced, aggressive) with tradeoffs explained.
```

**Output Used:** All three variants drafted; balanced variant selected for primary evaluation.

**Modifications Made:**
- Verified all three variants against the real Phi-3 tokenizer (`llama-tokenize`) to confirm under 500-token budget
- Added crop-specific context injection (crop name passed from UI dropdown into prompt template at inference time)
- Adjusted disease name format to match GhanaAgricVQA label schema (e.g. "Bacterial Spot" not "bacterial_spot")

---

### 2.2 — Evaluation Dataset Creation

**Date:** 2026-08-17
**Tool:** Claude (Anthropic)
**Purpose:** Create `scripts/create_eval_dataset.py` to extract 20–30 Q&A pairs from GhanaAgricVQA test split.

**Prompt Given:**
```
Create scripts/create_eval_dataset.py that:
1. Loads GhanaAgricVQA dataset from HuggingFace (toufiqmusah/GhanaAgricVQA-Dataset)
2. Extracts 20–30 diverse Q&A pairs from test split
3. Balances across identification/treatment/prevention question types
4. Saves to data/eval_samples.json
5. Prints summary statistics
```

**Output Used:** Script structure adopted with schema corrections.

**Modifications Made:**
- Corrected `disease_labels` parsing: dataset viewer suggested nested dict, but actual schema is a flat list. Script rewritten to match real schema confirmed by loading actual rows.
- Fixed dataset name (build plan had a typo: `GhanaAgricVOA` → corrected to `GhanaAgricVQA`)
- Added crop-balance check to ensure maize/pepper/tomato each represented

---

### 2.3 — Evaluation Metrics Script

**Date:** 2026-08-17
**Tool:** Claude (Anthropic)
**Purpose:** Create `src/evaluator.py` with BLEU, semantic similarity, and disease identification accuracy.

**Prompt Given:**
```
Create src/evaluator.py that:
1. Calculates BLEU-4, semantic similarity, disease identification accuracy
2. Outputs JSON report to benchmarks/results/accuracy_metrics.json
3. Includes per-sample results for debugging
```

**Output Used:** Adopted with normalization and structure-compliance additions.

**Modifications Made:**
- Added disease name normalization (strips crop-name prefixes like `Corn_`, `Pepper_` before comparing labels)
- Added question-type-aware structure compliance checking
- Normalized BLEU to 0–1 range via sacrebleu
- Switched embedding model to `all-MiniLM-L6-v2` (smaller, runs on 8GB target hardware)

---

## Phase 3: UI & Deployment

---

### 3.1 — Gradio UI

**Date:** 2026-08-18
**Tool:** Claude (Anthropic)
**Purpose:** Create `ui/app.py` — the full Gradio web interface for RK AgriDig.

**Prompt Given:**
```
Create ui/app.py — a Gradio web interface for RK AgriDig with:
- Symptom textbox, crop dropdown, question-type radio
- Structured response output (disease, treatment, prevention)
- Scan history panel
- Dark navy/lime color scheme
- English/Twi toggle
- Loading indicator and error handling
- Connect to Ollama API (localhost:11434)
Gradio 4.x+ compatible. Production-ready.
```

**Output Used:** Two candidate implementations evaluated (internal `app.py` and `fapp.py` from a secondary design pass). Final version is a structured merge of both.

**Modifications Made:**
- Moved `theme=` and `css=` from `gr.Blocks()` constructor to `demo.launch()` (Gradio 6.x breaking change)
- Fixed language toggle: `visible=False` removes elements from DOM in Gradio 6.x; replaced with CSS visually-hidden class to keep trigger buttons in DOM
- Wired `lang_js` active-state class-swap to both language buttons (was dead code)
- Fixed dark-theme bleed from Gradio's internal `.form` wrapper (targeted CSS override)
- Replaced global state variables with `gr.State` to prevent session bleed between concurrent users
- Added amber-bordered disclaimer banners for Twi diagnosis and Twi voice (honest degradation — full Twi AI output deferred to post-Gate 1)
- Added query cooldown mechanism for thermal mitigation (rate-limits consecutive inference calls)
- Replaced inline SVGs with base64-embedded PNGs from `ui/assets/` (resized via Pillow)
- Validated final version via AST parsing and headless Playwright browser test

---

### 3.2 — Ollama Client

**Date:** 2026-08-18
**Tool:** Claude (Anthropic)
**Purpose:** Create `src/ollama_client.py` bridging the UI to the Ollama inference server.

**Prompt Given:**
```
Create src/ollama_client.py that:
1. Connects to Ollama server (localhost:11434)
2. Handles inference requests with system prompt + user question
3. Returns structured response (disease, treatment, prevention)
4. Includes timeout handling and retry logic
5. Logs all requests/responses
6. Provides fallback if Ollama unavailable
```

**Output Used:** Adopted with keep-alive and model name fixes.

**Modifications Made:**
- Added `"keep_alive": -1` to all inference requests (without this, Ollama evicts the model after 5 minutes of idle, causing severe cold-start penalty on every call)
- Corrected model name: Ollama rejects hyphens in model names; `phi3-agridig` → `phi3agridig`
- Added structured response parser for the `Disease Name / Symptoms / Treatment / Prevention` format
- Added health check on UI launch with clear error message if Ollama server is unreachable

---

### 3.3 — Documentation

**Date:** 2026-08-20
**Tool:** Claude (Anthropic)
**Purpose:** Generate `docs/SETUP.md` for native WSL2/Ollama installation path.

**Prompt Given:**
```
Write docs/SETUP.md covering the native Ollama on WSL2 setup path for RK AgriDig.
Docker was abandoned due to repeated environment failures. Document:
- WSL2 prerequisites
- Ollama install and model pull
- venv setup and pip install
- Launching the UI
- Common errors and fixes
```

**Output Used:** Adopted as written with path corrections for Windows filesystem mounts.

**Modifications Made:**
- Corrected all paths to use `/mnt/c/Users/rocks/Documents/...` WSL2 mount syntax
- Added note on model naming: `phi3agridig` not `phi3-agridig`
- Added keep-alive reminder in the "Running inference" section

---

*Log maintained by Aaron Baidoo (RoniKid). All AI-generated code was reviewed, tested, and modified before integration into the project.*
