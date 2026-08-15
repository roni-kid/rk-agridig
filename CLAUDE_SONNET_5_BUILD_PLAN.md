# RK AgriDig — Claude Sonnet 5 Build Plan

**Goal:** Build a complete, working on-device crop disease diagnostic system for ADTC 2026 Gate 1 submission (11 days).

**Constraints:**
- Solo developer (you)
- 11 days to Gate 1 (Aug 14–25)
- 8GB RAM target hardware
- Production-ready code (not a prototype)
- ADTC scoring: 50% accuracy, 30% speed, 20% memory efficiency

---

## Phase 1: Model Selection & Profiling (Days 3–4, 2 days)

### Deliverables
- [ ] Phi-3-mini GGUF Q4_K_M downloaded locally
- [ ] llama.cpp compiled for Ubuntu 22.04
- [ ] llama-bench profiling results (TPS, latency, RAM)
- [ ] Confirmation: model fits 7GB RAM budget
- [ ] Decision: proceed vs. pivot to smaller model

### Claude Sonnet 5 Tasks

#### Task 1.1: Download & Setup Script
**Prompt:**
```
Create a Python script `models/download_model.py` that:
1. Downloads Phi-3-mini GGUF Q4_K_M from HuggingFace (microsoft/Phi-3-mini-4k-instruct)
2. Saves to `models/phi3_mini_4k_instruct.gguf`
3. Verifies MD5 checksum
4. Reports file size and download time
5. Handles network interruptions with retry logic

Use the `huggingface_hub` library. Output should be production-ready Python.
```

#### Task 1.2: llama.cpp Compilation Script
**Prompt:**
```
Create a bash script `setup.sh` that:
1. Clones llama.cpp from GitHub (https://github.com/ggml-org/llama.cpp)
2. Compiles with CMake for CPU-only optimization (OpenBLAS for matrix ops)
3. Installs llama-bench tool for benchmarking
4. Tests compilation with a simple inference command
5. Outputs compilation success/errors clearly

Target: Ubuntu 22.04 LTS with GCC 11+
Include comments for debugging if compilation fails.
```

#### Task 1.3: Benchmarking Script
**Prompt:**
```
Create `benchmarks/run_profiler.sh` that:
1. Runs llama-bench on the Phi-3-mini GGUF model
2. Tests with prompt_processing (pp) and text_generation (tg) separately
3. Measures tokens-per-second (TPS) at different batch sizes (1, 8, 32)
4. Measures peak RAM usage during inference
5. Outputs results in JSON format to `benchmarks/results/performance_metrics.json`
6. Include CPU thread count optimization (-t flag)

Key metrics to capture:
- prompt_processing_per_token_ms
- generation_per_token_ms
- peak_ram_usage_gb
- throughput_tokens_per_second
- model_size_gb

Include error handling for missing model file.
```

#### Task 1.4: Thermal Monitor
**Prompt:**
```
Create `benchmarks/thermal_monitor.py` that:
1. Uses lm-sensors to read CPU core/package temperature
2. Runs inference on the model in a loop for 5+ minutes
3. Logs temperature every 5 seconds to `benchmarks/results/thermal_logs.txt`
4. Alerts if temperature exceeds 85°C (ADTC disqualification threshold)
5. Outputs average, max, min temperature at end
6. Includes optional fan/CPU frequency governor tuning recommendations

Dependencies: psutil, subprocess for lm-sensors

This directly impacts ADTC scoring: thermal throttling = -10 points.
```

### Success Criteria
- [ ] Phi-3-mini GGUF loads in <30 seconds
- [ ] Peak RAM ≤ 7GB (ideally ≤ 5GB for headroom)
- [ ] TPS ≥ 15 tokens/second
- [ ] CPU temperature stays <80°C during 5-min inference
- [ ] All benchmarks logged to JSON for ADTC submission

---

## Phase 2: Prompt Engineering & Evaluation (Days 5–6, 2 days)

### Deliverables
- [ ] System prompt optimized for crop disease diagnosis
- [ ] 20–30 Q&A evaluation set extracted from GhanaAgricVOA
- [ ] Eval script calculating BLEU + semantic similarity
- [ ] Target: 85%+ accuracy on disease identification questions
- [ ] PROMPTS.md log of all AI usage (per ADTC rules)

### Claude Sonnet 5 Tasks

#### Task 2.1: System Prompt Development
**Prompt:**
```
Design a system prompt for Phi-3-mini that optimizes for crop disease diagnosis accuracy. 

The model will answer farmer questions like:
- "What disease is affecting my maize?"
- "My pepper has brown spots. How do I treat it?"
- "How do I prevent fungal diseases in tomato?"

Requirements:
1. Make the model act as an expert agricultural advisor
2. Ground responses in the GhanaAgricVOA knowledge base (maize, pepper, tomato diseases)
3. Enforce structured output: [Disease Name] → [Symptoms] → [Treatment] → [Prevention]
4. Encourage farmer-friendly language (avoid jargon)
5. Add safety guardrails: "I'm based on crop data from Ghana; consult local experts for confirmation"
6. Keep prompts under 500 tokens (quantized model constraint)

Provide 3 variations (conservative, balanced, aggressive) with tradeoffs explained.
```

#### Task 2.2: Evaluation Dataset Creation
**Prompt:**
```
Create `scripts/create_eval_dataset.py` that:
1. Loads GhanaAgricVOA dataset from HuggingFace
2. Extracts 20–30 diverse Q&A pairs from test split:
   - At least 5 identification questions ("What disease is this?")
   - At least 5 treatment questions ("How do I fix it?")
   - At least 5 prevention questions ("How do I prevent this?")
   - Spread across maize, pepper, tomato (5–7 per crop)
3. Saves to `data/eval_samples.json` with structure:
   {
     "sample_id": "str",
     "crop": "str",
     "question": "str",
     "reference_answer": "str",
     "disease": "str",
     "question_type": "identification|treatment|prevention",
     "language": "en"
   }
4. Prints summary statistics (# by disease, by crop, by type)

This eval set is the gold standard for measuring Sacc (accuracy).
```

#### Task 2.3: Evaluation Metrics Script
**Prompt:**
```
Create `src/evaluator.py` that:
1. Takes eval_samples.json + model outputs
2. Calculates:
   - BLEU-4 score (compares generated vs. reference answers)
   - Semantic similarity (cosine distance between embeddings, use sentence-transformers)
   - Disease identification accuracy (exact match on disease name)
   - Structured output compliance (% of responses with all 4 sections)
3. Outputs JSON report: `benchmarks/results/accuracy_metrics.json`
   {
     "overall_accuracy": 0.87,
     "accuracy_by_question_type": {...},
     "accuracy_by_crop": {...},
     "bleu_score": 0.72,
     "semantic_similarity": 0.81,
     "structured_compliance": 0.95
   }
4. Includes per-sample results for debugging

This directly measures Sacc (50% of final score).
```

#### Task 2.4: Prompt Testing Loop
**Prompt:**
```
Create `src/prompt_engineer.py` with:
1. Class PrompEngine that loads the system prompt
2. Method generate(question) → calls Ollama API with system prompt + question
3. Method evaluate_batch(eval_samples.json) → runs evaluator on all samples
4. Iteration workflow:
   - Load system prompt v1
   - Run on eval set
   - Calculate metrics
   - Log changes to `PROMPTS.md` (per ADTC requirement)
   - Suggest prompt tweaks based on low-accuracy samples
5. Template for trying 3 prompt variations

Include examples of low-accuracy outputs to debug why disease identification failed.
```

### Success Criteria
- [ ] Eval set has 20–30 Q&A pairs balanced across diseases/crops
- [ ] Accuracy on identification questions ≥ 85%
- [ ] BLEU score ≥ 0.70
- [ ] All prompts logged to PROMPTS.md with timestamps
- [ ] No hallucination or false disease claims in sample outputs

---

## Phase 3: UI & Deployment (Days 7–8, 2 days)

### Deliverables
- [ ] Gradio web interface running locally
- [ ] Screenshot proof of working UI
- [ ] Ollama server setup guide
- [ ] End-to-end integration test (question → model → answer → UI)

### Claude Sonnet 5 Tasks

#### Task 3.1: Gradio UI
**Prompt:**
```
Create `ui/app.py` — a Gradio web interface for RK AgriDig that:

1. Input: Textbox for farmer to describe crop symptoms
   - Placeholder: "E.g., My maize leaves have brown spots..."
   - Max length: 500 chars

2. Dropdown: Crop selection (Maize, Pepper, Tomato)
   - Default: Maize

3. Radio: Question type (Identification, Treatment, Prevention)
   - Default: Identification

4. Output: 
   - Structured response with disease name highlighted
   - Treatment steps as numbered list
   - Prevention tips as bullet points
   - Confidence score (from model if available)

5. History: Optional chat-like interface showing past queries

6. Design:
   - Theme: Green + blue color scheme (matching RK AgriDig branding)
   - Mobile-friendly layout
   - Simple, no advanced features
   - Loading indicator while model processes
   - Error handling for model timeouts

7. Connect to Ollama API (ollama.ai:11434)

Gradio code should be production-ready, minimal dependencies.
Include CSS styling if needed.
```

#### Task 3.2: Ollama Integration Script
**Prompt:**
```
Create `src/ollama_client.py` that:
1. Connects to Ollama server (localhost:11434)
2. Loads Phi-3-mini GGUF model (with retry logic)
3. Handles inference requests with system prompt + user question
4. Returns structured response (disease, treatment, prevention)
5. Includes timeout handling (inference should complete in <30 seconds)
6. Logs all requests/responses for debugging
7. Provides fallback behavior if Ollama unavailable

Methods:
- load_model(model_name) → bool
- infer(question, crop, question_type) → str
- health_check() → bool

This is the bridge between UI and model.
```

#### Task 3.3: Docker (Optional but Recommended)
**Prompt:**
```
Create `Dockerfile` and `docker-compose.yml` for easy deployment:

1. Dockerfile:
   - Base: ubuntu:22.04
   - Install: Python 3.11, llama.cpp, Ollama
   - Copy: model files, source code
   - Expose: port 7860 (Gradio) + 11434 (Ollama)
   - Entrypoint: Run both services

2. docker-compose.yml:
   - Service 1: Ollama (volumes for model cache)
   - Service 2: Gradio app (depends on Ollama)
   - Network: internal only (no external calls)
   - Resource limits: 8GB RAM max

This allows one-command deployment: `docker-compose up`
```

### Success Criteria
- [ ] Gradio UI launches on `http://localhost:7860`
- [ ] Sample query returns disease diagnosis in <10 seconds
- [ ] UI is responsive and mobile-friendly
- [ ] Screenshot shows working interface
- [ ] No errors in browser console

---

## Phase 4: Documentation & Submission (Days 9–10, 2 days)

### Deliverables
- [ ] GitHub repo pushed with all code
- [ ] REPORT.md (technical report for judges)
- [ ] PROMPTS.md (AI usage log)
- [ ] Demo screenshots (UI + benchmarks)
- [ ] 2-minute demo video
- [ ] DevPost form fully filled

### Claude Sonnet 5 Tasks

#### Task 4.1: Technical Report (REPORT.md)
**Prompt:**
```
Create `REPORT.md` — the ADTC technical report (~2000–3000 words) covering:

1. **Problem Definition**
   - Smallholder farmer needs (no internet, no API costs)
   - Current gap in solutions
   - Why this matters in West Africa

2. **Solution Overview**
   - What is RK AgriDig?
   - How it works (high-level)
   - Key differentiators vs. cloud AI

3. **Constraints & Design**
   - 8GB RAM budget → implications for model selection
   - No discrete GPU → CPU optimization needed
   - Offline requirement → no cloud calls
   - Design alternatives considered (why Phi-3-mini over alternatives)

4. **Technical Implementation**
   - Architecture diagram (ASCII or Mermaid)
   - Model selection rationale
   - Quantization strategy (why Q4_K_M)
   - Prompt engineering approach
   - Evaluation methodology

5. **Performance Results**
   - Sacc (accuracy): [your score]
   - Sperf (throughput): [TPS]
   - Seff (efficiency): [RAM score]
   - Thermal: [max CPU temp]
   - Comparison to ADTC targets

6. **Challenges & Solutions**
   - What went wrong? How did you fix it?
   - Thermal throttling mitigation
   - RAM optimization techniques
   - Accuracy bottlenecks

7. **African Use Case**
   - Grounded in GhanaAgricVOA dataset
   - Real farmer scenarios
   - Language support (English + Twi)
   - Validation from real maize/pepper/tomato farmers (if possible)

8. **Limitations & Future Work**
   - Current constraints
   - Potential improvements (fine-tuning, ensemble, retrieval-augmented generation)

Include figures: benchmarks, accuracy plots, architecture diagram.
Cite all tools: llama.cpp, Phi-3, GhanaAgricVOA, etc.
```

#### Task 4.2: AI Usage Log (PROMPTS.md)
**Prompt:**
```
Create `PROMPTS.md` — log of all AI prompts used in building this project (ADTC requirement).

Format (for each AI-generated component):
\`\`\`markdown
## [Component Name]
**Date:** YYYY-MM-DD
**Tool:** Claude/ChatGPT/Gemini
**Purpose:** [What was built]
**Prompt Given:**
[Full prompt text]
**Output Used:**
[Brief description of output, not full code]
**Modifications Made:**
[Any changes to original output]
\`\`\`

Include logs for:
- System prompt generation
- Code generation (UI, evaluator, benchmarking)
- Documentation writing
- Anything else AI-assisted

This is transparent to judges about AI usage in your build.
```

#### Task 4.3: Demo Video Script
**Prompt:**
```
Create `scripts/generate_demo_video.md` — a script/storyboard for the 2-minute demo video:

**Scene 1 (0:00–0:20):** The Problem
- Show title slide: "RK AgriDig"
- Narrate: "Farmers in Ghana face a problem: when crops get sick, they have no way to diagnose it."
- Show image of diseased crop

**Scene 2 (0:20–0:50):** The Solution
- Show laptop + Gradio UI
- Narrate: "RK AgriDig brings AI diagnosis directly to your laptop. No internet. No API bills."
- Demo typing a question: "My maize leaves have brown spots with yellow halos"

**Scene 3 (0:50–1:30):** Live Demo
- Show model processing (loading indicator)
- Show response: [Disease Name] → [Symptoms] → [Treatment] → [Prevention]
- Point out disease identification
- Point out treatment advice
- Point out prevention steps
- Narrate: "Disease identified in seconds. Actionable advice for farmers."

**Scene 4 (1:30–1:50):** Benchmarks
- Show performance metrics:
  - TPS: X tokens/second
  - RAM: X GB used
  - Accuracy: X%
- Narrate: "Optimized for the hardware African farmers actually own."

**Scene 5 (1:50–2:00):** Call to Action
- Title slide: "Built for Africa. By an African engineer."
- Mention: GhanaAgricVOA dataset, on-device AI, offline-first
- End: GitHub link

**Recording Tips:**
- Use OBS or built-in screen recorder
- Clear audio, speak slowly
- Crop out personal info
- Save as MP4, <100MB

Duration: Exactly 2 minutes (120 seconds)
```

#### Task 4.4: README Screenshots Guide
**Prompt:**
```
Create a checklist for screenshots to include in GitHub repo:

**Screenshot 1: UI Home**
- Gradio interface with empty fields
- Show input textbox, crop dropdown, question type radio
- Save to `outputs/demo_screenshots/01_ui_home.png`

**Screenshot 2: Sample Query**
- Show a filled-in query (e.g., pepper plant with brown spots)
- Highlight the crop dropdown + question type selection
- Save to `outputs/demo_screenshots/02_sample_query.png`

**Screenshot 3: Model Response**
- Show the full response with disease name, treatment, prevention
- Show response time and confidence score
- Save to `outputs/demo_screenshots/03_model_response.png`

**Screenshot 4: Benchmarks**
- Show performance metrics from `benchmarks/results/performance_metrics.json`
- TPS, RAM, accuracy, thermal
- Save to `outputs/demo_screenshots/04_benchmarks.png`

**Screenshot 5: Terminal Output**
- Show setup.sh running, model downloading, benchmarks completing
- Save to `outputs/demo_screenshots/05_terminal.png`

All images should be:
- 1200×800px minimum (3:2 aspect ratio)
- PNG format
- <1MB each
- Include the RK AgriDig branding/colors
```

### Success Criteria
- [ ] REPORT.md is complete, professional, 2000+ words
- [ ] PROMPTS.md logs all AI usage with dates/purposes
- [ ] Demo video is exactly 2 minutes, MP4 format, <100MB
- [ ] 5+ screenshots captured and organized
- [ ] GitHub repo is public, all code pushed
- [ ] DevPost form fully filled with thumbnails + links

---

## Phase 5: Buffer & Final Review (Day 11, 1 day)

### Final Checklist
- [ ] Run ADTC profiler on entire submission
- [ ] Verify all metrics: Sacc, Sperf, Seff, Thermal
- [ ] Test UI one more time
- [ ] Check GitHub links in DevPost
- [ ] Verify video duration = exactly 2 minutes
- [ ] Review REPORT.md for typos/clarity
- [ ] Confirm all files <5MB for images, <500MB for repo
- [ ] **Submit to DevPost by Aug 25, 6:45am GMT**

---

## Summary: What Claude Sonnet 5 Builds

| Phase | Days | What Gets Built | Claude Sonnet 5 Role |
|-------|------|-----------------|---------------------|
| 1 | 3–4 | Model setup + benchmarking | Write Python scripts for download, profiling, thermal monitoring |
| 2 | 5–6 | Prompt engineering + eval | Design system prompts, build evaluation metrics, iterate on accuracy |
| 3 | 7–8 | UI + deployment | Create Gradio app, Ollama integration, Docker setup |
| 4 | 9–10 | Documentation | Write REPORT.md, PROMPTS.md, demo script, screenshot guide |
| 5 | 11 | Final review | Verify everything works, troubleshoot issues |

---

## Key Constraints to Always Honor

1. **Model must fit 7GB RAM** — No exceptions, this is the ADTC spec
2. **Offline requirement** — Zero cloud API calls (no OpenAI, no Google, no Anthropic APIs in production)
3. **On-device inference only** — All compute happens on the farmer's laptop
4. **GhanaAgricVOA grounding** — Responses based on real Ghanaian crop data, not generic agriculture
5. **ADTC scoring formula** — Sacc (50%) + Sperf (30%) + Seff (20%) − Pthermal
6. **Thermal safety** — Keep CPU <85°C to avoid -10 point penalty

---

## Success Definition

**Gate 1 submission (Aug 25) = Complete working system with:**
- ✅ Offline model running on 8GB laptop
- ✅ Gradio UI for farmer interaction
- ✅ Benchmarked performance (Sacc, Sperf, Seff, Thermal)
- ✅ GitHub repo with docs + code
- ✅ 2-min demo video
- ✅ Technical report explaining design choices
- ✅ AI usage log (PROMPTS.md)

**If Gate 1 passes → Semicamp (Sept 22) and Finals (Oct 17) with:**
- Fine-tuning on more data (if time)
- Expanded disease coverage beyond maize/pepper/tomato
- Live pitch + Q&A from judges

---

**Ready to start Day 3? Let's begin with Task 1.1: Model download script.**
