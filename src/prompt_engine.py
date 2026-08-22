"""
prompt_engine.py — RK AgriDig Phase 2, Task 2.4

Batch evaluation runner. Loads data/eval_samples.json, runs every sample
through the local Ollama model once per prompt variant defined in
src/prompt_engineer.py, and writes one predictions file per variant to
benchmarks/results/predictions_<variant>.json in the exact
{sample_id: generated_text} format src/evaluator.py expects.

This module is the seam between two different casing conventions that
otherwise don't line up if you wire them together directly:

  - data/eval_samples.json (Task 2.2) uses lowercase, singular values:
        question_type ∈ {"identification", "action", "prevention"}
        crop           ∈ {"maize", "pepper", "tomato"}

  - OllamaClient.infer() (src/ollama_client.py) expects Title-case values
    matching the Gradio UI's dropdown/radio choices, and "Treatment" is
    the accepted question_type string -- there is no "Action":
        question_type ∈ {"Identification", "Treatment", "Prevention"}
        crop           ∈ {"Maize", "Pepper", "Tomato"}

  - src/evaluator.py's check_structure_compliance() then expects the
    *original* eval_samples.json lowercase casing back again, since it
    reads question_type straight off the sample dict, not off anything
    PromptEngine produces.

Get any of these three conversions backwards and you get a silent,
hard-to-notice bug: predictions still generate, evaluator.py still runs,
numbers still come out -- they're just measuring the wrong bucket (e.g.
every "action" sample scored against the "prevention" structure check).
_to_client_question_type() and _to_client_crop() are the only place this
translation happens; nothing else in this file or in evaluator.py should
need to know about the casing mismatch.

System prompt variant selection: OllamaClient.infer() has its own
hardcoded SYSTEM_PROMPT baked into the module. To actually A/B the three
variants from prompt_engineer.py (conservative / balanced / aggressive),
this runner monkeypatches ollama_client.SYSTEM_PROMPT per variant for the
duration of each run rather than forking OllamaClient -- see
_run_variant() for why, and the module docstring note on that tradeoff.

Usage:
    python3 src/prompt_engine.py
    python3 src/prompt_engine.py --variant balanced
    python3 src/prompt_engine.py --eval-set data/eval_samples.json --limit 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src import ollama_client as ollama_client_module  # noqa: E402
from src.ollama_client import OllamaClient  # noqa: E402
from src.prompt_engineer import PROMPTS  # noqa: E402

DEFAULT_EVAL_SET = REPO_ROOT / "data" / "eval_samples.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "benchmarks" / "results"

# --- Casing bridge: eval_samples.json (lowercase) -> OllamaClient (Title-case) ---
# "action" has no direct Title-case counterpart in OllamaClient/prompt_engineer.py
# -- "Treatment" is the closest match (see SYSTEM_PROMPT's structure and
# num_predict_by_type in ollama_client.py, which both use "Treatment").
QUESTION_TYPE_TO_CLIENT = {
    "identification": "Identification",
    "action": "Treatment",
    "prevention": "Prevention",
}
CROP_TO_CLIENT = {
    "maize": "Maize",
    "pepper": "Pepper",
    "tomato": "Tomato",
}


def _to_client_question_type(question_type: str) -> str:
    mapped = QUESTION_TYPE_TO_CLIENT.get(question_type)
    if mapped is None:
        raise ValueError(
            f"Unknown question_type {question_type!r} in eval sample -- "
            f"expected one of {sorted(QUESTION_TYPE_TO_CLIENT)}. "
            f"Add it to QUESTION_TYPE_TO_CLIENT if this is a legitimate new type."
        )
    return mapped


def _to_client_crop(crop: str) -> str:
    mapped = CROP_TO_CLIENT.get(crop)
    if mapped is None:
        raise ValueError(
            f"Unknown crop {crop!r} in eval sample -- "
            f"expected one of {sorted(CROP_TO_CLIENT)}. "
            f"Add it to CROP_TO_CLIENT if this is a legitimate new crop."
        )
    return mapped


def load_eval_samples(eval_set_path: Path) -> list[dict]:
    if not eval_set_path.exists():
        print(f"ERROR: eval set not found at {eval_set_path}", file=sys.stderr)
        sys.exit(1)
    with open(eval_set_path) as f:
        data = json.load(f)
    return data["samples"]


def _run_variant(
    variant_name: str,
    system_prompt: str,
    samples: list[dict],
    client: OllamaClient,
    limit: int | None,
) -> tuple[dict[str, str], list[dict]]:
    """
    Runs one prompt variant across all (or --limit) eval samples.

    Monkeypatches ollama_client.SYSTEM_PROMPT for the duration of this call
    rather than modifying OllamaClient itself, since SYSTEM_PROMPT is baked
    into the payload construction inside infer() as a module-level constant,
    not an __init__ parameter -- changing that would mean editing the
    already-working, separately-reviewed ollama_client.py just to make it
    swappable for this one offline evaluation script. Restored in a
    try/finally so a crash mid-run can't leave the module patched for
    anything importing it afterward (e.g. the Gradio app, if run in the
    same process).
    """
    original_prompt = ollama_client_module.SYSTEM_PROMPT
    ollama_client_module.SYSTEM_PROMPT = system_prompt

    predictions: dict[str, str] = {}
    per_sample_timing: list[dict] = []

    try:
        run_samples = samples[:limit] if limit else samples
        total = len(run_samples)

        for i, sample in enumerate(run_samples, start=1):
            sid = sample["sample_id"]
            question = sample["question"]

            try:
                client_crop = _to_client_crop(sample["crop"])
                client_qtype = _to_client_question_type(sample["question_type"])
            except ValueError as e:
                print(f"  [{i}/{total}] SKIP {sid}: {e}", file=sys.stderr)
                predictions[sid] = ""
                continue

            print(f"  [{i}/{total}] {sid} (crop={client_crop}, type={client_qtype})...", end=" ", flush=True)
            start = time.time()
            result = client.infer(question=question, crop=client_crop, question_type=client_qtype)
            elapsed = time.time() - start

            if result.success:
                predictions[sid] = result.raw_text
                print(f"ok ({elapsed:.1f}s, confidence={result.structured_confidence:.2f})")
            else:
                predictions[sid] = ""
                print(f"FAILED: {result.error}")

            per_sample_timing.append({
                "sample_id": sid,
                "elapsed_seconds": round(elapsed, 2),
                "success": result.success,
                "structured_confidence": result.structured_confidence,
            })

    finally:
        ollama_client_module.SYSTEM_PROMPT = original_prompt

    return predictions, per_sample_timing


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run eval_samples.json through the local Ollama model for each prompt variant."
    )
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variant", choices=list(PROMPTS.keys()), default=None,
        help="Run only this variant. Default: run all three (conservative, balanced, aggressive).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Only run the first N samples (useful for a fast smoke test before a full run).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("RK AgriDig — Prompt Engine Batch Runner (Task 2.4)")
    print("=" * 70)

    samples = load_eval_samples(args.eval_set)
    print(f"Loaded {len(samples)} eval samples from {args.eval_set}")

    client = OllamaClient()
    print("Checking Ollama server health...")
    if not client.health_check():
        print(
            "ERROR: Ollama server not reachable at http://localhost:11434\n"
            "       Start it with: ollama serve",
            file=sys.stderr,
        )
        return 1
    if not client.load_model():
        print(
            f"ERROR: Model '{client.model_name}' not found on the Ollama server.\n"
            f"       Run: ollama create {client.model_name} -f Modelfile",
            file=sys.stderr,
        )
        return 1
    print("Server + model OK.\n")

    variants_to_run = [args.variant] if args.variant else list(PROMPTS.keys())
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_summary = {}
    for variant_name in variants_to_run:
        print(f"--- Running variant: {variant_name} ---")
        system_prompt = PROMPTS[variant_name]
        predictions, timing = _run_variant(variant_name, system_prompt, samples, client, args.limit)

        pred_path = args.output_dir / f"predictions_{variant_name}.json"
        with open(pred_path, "w") as f:
            json.dump(predictions, f, indent=2)

        timing_path = args.output_dir / f"timing_{variant_name}.json"
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=2)

        n_ok = sum(1 for t in timing if t["success"])
        avg_elapsed = round(sum(t["elapsed_seconds"] for t in timing) / len(timing), 2) if timing else 0.0
        run_summary[variant_name] = {
            "n_samples_run": len(timing),
            "n_succeeded": n_ok,
            "n_failed": len(timing) - n_ok,
            "avg_elapsed_seconds": avg_elapsed,
            "predictions_path": str(pred_path),
        }
        print(f"  -> wrote {pred_path}")
        print(f"  -> {n_ok}/{len(timing)} succeeded, avg {avg_elapsed}s/sample\n")

    summary_path = args.output_dir / "run_summary.json"
    with open(summary_path, "w") as f:
        json.dump(run_summary, f, indent=2)

    print("=" * 70)
    print("Done. Run summary:")
    for variant_name, s in run_summary.items():
        print(f"  {variant_name:14s} {s['n_succeeded']}/{s['n_samples_run']} ok, avg {s['avg_elapsed_seconds']}s")
    print(f"\nNext step -- score each variant, e.g.:")
    for variant_name in variants_to_run:
        print(f"  python3 src/evaluator.py --predictions benchmarks/results/predictions_{variant_name}.json "
              f"--output benchmarks/results/accuracy_{variant_name}.json")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())