#!/usr/bin/env python3
"""
create_eval_dataset.py — RK AgriDig Phase 2, Task 2.2

Extracts a balanced evaluation set from the real GhanaAgricVQA-Dataset
(toufiqmusah/GhanaAgricVQA-Dataset on HuggingFace) test split, for use in
accuracy scoring against the deployed model's responses.

CONFIRMED REAL SCHEMA (verified against live dataset viewer, not assumed):
    image, question, answer, question_type, crop, language, sample_id,
    rail_image_id, generation_confidence, validated, question_tw, answer_text_tw

IMPORTANT: `answer` is NOT a plain string. It is a nested object:
    {'text': '<farmer-facing answer>', 'detections': [{'bbox': [...],
     'disease_label': '...', 'severity': '...'}, ...]}
This script extracts `answer['text']` as the reference answer, and derives
a `disease` field from the most common `disease_label` in `detections`
(falls back to null for healthy-plant samples with no detections).

Every sample currently has validated=False and generation_confidence=0.9
uniformly (AI-generated via Qwen3.5-9B per the dataset card, not yet
human-validated) — this is noted in the output metadata for transparency,
since it's relevant to how much to trust these as ground truth.

Usage:
    python3 scripts/create_eval_dataset.py
    python3 scripts/create_eval_dataset.py --n-samples 30 --seed 42
    python3 scripts/create_eval_dataset.py --split test --question-types identification,action,prevention
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print(
        "ERROR: 'datasets' library not installed.\n"
        "Install it with: pip install datasets --break-system-packages\n",
        file=sys.stderr,
    )
    sys.exit(1)


REPO_ID = "toufiqmusah/GhanaAgricVQA-Dataset"
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "eval_samples.json"


def extract_answer_text(answer_field) -> str:
    """
    The 'answer' column is a nested dict {'text': ..., 'detections': [...]}
    on the live dataset, not a plain string as originally assumed in early
    project docs. Handle both shapes defensively in case this changes.
    """
    if isinstance(answer_field, dict):
        return answer_field.get("text", "")
    if isinstance(answer_field, str):
        return answer_field
    return str(answer_field)


def extract_disease_label(answer_field) -> str | None:
    """
    Derive a single representative disease label from the detections list,
    for convenience when building prompts/scoring. Returns None for
    healthy-plant samples (empty detections) rather than a fake label.
    """
    if not isinstance(answer_field, dict):
        return None
    detections = answer_field.get("detections", [])
    if not detections:
        return None
    labels = [d.get("disease_label") for d in detections if d.get("disease_label")]
    if not labels:
        return None
    # Most common label across detections (usually all the same disease,
    # multiple bounding boxes on the same leaf/plant)
    return Counter(labels).most_common(1)[0][0]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract eval set from GhanaAgricVQA-Dataset.")
    parser.add_argument("--split", default="test", choices=["train", "test"],
                         help="Which dataset split to sample from (default: test, to avoid overlap with any future fine-tuning use of train)")
    parser.add_argument("--n-samples", type=int, default=30,
                         help="Total number of Q&A pairs to extract (default: 30, per build plan's 20-30 spec)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling")
    parser.add_argument("--question-types", default="identification,action,prevention",
                         help="Comma-separated question types to include")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--language", default="en", help="Language filter (default: en; dataset also has 'tw' fields but no non-English rows observed in schema check)")
    parser.add_argument("--save-full-dataset", action="store_true",
                         help="Also save the FULL dataset (both splits, ~3.6GB with images) to "
                              "data/ghanaagricvoa/ as per the project structure spec. Off by "
                              "default since eval_samples.json alone is sufficient for scoring "
                              "and this adds significant download time/disk usage.")
    args = parser.parse_args()

    question_types = [q.strip() for q in args.question_types.split(",")]

    print("=" * 70)
    print("RK AgriDig — Eval Dataset Extraction (Task 2.2)")
    print("=" * 70)
    print(f"Source:          {REPO_ID}")
    print(f"Split:           {args.split}")
    print(f"Target samples:  {args.n_samples}")
    print(f"Question types:  {question_types}")
    print(f"Seed:            {args.seed}")
    print()

    print("Loading dataset (this fetches from HuggingFace Hub)...")
    try:
        ds = load_dataset(REPO_ID, split=args.split)
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: Failed to load dataset: {exc}", file=sys.stderr)
        print(
            "\nIf this is a connection error, verify network access to huggingface.co.\n"
            "If this is a 'dataset not found' error, the repo ID may have changed —\n"
            "check https://huggingface.co/datasets/toufiqmusah/GhanaAgricVQA-Dataset directly.",
            file=sys.stderr,
        )
        return 1

    print(f"Loaded {len(ds)} rows from '{args.split}' split.")

    # --- Filter by question type and language ---
    filtered_indices = [
        i for i, row in enumerate(ds)
        if row.get("question_type") in question_types
        and row.get("language", "en") == args.language
    ]
    print(f"After filtering (question_type in {question_types}, language='{args.language}'): {len(filtered_indices)} rows")

    if not filtered_indices:
        print("ERROR: No rows matched the filter criteria. Check --question-types and --language.", file=sys.stderr)
        return 1

    # --- Balanced sampling: try to get an even split across question_type and crop ---
    random.seed(args.seed)
    by_type_crop = defaultdict(list)
    for i in filtered_indices:
        row = ds[i]
        key = (row.get("question_type"), row.get("crop"))
        by_type_crop[key].append(i)

    buckets = list(by_type_crop.keys())
    per_bucket = max(1, args.n_samples // len(buckets))
    selected_indices = []
    for key in buckets:
        pool = by_type_crop[key]
        random.shuffle(pool)
        selected_indices.extend(pool[:per_bucket])

    # Top up / trim to hit the exact target count
    remaining_pool = [i for i in filtered_indices if i not in selected_indices]
    random.shuffle(remaining_pool)
    if len(selected_indices) < args.n_samples:
        selected_indices.extend(remaining_pool[: args.n_samples - len(selected_indices)])
    selected_indices = selected_indices[: args.n_samples]

    print(f"Selected {len(selected_indices)} samples, balanced across question_type x crop where possible.")

    # --- Build eval samples ---
    eval_samples = []
    validated_count = 0
    confidence_values = []

    for i in selected_indices:
        row = ds[i]
        answer_text = extract_answer_text(row.get("answer"))
        disease = extract_disease_label(row.get("answer"))
        validated = bool(row.get("validated", False))
        confidence = row.get("generation_confidence")

        if validated:
            validated_count += 1
        if confidence is not None:
            confidence_values.append(confidence)

        eval_samples.append({
            "sample_id": row.get("sample_id"),
            "rail_image_id": row.get("rail_image_id"),
            "question": row.get("question"),
            "reference_answer": answer_text,
            "question_type": row.get("question_type"),
            "crop": row.get("crop"),
            "disease": disease,
            "language": row.get("language"),
            "source_validated": validated,
            "source_generation_confidence": confidence,
        })

    # --- Summary stats ---
    type_counts = Counter(s["question_type"] for s in eval_samples)
    crop_counts = Counter(s["crop"] for s in eval_samples)

    output_data = {
        "metadata": {
            "source_dataset": REPO_ID,
            "source_split": args.split,
            "extraction_seed": args.seed,
            "n_samples": len(eval_samples),
            "question_type_distribution": dict(type_counts),
            "crop_distribution": dict(crop_counts),
            "n_validated_by_source": validated_count,
            "n_unvalidated_by_source": len(eval_samples) - validated_count,
            "avg_generation_confidence": round(sum(confidence_values) / len(confidence_values), 3) if confidence_values else None,
            "important_note": (
                "Reference answers come from an AI-generated dataset (Qwen3.5-9B per "
                "dataset card) that is NOT human-validated (validated=false on all "
                "observed rows). Treat these as a reasonable but imperfect ground "
                "truth for automated scoring (BLEU/semantic similarity), not as "
                "expert-verified agronomic fact. Worth spot-checking a handful "
                "manually before relying on accuracy scores derived from this set."
            ),
        },
        "samples": eval_samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print()
    print("=" * 70)
    print(f"Wrote {len(eval_samples)} samples to {args.output}")
    print(f"  Question types: {dict(type_counts)}")
    print(f"  Crops: {dict(crop_counts)}")
    print(f"  Source-validated: {validated_count}/{len(eval_samples)}")
    print("=" * 70)
    print()
    print("NOTE: source dataset rows are NOT human-validated (see metadata.important_note).")
    print("Consider manually spot-checking a sample before treating scores as reliable.")

    # --- Optional: save full local copy of both splits ---
    if args.save_full_dataset:
        full_dir = Path(__file__).resolve().parent.parent / "data" / "ghanaagricvoa"
        print()
        print(f"Saving FULL dataset (train + test, with images) to {full_dir} ...")
        print("This is several GB and may take a while.")
        try:
            train_ds = load_dataset(REPO_ID, split="train")
            test_ds = ds if args.split == "test" else load_dataset(REPO_ID, split="test")
            (full_dir / "train").mkdir(parents=True, exist_ok=True)
            (full_dir / "test").mkdir(parents=True, exist_ok=True)
            train_ds.save_to_disk(str(full_dir / "train"))
            test_ds.save_to_disk(str(full_dir / "test"))
            print(f"✓ Saved full dataset to {full_dir}")
            print(f"  Reload later with: datasets.load_from_disk('{full_dir}/train')")
        except Exception as exc:  # noqa: BLE001
            print(f"WARNING: Failed to save full dataset: {exc}", file=sys.stderr)
            print("This does not affect eval_samples.json, which was already written successfully.", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())