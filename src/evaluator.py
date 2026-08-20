"""
evaluator.py — RK AgriDig Phase 2, Task 2.3

Scores model-generated answers against the reference answers in
data/eval_samples.json. This directly measures Sacc (50% of the ADTC
final score), so correctness here matters more than almost anything
else in this project.

Metrics computed:
    - BLEU-4 (via sacrebleu, NOT nltk -- sacrebleu is the reproducible,
      standardized choice; see docs/PROMPTING_STRATEGY.md-adjacent
      reasoning. Score normalized to 0-1 to match this project's schema;
      sacrebleu itself returns 0-100.)
    - Semantic similarity (cosine similarity between sentence-transformers
      embeddings, all-MiniLM-L6-v2 -- small and fast enough to run
      alongside CPU-only inference without becoming its own bottleneck)
    - Disease identification accuracy (normalized exact-ish match against
      the reference's `disease` field, when present)
    - Structured output compliance (% of responses containing all 4
      expected sections: Disease/Symptoms/Treatment/Prevention, allowing
      for the fact that action-only or prevention-only questions don't
      need all 4 -- see _check_structure for the real logic)

Usage:
    python3 src/evaluator.py --predictions path/to/model_outputs.json
    python3 src/evaluator.py --predictions outputs.json --eval-set data/eval_samples.json

Expected input format for --predictions: a JSON file mapping sample_id ->
generated answer text, e.g.:
    {"rail_id_2twrvk_ide": "The maize leaf looks healthy...", ...}
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

try:
    import sacrebleu
except ImportError:
    print("ERROR: sacrebleu not installed. Install with: pip install sacrebleu --break-system-packages", file=sys.stderr)
    sys.exit(1)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:
    print(
        "ERROR: sentence-transformers/scikit-learn not installed.\n"
        "Install with: pip install sentence-transformers scikit-learn --break-system-packages",
        file=sys.stderr,
    )
    sys.exit(1)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_SET = REPO_ROOT / "data" / "eval_samples.json"
DEFAULT_OUTPUT = REPO_ROOT / "benchmarks" / "results" / "accuracy_metrics.json"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

REQUIRED_SECTIONS = ["disease", "symptoms", "treatment", "prevention"]


def normalize_disease_name(name: str) -> str:
    """
    Normalize disease names for comparison: lowercase, strip crop prefix
    (e.g. 'Corn_' / 'Pepper_' / 'Tomato_'), replace underscores with spaces,
    strip punctuation. This is intentionally loose -- an exact string match
    on raw labels like 'Corn_Cercospora_Leaf_Spot' vs a model saying
    "Cercospora Leaf Spot" would otherwise always fail despite being
    correct.
    """
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r"^(corn|maize|pepper|tomato)[_\s]+", "", name)
    name = name.replace("_", " ")
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name


def disease_match(reference_disease: str | None, generated_text: str) -> bool | None:
    """
    Check whether the generated text mentions the reference disease name.
    Returns None (not applicable) if there's no reference disease to check
    against (e.g. healthy-plant samples) -- callers should exclude None
    results from the accuracy denominator, not count them as wrong.
    """
    if not reference_disease:
        return None
    normalized_ref = normalize_disease_name(reference_disease)
    if not normalized_ref:
        return None
    normalized_gen = normalize_disease_name(generated_text)
    # Substring match on normalized forms -- catches "Cercospora Leaf Spot"
    # appearing anywhere in a longer generated response.
    return normalized_ref in normalized_gen


def check_structure_compliance(generated_text: str, question_type: str) -> bool:
    """
    Check whether the response follows the required structured format.

    Per the system prompts (src/prompt_engineer.py), the FULL 4-section
    structure (Disease/Symptoms/Treatment/Prevention) is only expected for
    general diagnosis questions. Action-only or prevention-only questions
    are explicitly allowed to answer just that section without repeating
    the full structure -- so this function checks for the presence of the
    section header relevant to the question_type, not all 4 unconditionally.
    """
    text_lower = generated_text.lower()

    if question_type == "identification":
        # Expect at least Disease + Symptoms sections for a diagnosis question
        return ("disease:" in text_lower or "disease " in text_lower) and \
               ("symptom" in text_lower)
    elif question_type == "action":
        return "treatment" in text_lower or "step" in text_lower
    elif question_type == "prevention":
        return "prevent" in text_lower or "next season" in text_lower or "next planting" in text_lower
    else:
        # Unknown question type -- fall back to checking for any structure at all
        return any(section in text_lower for section in REQUIRED_SECTIONS)


def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Sentence-level BLEU-4 via sacrebleu, normalized to 0-1 (sacrebleu
    itself returns 0-100). Returns 0.0 for empty hypothesis/reference
    rather than raising, since a missing prediction is a real (bad)
    result, not an error condition to crash on.
    """
    if not hypothesis.strip() or not reference.strip():
        return 0.0
    result = sacrebleu.sentence_bleu(hypothesis, [reference])
    return result.score / 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Score model outputs against the RK AgriDig eval set.")
    parser.add_argument("--predictions", type=Path, required=True,
                         help="JSON file mapping sample_id -> generated answer text")
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_SET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL_NAME)
    args = parser.parse_args()

    print("=" * 70)
    print("RK AgriDig — Accuracy Evaluator (Task 2.3)")
    print("=" * 70)

    if not args.eval_set.exists():
        print(f"ERROR: eval set not found at {args.eval_set}. Run scripts/create_eval_dataset.py first.", file=sys.stderr)
        return 1
    if not args.predictions.exists():
        print(f"ERROR: predictions file not found at {args.predictions}.", file=sys.stderr)
        return 1

    with open(args.eval_set) as f:
        eval_data = json.load(f)
    with open(args.predictions) as f:
        predictions = json.load(f)

    samples = eval_data["samples"]
    print(f"Loaded {len(samples)} eval samples, {len(predictions)} predictions.")

    missing_predictions = [s["sample_id"] for s in samples if s["sample_id"] not in predictions]
    if missing_predictions:
        print(f"WARNING: {len(missing_predictions)} eval samples have no matching prediction "
              f"(will be scored as 0 / failed): {missing_predictions[:5]}{'...' if len(missing_predictions) > 5 else ''}")

    print(f"\nLoading embedding model ({args.embedding_model})...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        embedder = SentenceTransformer(args.embedding_model)

    # --- Score each sample ---
    per_sample_results = []
    for sample in samples:
        sid = sample["sample_id"]
        reference = sample["reference_answer"]
        generated = predictions.get(sid, "")

        bleu = compute_bleu(reference, generated)

        if generated.strip() and reference.strip():
            embeddings = embedder.encode([reference, generated])
            semantic_sim = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
        else:
            semantic_sim = 0.0

        disease_correct = disease_match(sample.get("disease"), generated)
        structured = check_structure_compliance(generated, sample.get("question_type", ""))

        per_sample_results.append({
            "sample_id": sid,
            "question": sample["question"],
            "question_type": sample.get("question_type"),
            "crop": sample.get("crop"),
            "reference_disease": sample.get("disease"),
            "reference_answer": reference,
            "generated_answer": generated,
            "has_prediction": sid in predictions,
            "bleu_score": round(bleu, 4),
            "semantic_similarity": round(semantic_sim, 4),
            "disease_correct": disease_correct,  # True / False / None (not applicable)
            "structured_compliance": structured,
        })

    # --- Aggregate metrics ---
    def safe_mean(values):
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 4) if values else 0.0

    overall_bleu = safe_mean([r["bleu_score"] for r in per_sample_results])
    overall_semantic = safe_mean([r["semantic_similarity"] for r in per_sample_results])
    overall_structured = safe_mean([1.0 if r["structured_compliance"] else 0.0 for r in per_sample_results])

    disease_results = [r["disease_correct"] for r in per_sample_results if r["disease_correct"] is not None]
    disease_accuracy = safe_mean([1.0 if d else 0.0 for d in disease_results]) if disease_results else None

    # "overall_accuracy" per the build plan's schema is a blended score --
    # defined here as the average of BLEU, semantic similarity, and disease
    # accuracy (structured compliance tracked separately since it's a
    # format check, not an accuracy check). This blend is a judgment call;
    # documented here rather than left implicit.
    accuracy_components = [overall_bleu, overall_semantic]
    if disease_accuracy is not None:
        accuracy_components.append(disease_accuracy)
    overall_accuracy = round(sum(accuracy_components) / len(accuracy_components), 4)

    # --- Breakdown by question_type and crop ---
    def breakdown_by(key: str) -> dict:
        groups = defaultdict(list)
        for r in per_sample_results:
            groups[r[key]].append(r)
        out = {}
        for group_key, rows in groups.items():
            group_disease = [r["disease_correct"] for r in rows if r["disease_correct"] is not None]
            out[group_key] = {
                "n_samples": len(rows),
                "bleu_score": safe_mean([r["bleu_score"] for r in rows]),
                "semantic_similarity": safe_mean([r["semantic_similarity"] for r in rows]),
                "structured_compliance": safe_mean([1.0 if r["structured_compliance"] else 0.0 for r in rows]),
                "disease_accuracy": safe_mean([1.0 if d else 0.0 for d in group_disease]) if group_disease else None,
            }
        return out

    accuracy_by_question_type = breakdown_by("question_type")
    accuracy_by_crop = breakdown_by("crop")

    report = {
        "overall_accuracy": overall_accuracy,
        "bleu_score": overall_bleu,
        "semantic_similarity": overall_semantic,
        "structured_compliance": overall_structured,
        "disease_identification_accuracy": disease_accuracy,
        "accuracy_by_question_type": accuracy_by_question_type,
        "accuracy_by_crop": accuracy_by_crop,
        "n_samples_total": len(samples),
        "n_samples_with_predictions": len(samples) - len(missing_predictions),
        "n_samples_missing_predictions": len(missing_predictions),
        "adtc_targets": {
            "identification_accuracy_target": 0.85,
            "bleu_target": 0.70,
            "identification_accuracy_actual": accuracy_by_question_type.get("identification", {}).get("disease_accuracy"),
            "bleu_actual": overall_bleu,
            "meets_identification_target": (
                accuracy_by_question_type.get("identification", {}).get("disease_accuracy") is not None
                and accuracy_by_question_type["identification"]["disease_accuracy"] >= 0.85
            ),
            "meets_bleu_target": overall_bleu >= 0.70,
        },
        "per_sample_results": per_sample_results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    # --- Print summary ---
    print()
    print("=" * 70)
    print("Results Summary")
    print("=" * 70)
    print(f"  Overall accuracy (blended):       {overall_accuracy}")
    print(f"  BLEU-4:                            {overall_bleu}  (target: >=0.70, {'PASS' if overall_bleu >= 0.70 else 'FAIL'})")
    print(f"  Semantic similarity:               {overall_semantic}")
    print(f"  Structured output compliance:      {overall_structured}")
    if disease_accuracy is not None:
        ident_acc = accuracy_by_question_type.get("identification", {}).get("disease_accuracy")
        print(f"  Disease ID accuracy (overall):      {disease_accuracy}")
        if ident_acc is not None:
            print(f"  Disease ID accuracy (identification-type only): {ident_acc}  (target: >=0.85, {'PASS' if ident_acc >= 0.85 else 'FAIL'})")
    else:
        print("  Disease ID accuracy:                N/A (no samples with a reference disease label)")
    print()
    print(f"  Wrote full report + per-sample results to: {args.output}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())