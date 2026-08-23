"""
image_symptom_extractor.py — RK AgriDig

Classical computer-vision pre-processor for leaf photos. This module does NOT
diagnose disease — Phi-3-mini-4k-instruct is text-only and has no vision
tower, and swapping in a vision-capable model (e.g. Phi-3.5-vision) would
blow past the ~7.3GB RAM ceiling and thermal budget this project already
operates under (see Phase 1 profiling notes).

Instead, this module extracts *observable, describable* symptom signals from
a leaf photo using deterministic OpenCV heuristics (color segmentation,
lesion counting, coverage ratio) and turns them into a short natural-language
symptom description. That description is appended to whatever the farmer
typed, and the existing text-only OllamaClient / prompt_engine pipeline
handles it completely unchanged downstream.

This is intentionally NOT a disease classifier. It answers "what does this
leaf look like" (yellowing %, brown lesion count/coverage, spot pattern),
not "what disease is this" — that inference step stays with the LLM, which
already has the GhanaAgricVQA-grounded prompting to reason over symptoms.

Team: Aaron Baidoo (RoniKid) & Firdaus Kudus (github.com/KudusFirdaus)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np


@dataclass
class ImageSymptomSignals:
    """Raw quantitative signals pulled from a leaf photo."""
    green_pct: float = 0.0
    yellow_pct: float = 0.0
    brown_pct: float = 0.0
    dark_spot_count: int = 0
    dark_spot_coverage_pct: float = 0.0
    largest_spot_area_pct: float = 0.0
    analyzed: bool = False
    error: Optional[str] = None

    def to_symptom_text(self) -> str:
        """Turn raw signals into a short natural-language fragment suitable
        for appending to a farmer's typed symptom description. Deliberately
        conservative — describes what's visible, doesn't name a disease."""
        if not self.analyzed:
            return ""

        fragments = []

        if self.yellow_pct >= 15:
            fragments.append(f"approximately {self.yellow_pct:.0f}% of the leaf area appears yellowed")
        elif self.yellow_pct >= 5:
            fragments.append("mild yellowing is visible in patches")

        if self.dark_spot_count >= 8:
            fragments.append(
                f"numerous small dark/brown lesions are visible (roughly {self.dark_spot_count} distinct spots, "
                f"covering ~{self.dark_spot_coverage_pct:.0f}% of the leaf)"
            )
        elif self.dark_spot_count >= 2:
            fragments.append(
                f"a few dark/brown lesions are visible ({self.dark_spot_count} spots, "
                f"~{self.dark_spot_coverage_pct:.0f}% coverage)"
            )
        elif self.dark_spot_count == 1 and self.largest_spot_area_pct >= 5:
            fragments.append(
                f"one large lesion covers roughly {self.largest_spot_area_pct:.0f}% of the visible leaf area"
            )

        if self.brown_pct >= 25:
            fragments.append(f"large areas ({self.brown_pct:.0f}%) show browning/necrosis")

        if not fragments:
            return "the leaf photo shows no strong discoloration or lesion pattern detected by automated analysis"

        return "Image analysis detected: " + "; ".join(fragments) + "."


def analyze_leaf_image(image_bgr: np.ndarray) -> ImageSymptomSignals:
    """
    Run deterministic color/lesion heuristics on a leaf photo.

    image_bgr: OpenCV-style BGR array (as returned by cv2.imread, or converted
               from a PIL/Gradio RGB array via cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)).

    Runs in well under 100ms on commodity CPUs at typical phone-photo
    resolutions — negligible next to LLM inference time and irrelevant to the
    thermal budget, since it's plain array math, not a neural forward pass.
    """
    try:
        if image_bgr is None or image_bgr.size == 0:
            return ImageSymptomSignals(analyzed=False, error="Empty image.")

        # Downscale for speed/consistency — heuristics are ratio-based so
        # resolution doesn't need to be native.
        h, w = image_bgr.shape[:2]
        max_dim = 800
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))

        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        # Isolate the leaf itself from background. Note this must include
        # low-hue brown/necrotic tissue (hue ~0-15) as well as green-yellow
        # (~15-95) — a lesion is still leaf, just diseased leaf. An earlier
        # version of this mask started at hue 15, which silently excluded
        # brown lesions from "leaf" and made every dark-spot count come back
        # zero (caught via a synthetic-image test before this shipped).
        # Background exclusion instead relies on saturation/value bounds:
        # bare soil, skin, and gray backgrounds tend to be either very
        # desaturated or very bright/dark relative to leaf tissue.
        leaf_mask = cv2.inRange(hsv, (0, 25, 20), (95, 255, 255))
        leaf_pixel_count = int(np.count_nonzero(leaf_mask))
        if leaf_pixel_count < 500:
            # Couldn't confidently find a leaf in-frame — degrade honestly
            # rather than reporting misleading 0%/100% ratios.
            return ImageSymptomSignals(
                analyzed=False,
                error="Could not confidently isolate a leaf in this image — try a closer, well-lit photo against a plain background.",
            )

        def pct_of_leaf(mask: np.ndarray) -> float:
            return 100.0 * np.count_nonzero(mask) / leaf_pixel_count

        # Healthy green
        green_mask = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255)) & leaf_mask
        # Yellowing (chlorosis) — narrower hue band than green
        yellow_mask = cv2.inRange(hsv, (18, 40, 60), (34, 255, 255)) & leaf_mask
        # Brown/necrotic — low saturation/value, warm hue
        brown_mask = cv2.inRange(hsv, (5, 30, 20), (25, 200, 150)) & leaf_mask

        green_pct = pct_of_leaf(green_mask)
        yellow_pct = pct_of_leaf(yellow_mask)
        brown_pct = pct_of_leaf(brown_mask)

        # Dark lesion / spot detection via contours on a dark-value mask
        # restricted to the leaf area.
        dark_mask = cv2.inRange(hsv, (0, 0, 0), (180, 255, 90)) & leaf_mask
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        min_spot_area = max(15, leaf_pixel_count * 0.0008)  # ignore noise specks
        spot_areas = [cv2.contourArea(c) for c in contours if cv2.contourArea(c) >= min_spot_area]
        dark_spot_count = len(spot_areas)
        dark_spot_coverage_pct = 100.0 * sum(spot_areas) / leaf_pixel_count if spot_areas else 0.0
        largest_spot_area_pct = 100.0 * max(spot_areas) / leaf_pixel_count if spot_areas else 0.0

        return ImageSymptomSignals(
            green_pct=round(green_pct, 1),
            yellow_pct=round(yellow_pct, 1),
            brown_pct=round(brown_pct, 1),
            dark_spot_count=dark_spot_count,
            dark_spot_coverage_pct=round(dark_spot_coverage_pct, 1),
            largest_spot_area_pct=round(largest_spot_area_pct, 1),
            analyzed=True,
        )
    except Exception as exc:  # Never let a bad photo crash the diagnosis flow
        return ImageSymptomSignals(analyzed=False, error=f"Image analysis failed: {exc}")


def build_augmented_question(typed_question: str, image_bgr: Optional[np.ndarray]) -> tuple[str, Optional[ImageSymptomSignals]]:
    """
    Combine the farmer's typed description with image-derived signals (if a
    photo was provided) into the single question string the existing
    OllamaClient.infer() already expects — no change needed downstream.

    Returns (augmented_question, signals_or_None) so the UI can also show
    the raw signals to the farmer for transparency.
    """
    typed_question = (typed_question or "").strip()

    if image_bgr is None:
        return typed_question, None

    signals = analyze_leaf_image(image_bgr)

    if not signals.analyzed:
        # Honest degrade: image didn't yield usable signals. Don't silently
        # drop it without telling the caller why.
        return typed_question, signals

    image_text = signals.to_symptom_text()
    if not typed_question:
        return image_text, signals

    return f"{typed_question} {image_text}", signals