"""
ollama_client.py — RK AgriDig

Bridge between the Gradio UI and the locally-running Ollama server
(serving the quantized Phi-3-mini GGUF model). All inference stays
on-device: no cloud calls, no external API keys.

Public interface:
    client = OllamaClient()
    client.health_check() -> bool
    client.load_model("phi3agridig") -> bool
    client.infer(question, crop, question_type) -> InferenceResult
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

# ---------------------------------------------------------------------------
# Logging setup — logs to file so requests/responses can be inspected later
# for debugging or for the PROMPTS.md / REPORT.md writeups.
# ---------------------------------------------------------------------------

LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("rk_agridig.ollama_client")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_DIR / "ollama_client.log", encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    )
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(console_handler)


# ---------------------------------------------------------------------------
# System prompt — kept in this module so infer() is self-contained.
# Swap out for the "balanced" variant from Phase 2 prompt engineering
# if you want to A/B against a different system prompt.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert agricultural advisor helping smallholder farmers in Ghana diagnose and treat crop diseases affecting maize, pepper, and tomato.

Always respond in this exact structure:
Disease Name: <name>
Symptoms: <what the farmer is likely seeing>
Treatment: <numbered steps, 2-4 items>
Prevention: <bullet steps, 2-4 items>

Rules:
- Use simple, farmer-friendly language. Avoid jargon.
- Base answers on known maize, pepper, and tomato diseases common in West Africa.
- If uncertain, say so plainly rather than guessing.
- End every response with: "Based on Ghana crop data — confirm with a local agricultural extension officer if unsure."
"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class InferenceResult:
    success: bool
    raw_text: str = ""
    disease: str = ""
    symptoms: str = ""
    treatment: str = ""
    prevention: str = ""
    latency_seconds: float = 0.0
    error: Optional[str] = None
    # Rough confidence proxy: 1.0 if all 4 sections parsed, less if partial.
    structured_confidence: float = 0.0


class OllamaUnavailableError(RuntimeError):
    """Raised when the Ollama server cannot be reached at all."""


class OllamaTimeoutError(RuntimeError):
    """Raised when inference exceeds the configured timeout."""


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

class OllamaClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model_name: str = "phi3-agridig",   # must match `ollama create <name> -f Modelfile`
        timeout_seconds: int = 150,          # headroom: ~32s cold tg + model load + Docker overhead
        max_retries: int = 3,
        retry_backoff_seconds: float = 2.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._model_loaded = False

    # -- Health / lifecycle -------------------------------------------------

    def health_check(self) -> bool:
        """Returns True if the Ollama server responds at all."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            ok = resp.status_code == 200
            logger.info(f"health_check: {'OK' if ok else f'HTTP {resp.status_code}'}")
            return ok
        except requests.exceptions.RequestException as e:
            logger.warning(f"health_check failed: {e}")
            return False

    def load_model(self, model_name: Optional[str] = None) -> bool:
        """
        Verifies the target model is available on the Ollama server.
        Ollama loads models lazily on first inference, so this mainly
        confirms the model has been pulled/created (e.g. via `ollama create`).
        """
        target = model_name or self.model_name
        if not self.health_check():
            logger.error("load_model: Ollama server unreachable")
            return False

        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            available = [m["name"] for m in resp.json().get("models", [])]

            # Ollama model names may or may not carry a ":tag" suffix.
            matched = any(target == m or m.startswith(f"{target}:") for m in available)

            if matched:
                logger.info(f"load_model: '{target}' found on server")
                self._model_loaded = True
                return True

            logger.error(
                f"load_model: '{target}' not found. Available models: {available}. "
                f"Run `ollama create {target} -f Modelfile` or `ollama pull {target}` first."
            )
            return False

        except requests.exceptions.RequestException as e:
            logger.error(f"load_model: error listing models: {e}")
            return False

    # -- Inference ------------------------------------------------------------

    def infer(
        self,
        question: str,
        crop: str = "Maize",
        question_type: str = "Identification",
    ) -> InferenceResult:
        """
        Runs inference against the Ollama server with retry logic.

        Args:
            question: farmer's free-text description of the problem.
            crop: one of "Maize", "Pepper", "Tomato".
            question_type: one of "Identification", "Treatment", "Prevention".

        Returns:
            InferenceResult with parsed structured fields, or success=False
            with `.error` set if inference failed after all retries.
        """
        user_prompt = self._build_user_prompt(question, crop, question_type)

        # Token budget by question type.
        # Identification needs all 4 sections → more tokens.
        # Treatment / Prevention are single-section answers → cap lower.
        # Tighter caps = faster responses + less thermal load on sustained use.
        # At ~17 TPS: 300 tok ≈ 18s, 200 tok ≈ 12s, 150 tok ≈ 9s.
        num_predict_by_type = {
            "Identification": 300,
            "Treatment": 200,
            "Prevention": 150,
        }
        num_predict = num_predict_by_type.get(question_type, 300)

        payload = {
            "model": self.model_name,
            "system": SYSTEM_PROMPT,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": num_predict,
            },
        }

        last_error: Optional[str] = None

        for attempt in range(1, self.max_retries + 1):
            start = time.time()
            try:
                logger.info(
                    f"infer: attempt {attempt}/{self.max_retries} "
                    f"(crop={crop}, type={question_type})"
                )
                resp = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    timeout=self.timeout_seconds,
                )
                resp.raise_for_status()
                elapsed = time.time() - start

                data = resp.json()
                raw_text = data.get("response", "").strip()

                logger.info(
                    f"infer: success in {elapsed:.2f}s, "
                    f"{len(raw_text)} chars returned"
                )
                logger.debug(f"infer: raw_text={raw_text!r}")

                parsed = self._parse_structured_response(raw_text)
                parsed.latency_seconds = elapsed
                parsed.success = True
                return parsed

            except requests.exceptions.Timeout:
                last_error = f"Inference timed out after {self.timeout_seconds}s"
                logger.warning(f"infer: attempt {attempt} timed out")

            except requests.exceptions.ConnectionError as e:
                last_error = f"Cannot reach Ollama server at {self.base_url}: {e}"
                logger.warning(f"infer: attempt {attempt} connection error: {e}")

            except requests.exceptions.RequestException as e:
                last_error = f"Ollama request failed: {e}"
                logger.warning(f"infer: attempt {attempt} request error: {e}")

            except (json.JSONDecodeError, KeyError) as e:
                last_error = f"Malformed response from Ollama: {e}"
                logger.error(f"infer: attempt {attempt} parse error: {e}")

            if attempt < self.max_retries:
                sleep_time = self.retry_backoff_seconds * attempt
                logger.info(f"infer: retrying in {sleep_time:.1f}s")
                time.sleep(sleep_time)

        logger.error(f"infer: all {self.max_retries} attempts failed: {last_error}")
        return InferenceResult(
            success=False,
            error=last_error or "Unknown inference error",
            raw_text=self._fallback_message(),
        )

    # -- Internal helpers -----------------------------------------------------

    @staticmethod
    def _build_user_prompt(question: str, crop: str, question_type: str) -> str:
        return (
            f"Crop: {crop}\n"
            f"Question type: {question_type}\n"
            f"Farmer's question: {question.strip()}"
        )

    @staticmethod
    def _parse_structured_response(raw_text: str) -> InferenceResult:
        """
        Parses the expected 'Disease Name / Symptoms / Treatment / Prevention'
        structure out of the model's free-text response. Falls back gracefully
        if the model didn't follow the format exactly.
        """
        sections = {
            "disease": "",
            "symptoms": "",
            "treatment": "",
            "prevention": "",
        }

        patterns = {
            "disease": r"Disease Name:\s*(.+?)(?=\n\s*Symptoms:|\Z)",
            "symptoms": r"Symptoms:\s*(.+?)(?=\n\s*Treatment:|\Z)",
            "treatment": r"Treatment:\s*(.+?)(?=\n\s*Prevention:|\Z)",
            "prevention": r"Prevention:\s*(.+?)(?=\Z)",
        }

        found_count = 0
        for key, pattern in patterns.items():
            match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
            if match:
                sections[key] = match.group(1).strip()
                found_count += 1

        confidence = found_count / 4.0

        return InferenceResult(
            success=True,
            raw_text=raw_text,
            disease=sections["disease"],
            symptoms=sections["symptoms"],
            treatment=sections["treatment"],
            prevention=sections["prevention"],
            structured_confidence=confidence,
        )

    @staticmethod
    def _fallback_message() -> str:
        return (
            "I couldn't reach the on-device model right now. "
            "Please make sure Ollama is running (`ollama serve`) and try again. "
            "If the problem persists, restart the app."
        )


# ---------------------------------------------------------------------------
# Manual smoke test — run `python src/ollama_client.py` to sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    client = OllamaClient()

    print("Checking Ollama server health...")
    if not client.health_check():
        print("❌ Ollama server not reachable at http://localhost:11434")
        print("   Start it with: ollama serve")
        raise SystemExit(1)

    print("✅ Server reachable")

    print(f"Checking model '{client.model_name}' is available...")
    if not client.load_model():
        print("❌ Model not found. See log for details.")
        raise SystemExit(1)

    print("✅ Model available")

    print("\nRunning sample inference...")
    result = client.infer(
        question="My pepper leaves have brown spots with yellow halos.",
        crop="Pepper",
        question_type="Identification",
    )

    if result.success:
        print(f"\n--- Response ({result.latency_seconds:.2f}s, "
              f"structured_confidence={result.structured_confidence:.2f}) ---")
        print(f"Disease:    {result.disease}")
        print(f"Symptoms:   {result.symptoms}")
        print(f"Treatment:  {result.treatment}")
        print(f"Prevention: {result.prevention}")
    else:
        print(f"❌ Inference failed: {result.error}")