"""
voice_output.py — RK AgriDig

Offline text-to-speech for the diagnosis output, using pyttsx3 (wraps the
OS's native engine — SAPI5 on Windows, espeak-ng on Linux/WSL2). Chosen
deliberately over any neural TTS (Coqui, Bark, etc.): those need their own
model weights and RAM/CPU budget, which directly competes with Phi-3-mini-4k
for this project's ~7.3GB RAM ceiling and thermal budget. pyttsx3 has no
model to load — it's near-instant and effectively free next to LLM inference.

IMPORTANT — Twi is not supported, and this module does not pretend otherwise:
espeak-ng 1.51 (pyttsx3's underlying engine) ships no Akan/Twi voice or
phoneme data. This was checked directly, not assumed:

    $ espeak-ng --voices | grep -i "ak\\b"      -> no match, 132 voices total
    $ espeak-ng -v ak "test"                     -> "Error: the specified
                                                      espeak-ng voice does
                                                      not exist."

Pointing an English-phoneme engine at Twi text would produce mispronounced
audio that could genuinely mislead a farmer who reads less than they
understand spoken Twi — worse than no voice at all. So: English gets real
speech, Twi gets an honest "not available yet" signal from
`speak_diagnosis()` rather than a garbled attempt. If a real Twi voice model
(e.g. Meta MMS-TTS's Akan checkpoint) is ever integrated, only
SUPPORTED_LANGUAGES and the synth call below need to change.

Team: Aaron Baidoo (RoniKid) & Firdaus Kudus (github.com/KudusFirdaus)
"""
from __future__ import annotations

import base64
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Languages this module can actually speak correctly. "tw" (Twi) is
# deliberately excluded — see module docstring.
SUPPORTED_LANGUAGES = {"en"}

_MAX_CHARS = 1200  # keep synth time low and bounded on modest CPUs


@dataclass
class SpeechResult:
    success: bool
    audio_path: Optional[str] = None
    reason: Optional[str] = None  # set when success=False, e.g. "unsupported_language"


def _strip_html_for_speech(html_or_text: str) -> str:
    """The diagnosis result is HTML (rk-card markup). Speech needs plain
    text — strip tags and collapse whitespace rather than reading '<div
    class=rk-sec-title>' aloud."""
    text = re.sub(r"<[^>]+>", " ", html_or_text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def speak_diagnosis(html_or_text: str, lang: str) -> SpeechResult:
    """
    Synthesize the diagnosis text to a temporary WAV file and return its
    path. Returns success=False with a reason (never raises) if the
    language isn't supported or synthesis fails — callers should show that
    reason to the user rather than silently doing nothing.
    """
    if lang not in SUPPORTED_LANGUAGES:
        return SpeechResult(success=False, reason="unsupported_language")

    plain_text = _strip_html_for_speech(html_or_text)[:_MAX_CHARS]
    if not plain_text:
        return SpeechResult(success=False, reason="empty_text")

    try:
        import pyttsx3
    except ImportError:
        return SpeechResult(success=False, reason="pyttsx3_not_installed")

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)  # slightly slower than default for clarity

        out_path = Path(tempfile.gettempdir()) / f"rk_agridig_speech_{abs(hash(plain_text)) % 10**8}.wav"
        engine.save_to_file(plain_text, str(out_path))
        engine.runAndWait()

        if not out_path.exists() or out_path.stat().st_size == 0:
            return SpeechResult(success=False, reason="synthesis_produced_no_audio")

        return SpeechResult(success=True, audio_path=str(out_path))
    except Exception as exc:
        return SpeechResult(success=False, reason=f"engine_error: {exc}")


def speech_status_message(lang: str) -> str:
    """Short, honest label for the UI badge next to the voice button —
    called before synth even runs, so an unsupported language shows a clear
    reason instead of a button that silently does nothing when clicked."""
    if lang in SUPPORTED_LANGUAGES:
        return ""
    return "🔇 Voice narration isn't available in Twi yet — espeak-ng has no Twi voice data."