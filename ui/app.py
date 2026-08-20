"""
app.py — RK AgriDig

Gradio web interface for farmer interaction. Connects to the local
Ollama server via src/ollama_client.py — no cloud calls, fully offline.

Run:
    ollama serve &
    python ui/app.py

Then open http://localhost:7860
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import gradio as gr

# Allow `python ui/app.py` to find the sibling src/ package regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ollama_client import OllamaClient, InferenceResult  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_QUESTION_CHARS = 500
CROPS = ["Maize", "Pepper", "Tomato"]
QUESTION_TYPES = ["Identification", "Treatment", "Prevention"]

client = OllamaClient()

# ---------------------------------------------------------------------------
# Styling — green + blue, matching RK AgriDig branding. Mobile-friendly.
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
:root {
    --rk-green: #2e7d32;
    --rk-green-light: #e8f5e9;
    --rk-blue: #1565c0;
    --rk-blue-light: #e3f2fd;
}

.rk-header {
    text-align: center;
    padding: 8px 0 4px 0;
}

.rk-header h1 {
    color: var(--rk-green);
    margin-bottom: 4px;
}

.rk-header p {
    color: #555;
    font-size: 0.95em;
}

.rk-response-box {
    border-left: 4px solid var(--rk-green);
    background: var(--rk-green-light);
    border-radius: 8px;
    padding: 16px;
}

.rk-disease-name {
    color: var(--rk-green);
    font-size: 1.2em;
    font-weight: 700;
    margin-bottom: 8px;
}

.rk-section-label {
    color: var(--rk-blue);
    font-weight: 600;
    margin-top: 10px;
}

.rk-confidence-pill {
    display: inline-block;
    background: var(--rk-blue-light);
    color: var(--rk-blue);
    border-radius: 999px;
    padding: 2px 10px;
    font-size: 0.8em;
    font-weight: 600;
}

.rk-footer-note {
    font-size: 0.8em;
    color: #777;
    margin-top: 12px;
    font-style: italic;
}

@media (max-width: 600px) {
    .rk-header h1 { font-size: 1.4em; }
}
"""

THEME = gr.themes.Soft(
    primary_hue="green",
    secondary_hue="blue",
)

# ---------------------------------------------------------------------------
# Core handler
# ---------------------------------------------------------------------------

def diagnose(
    question: str,
    crop: str,
    question_type: str,
    history: list,
):
    """
    Runs a single diagnosis request against the Ollama-backed model
    and formats the result as Markdown for display, plus updates
    the running chat-style history.
    """
    question = (question or "").strip()

    if not question:
        warning = "⚠️ Please describe what you're seeing on your crop before submitting."
        return warning, history, history

    if len(question) > MAX_QUESTION_CHARS:
        question = question[:MAX_QUESTION_CHARS]

    if not client.health_check():
        error_md = (
            "### ⚠️ Model server not reachable\n\n"
            "The on-device AI model isn't running. Please start it with:\n\n"
            "```\nollama serve\n```\n\n"
            "Then try your question again."
        )
        return error_md, history, history

    start = time.time()
    result: InferenceResult = client.infer(
        question=question, crop=crop, question_type=question_type
    )
    elapsed = time.time() - start

    response_md = _format_response(result, elapsed)

    entry = {
        "question": question,
        "crop": crop,
        "question_type": question_type,
        "response_md": response_md,
        "timestamp": time.strftime("%H:%M:%S"),
    }
    new_history = history + [entry]

    return response_md, new_history, new_history


def _format_response(result: InferenceResult, elapsed: float) -> str:
    if not result.success:
        return (
            "### ⚠️ Couldn't complete diagnosis\n\n"
            f"{result.error or 'Unknown error.'}\n\n"
            f"{result.raw_text}"
        )

    confidence_pct = int(result.structured_confidence * 100)

    # If parsing mostly failed, fall back to showing the raw model text
    # rather than a half-empty structured card.
    if result.structured_confidence < 0.5:
        return (
            f'<span class="rk-confidence-pill">⏱ {elapsed:.1f}s</span>\n\n'
            f"{result.raw_text}\n\n"
            '<p class="rk-footer-note">Response shown as-is — the model\'s answer '
            "didn't fully match the expected structured format.</p>"
        )

    parts = [f'<span class="rk-confidence-pill">⏱ {elapsed:.1f}s · '
             f'{confidence_pct}% structured</span>\n']

    if result.disease:
        parts.append(f'<div class="rk-disease-name">🦠 {result.disease}</div>')

    if result.symptoms:
        parts.append('<div class="rk-section-label">Symptoms</div>')
        parts.append(result.symptoms)

    if result.treatment:
        parts.append('<div class="rk-section-label">🛠️ Treatment</div>')
        parts.append(result.treatment)

    if result.prevention:
        parts.append('<div class="rk-section-label">🛡️ Prevention for Next Season</div>')
        parts.append(result.prevention)

    parts.append(
        '<p class="rk-footer-note">Based on Ghana crop data (GhanaAgricVQA) — '
        "confirm with a local agricultural extension officer if unsure.</p>"
    )

    return "\n\n".join(parts)


def format_history_display(history: list) -> str:
    """Renders the running history as a simple chat-style Markdown log."""
    if not history:
        return "*No questions asked yet this session.*"

    lines = []
    for entry in reversed(history[-10:]):  # most recent first, capped
        lines.append(
            f"**[{entry['timestamp']}] {entry['crop']} · {entry['question_type']}**\n"
            f"> {entry['question']}\n"
        )
    return "\n\n---\n\n".join(lines)


def clear_all():
    return "", [], "*No questions asked yet this session.*", []


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------

with gr.Blocks(title="RK AgriDig") as demo:
    history_state = gr.State([])

    with gr.Column(elem_classes="rk-header"):
        gr.Markdown(
            "# 🌾 RK AgriDig\n"
            "Offline crop disease diagnosis for Ghanaian farmers — "
            "no internet required."
        )

    with gr.Row():
        with gr.Column(scale=1):
            question_input = gr.Textbox(
                label="Describe what you're seeing",
                placeholder="E.g., My maize leaves have brown spots with yellow edges...",
                lines=4,
                max_lines=8,
            )

            crop_dropdown = gr.Dropdown(
                choices=CROPS,
                value="Maize",
                label="Crop",
            )

            question_type_radio = gr.Radio(
                choices=QUESTION_TYPES,
                value="Identification",
                label="What do you need?",
            )

            with gr.Row():
                submit_btn = gr.Button("🔍 Diagnose", variant="primary")
                clear_btn = gr.Button("Clear", variant="secondary")

        with gr.Column(scale=1):
            response_output = gr.Markdown(
                value="*Your diagnosis will appear here.*",
                elem_classes="rk-response-box",
            )

    with gr.Accordion("📜 Past Questions (this session)", open=False):
        history_display = gr.Markdown(value="*No questions asked yet this session.*")

    gr.Markdown(
        "---\n"
        "**RK AgriDig** runs entirely on your device using a quantized Phi-3-mini "
        "model. Built on the GhanaAgricVQA dataset (Maize, Pepper, Tomato). "
        "For the Africa Deep Tech Challenge 2026."
    )

    # -- Wiring -------------------------------------------------------------

    submit_btn.click(
        fn=diagnose,
        inputs=[question_input, crop_dropdown, question_type_radio, history_state],
        outputs=[response_output, history_state, history_state],
    ).then(
        fn=format_history_display,
        inputs=[history_state],
        outputs=[history_display],
    )

    question_input.submit(
        fn=diagnose,
        inputs=[question_input, crop_dropdown, question_type_radio, history_state],
        outputs=[response_output, history_state, history_state],
    ).then(
        fn=format_history_display,
        inputs=[history_state],
        outputs=[history_display],
    )

    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[question_input, history_state, history_display, history_state],
    )


if __name__ == "__main__":
    print("Checking Ollama server before launch...")
    if not client.health_check():
        print(
            "⚠️  Warning: Ollama server not reachable at http://localhost:11434\n"
            "   The UI will still launch, but diagnosis requests will fail until "
            "you run `ollama serve`."
        )

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        theme=THEME,
        css=CUSTOM_CSS,
    )