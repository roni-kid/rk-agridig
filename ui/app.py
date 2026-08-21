"""
app.py — RK AgriDig
Gradio web interface for farmer interaction. Connects to the local
Ollama server via src/ollama_client.py — no cloud calls, fully offline.

Built for the Africa Deep Tech Challenge 2026.
Team: Aaron Baidoo (RoniKid) & Firdaus Kudus (github.com/KudusFirdaus)
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
# Config & Helpers
# ---------------------------------------------------------------------------
MAX_QUESTION_CHARS = 500
CROPS = ["Maize", "Pepper", "Tomato"]
QUESTION_TYPES = ["Identification", "Treatment", "Prevention"]
CROP_EMOJIS = {
    "Maize": "🌽",
    "Pepper": "🌶️",
    "Tomato": "🍅",
}
# Placeholder paths for generated crop images. 
# Place the generated images in a folder named 'assets' next to app.py.
CROP_IMAGE_PATHS = {
    "Maize": "/assets/maize.png",
    "Pepper": "/assets/cpepper.png",
    "Tomato": "/assets/tomato1.png",
}

client = OllamaClient()

# ---------------------------------------------------------------------------
# Twi Translation Dictionary
# ---------------------------------------------------------------------------
TRANSLATIONS = {
    "en": {
        "header_title": "RK AgriDig",
        "offline_mode": "Offline Mode: Enabled",
        "language": "Language",
        "col1_title": "1 Diagnose a New Problem",
        "select_crop": "Select Crop",
        "question_type": "Question Type",
        "identify": "Identification",
        "treat": "Treatment",
        "prevent": "Prevention",
        "describe_symptoms": "Describe Symptoms",
        "symptoms_placeholder": "Explain what you see on your crop leaves (in simple English or Twi)...",
        "upload_image": "Upload Crop Image (Optional)",
        "submit_title": "2 Submit",
        "diagnose_btn": "Get Expert Diagnosis",
        "analyzing_btn": "Analyzing... (~25s)",
        "col2_title": "3 Expert Diagnosis & Advice",
        "no_diag_running": "No Diagnosis running",
        "results_placeholder": "Diagnosis results will appear here.",
        "col3_title": "4 Your Scan History",
        "search_history": "Search...",
        "reload_history": "Reload past results",
        "empty_history": "No queries this session.",
        "clear_btn": "Clear",
        "footer_text": "Grounded in GhanaAgricVQA dataset — verify with a local extension officer."
    },
    "tw": {
        "header_title": "RK AgriDig",
        "offline_mode": "Ɔman mu nhyehye: Ɛyɛ",
        "language": "Kasa",
        "col1_title": "1 Hwehwɛ Ɔyare foforo",
        "select_crop": "Paw Ɔdua",
        "question_type": "Abisa Kwan",
        "identify": "Hunhu",
        "treat": "Adwuma",
        "prevent": "Akwan a wɔfa so siw ano",
        "describe_symptoms": "Kyerkyerɛ Nsɛnkyerɛnne",
        "symptoms_placeholder": "Kyerɛkyerɛ nea wuhu wɔ wo ɔdua nhaban so (wɔ Borɔfo anaa Twi mu)...",
        "upload_image": "Fa Ɔdua Mfonini kɔ mu (Ɛnyɛ dm)",
        "submit_title": "2 Fa Kɔ",
        "diagnose_btn": "Hwehwɛ Ɔyare",
        "analyzing_btn": "Ɛrehwehwɛ... (~25s)",
        "col2_title": "3 yare Hunhu ne Afotu",
        "no_diag_running": "Ɔyare hunhu bi nni hɔ",
        "results_placeholder": "Ɔyare hunhu bɛba ha.",
        "col3_title": "4 Wo Hwehwɛ Abakɔsɛm",
        "search_history": "Hwehwɛ...",
        "reload_history": "San fa abakɔsɛm no bra",
        "empty_history": "Abisa biara nni hɔ wɔ nnɛ.",
        "clear_btn": "Pepa",
        "footer_text": "Yɛde GhanaAgricVQA data na yɛyɛ — kɔsra ɔyare ho ɔbenfo."
    }
}

def get_crop_selector_html(selected_crop="Maize"):
    """Generates HTML for the crop image selector."""
    html = '<div class="crop-selector-row">'
    for crop in CROPS:
        path = CROP_IMAGE_PATHS.get(crop, "")
        is_active = "active" if crop == selected_crop else ""
        # Using a placeholder div if image not found, otherwise img tag
        html += f'''
        <div class="crop-icon-btn {is_active}" data-crop="{crop}">
            <div class="crop-img-placeholder" style="background-image: url('{path}'); background-size: cover; background-position: center;"></div>
            <span class="crop-name">{crop}</span>
        </div>
        '''
    html += '</div>'
    return html

# ---------------------------------------------------------------------------
# Styling — Dark Tactile Aesthetic (RK AgriDig Branding)
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
:root {
    --rk-navy: #061d32;
    --rk-navy2: #092941;
    --rk-panel: #0d3048;
    --rk-green: #83c62d;
    --rk-green-dark: #4f9f2a;
    --rk-green-light: #a8dd43;
    --rk-blue: #26b7ee;
    --rk-orange: #f5a817;
    --rk-bg-dark: #08263b;
    --rk-card-dark: rgba(255,255,255,.035);
    --rk-card-border: rgba(255,255,255,.09);
    --rk-text-light: #f3f7f9;
    --rk-text-muted: #91a7b6;
    --rk-danger: #ed6b5d;
}
body, .gradio-container {
    background: radial-gradient(circle at 80% 5%, rgba(38,183,238,.08), transparent 28%),
                linear-gradient(135deg, #041725, #08263b 55%, #061b2e) !important;
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    color: var(--rk-text-light);
}
.gradio-container {
    max-width: 1400px !important;
    margin: 0 auto;
    padding: 20px;
}

/* Header Styling */
.rk-header-bar {
    background: radial-gradient(circle at 82% 18%, rgba(38,183,238,.16), transparent 30%),
                linear-gradient(135deg, #0c344d, #0b283f);
    border: 1px solid var(--rk-card-border);
    border-radius: 22px;
    padding: 18px 26px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 20px 60px rgba(0,0,0,.24);
}
.rk-header-bar h1 {
    color: var(--rk-text-light) !important;
    font-size: 1.7em !important;
    font-weight: 800 !important;
    letter-spacing: -0.5px;
    margin: 0 !important;
    display: flex;
    align-items: center;
    gap: 10px;
}
.rk-header-bar h1::before {
    content: "";
    display: inline-block;
    width: 34px;
    height: 34px;
    border: 2px solid var(--rk-orange);
    border-radius: 11px;
}
.header-controls {
    display: flex;
    align-items: center;
    gap: 24px;
    color: var(--rk-text-light);
    font-weight: 500;
}
.offline-badge {
    display: flex;
    align-items: center;
    gap: 8px;
    background: rgba(131,198,45,.1);
    border: 1px solid rgba(131,198,45,.25);
    padding: 6px 13px;
    border-radius: 99px;
    font-size: 0.85em;
    color: #b8dc78;
}
.offline-dot {
    width: 8px;
    height: 8px;
    background: var(--rk-green);
    border-radius: 50%;
    box-shadow: 0 0 10px var(--rk-green);
}
.lang-toggle {
    display: flex;
    gap: 4px;
    background: rgba(255,255,255,.04);
    border: 1px solid var(--rk-card-border);
    padding: 4px;
    border-radius: 10px;
}
.lang-btn {
    background: transparent;
    border: none;
    color: var(--rk-text-muted);
    padding: 7px 16px;
    border-radius: 7px;
    cursor: pointer;
    font-weight: 700;
    font-size: 0.85em;
    transition: all 0.2s;
}
.lang-btn.active {
    background: linear-gradient(135deg, var(--rk-green), var(--rk-green-dark));
    color: #071c10;
}

/* Tactile/Clay Card Styling */
.tactile-card {
    background: var(--rk-card-dark);
    border-radius: 18px;
    padding: 24px;
    box-shadow: 0 20px 60px rgba(0,0,0,.24);
    border: 1px solid var(--rk-card-border);
    height: 100%;
}
.panel-title {
    font-size: 1.05em;
    font-weight: 800;
    color: var(--rk-text-light);
    margin-bottom: 20px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.panel-title .step-num {
    background: linear-gradient(135deg, var(--rk-green), var(--rk-green-dark));
    color: #071c10;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8em;
    font-weight: 800;
}

/* Crop Selector */
.crop-selector-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    justify-content: space-around;
}
.crop-icon-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    cursor: pointer;
    padding: 8px;
    border-radius: 12px;
    transition: all 0.2s;
    border: 2px solid transparent;
}
.crop-icon-btn:hover, .crop-icon-btn.active {
    background: rgba(131,198,45,.12);
    border-color: var(--rk-green);
}
.crop-img-placeholder {
    width: 60px;
    height: 60px;
    border-radius: 50%;
    background: rgba(255,255,255,.05);
    border: 2px solid var(--rk-card-border);
}
.crop-name {
    font-size: 0.8em;
    color: var(--rk-text-muted);
    font-weight: 600;
}

/* Form Elements */
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid var(--rk-card-border) !important;
    color: var(--rk-text-light) !important;
    border-radius: 10px !important;
}
.gradio-container input:focus, .gradio-container textarea:focus, .gradio-container select:focus {
    border-color: rgba(38,183,238,.5) !important;
}
.gradio-container label {
    color: var(--rk-text-muted) !important;
    font-weight: 600 !important;
    font-size: 0.9em !important;
}

/* Submit Button */
.submit-btn-large {
    background: linear-gradient(135deg, var(--rk-green), var(--rk-green-dark)) !important;
    color: #071c10 !important;
    font-size: 1.05em !important;
    font-weight: 800 !important;
    padding: 16px !important;
    border-radius: 11px !important;
    border: none !important;
    box-shadow: 0 10px 25px rgba(131,198,45,.2) !important;
    transition: all 0.2s !important;
    width: 100%;
}
.submit-btn-large:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 14px 30px rgba(131,198,45,.3) !important;
}
.submit-btn-large:disabled {
    opacity: 0.6;
    cursor: not-allowed;
}

/* Medical Report Layout */
.med-report-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid var(--rk-green);
    padding-bottom: 16px;
    margin-bottom: 20px;
}
.med-disease {
    font-size: 1.4em;
    font-weight: 800;
    color: var(--rk-text-light);
}
.med-pills { display: flex; gap: 8px; }
.pill {
    padding: 6px 14px;
    border-radius: 99px;
    font-size: 0.85em;
    font-weight: 800;
}
.pill-conf { background: linear-gradient(135deg, var(--rk-green), var(--rk-green-dark)); color: #071c10; }
.pill-time { background: rgba(38,183,238,.15); color: var(--rk-blue); border: 1px solid rgba(38,183,238,.3); }

.med-section {
    background: rgba(255,255,255,.03);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
    border-left: 4px solid var(--rk-text-muted);
}
.med-sec-symptoms { border-left-color: #a855f7; }
.med-sec-treatment { border-left-color: var(--rk-orange); }
.med-sec-prevention { border-left-color: var(--rk-green); }
.med-sec-title {
    font-size: 0.85em;
    text-transform: uppercase;
    letter-spacing: .6px;
    font-weight: 800;
    margin-bottom: 8px;
    color: var(--rk-text-muted);
    display: flex;
    align-items: center;
    gap: 8px;
}
.med-sec-content {
    color: var(--rk-text-light);
    line-height: 1.6;
    font-size: 0.95em;
}

/* Status Box */
.status-box {
    background: rgba(255,255,255,.025);
    border: 1px dashed rgba(131,198,45,.35);
    border-radius: 13px;
    padding: 20px;
    text-align: center;
    color: var(--rk-text-muted);
    margin-bottom: 20px;
    min-height: 80px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}

/* History Panel */
.history-search {
    margin-bottom: 16px;
}
.history-list {
    max-height: 400px;
    overflow-y: auto;
}
.history-item {
    background: rgba(255,255,255,.025);
    border: 1px solid var(--rk-card-border);
    border-radius: 10px;
    padding: 12px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hist-icon { font-size: 1.5em; }
.hist-info { flex: 1; }
.hist-crop { color: var(--rk-green-light); font-weight: 700; font-size: 0.9em; }
.hist-date { color: var(--rk-text-muted); font-size: 0.8em; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.12); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,.2); }

/* Footer credit */
.rk-footer {
    text-align: center;
    color: var(--rk-text-muted);
    font-size: 0.78em;
    padding: 22px 0 6px;
    border-top: 1px solid var(--rk-card-border);
    margin-top: 28px;
    line-height: 1.7;
}
.rk-footer a { color: var(--rk-blue); text-decoration: none; }
.rk-footer a:hover { text-decoration: underline; }

footer { display: none !important; }
"""

THEME = gr.themes.Soft(
    primary_hue="lime",
    secondary_hue="cyan",
).set(
    body_background_fill="#08263b",
    block_background_fill="rgba(255,255,255,.035)",
    block_border_color="rgba(255,255,255,.09)",
    body_text_color="#f3f7f9",
    button_primary_background_fill="#83c62d",
    button_primary_text_color="#071c10",
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
    and formats the result as HTML for display, plus updates
    the running chat-style history.
    """
    question = (question or "").strip()
    if not question:
        warning = "<div class='tactile-card' style='text-align:center; color:#f59e0b;'>⚠️ Please describe the symptoms before submitting.</div>"
        return warning, history, history

    if len(question) > MAX_QUESTION_CHARS:
        question = question[:MAX_QUESTION_CHARS]

    if not client.health_check():
        error_md = (
            "<div class='tactile-card' style='border: 1px solid #ef4444;'>"
            "<h3 style='color:#ef4444; margin-top:0;'>️ Model server not reachable</h3>"
            "<p>The on-device AI model isn't running. Please start it with: <code>ollama serve</code></p>"
            "</div>"
        )
        return error_md, history, history

    start = time.time()
    result: InferenceResult = client.infer(
        question=question, crop=crop, question_type=question_type
    )
    elapsed = time.time() - start
    response_html = _format_response(result, elapsed)

    entry = {
        "question": question,
        "crop": crop,
        "question_type": question_type,
        "timestamp": time.strftime("%H:%M:%S"),
    }
    new_history = history + [entry]
    return response_html, new_history, new_history

def _format_response(result: InferenceResult, elapsed: float) -> str:
    """Formats the InferenceResult into a clean, structured HTML medical report."""
    if not result.success:
        return (
            f"<div class='tactile-card'>"
            f"<h3 style='color:#ef4444; margin-top:0;'>️ Diagnosis Failed</h3>"
            f"<p>{result.error or 'Unknown error.'}</p><pre>{result.raw_text}</pre>"
            f"</div>"
        )

    confidence_pct = int(result.structured_confidence * 100)
    if result.structured_confidence < 0.5:
        return (
            f"<div class='tactile-card'>"
            f"<div style='margin-bottom:12px;'><span class='pill pill-time'>⏱ {elapsed:.1f}s</span></div>"
            f"<div class='med-sec-content'>{result.raw_text}</div>"
            f"<p style='color:#64748b; font-size:0.8em; font-style:italic; margin-top:16px;'>Response shown as-is — structure matching failed.</p>"
            f"</div>"
        )

    disease_name = result.disease if result.disease else "Diagnosis Complete"
    parts = [
        f"<div class='tactile-card'>\n",
        f"  <div class='med-report-header'>\n",
        f"    <div class='med-disease'>🦠 {disease_name}</div>\n",
        f"    <div class='med-pills'>\n",
        f"      <span class='pill pill-conf'>{confidence_pct}% Confidence</span>\n",
        f"      <span class='pill pill-time'>⏱ {elapsed:.1f}s</span>\n",
        f"    </div>\n",
        f"  </div>\n"
    ]

    if result.symptoms:
        parts.append(
            f"  <div class='med-section med-sec-symptoms'>\n"
            f"    <div class='med-sec-title'>🩺 Symptoms</div>\n"
            f"    <div class='med-sec-content'>{result.symptoms}</div>\n"
            f"  </div>\n"
        )
    if result.treatment:
        parts.append(
            f"  <div class='med-section med-sec-treatment'>\n"
            f"    <div class='med-sec-title'>🛠️ Treatment Plan</div>\n"
            f"    <div class='med-sec-content'>{result.treatment}</div>\n"
            f"  </div>\n"
        )
    if result.prevention:
        parts.append(
            f"  <div class='med-section med-sec-prevention'>\n"
            f"    <div class='med-sec-title'>🛡️ Prevention Guide</div>\n"
            f"    <div class='med-sec-content'>{result.prevention}</div>\n"
            f"  </div>\n"
        )

    parts.append(
        "  <div style='text-align:center; color:#64748b; font-size:0.8em; margin-top:20px; font-style:italic;'>"
        "Grounded in GhanaAgricVQA dataset — verify with a local extension officer."
        "  </div>\n</div>"
    )
    return "".join(parts)

def format_history_display(history: list) -> str:
    """Renders the running history into custom HTML cards."""
    if not history:
        return "<div style='color:#64748b; font-style:italic; text-align:center; padding:20px;'>No queries this session.</div>"
    
    lines = []
    for entry in reversed(history[-10:]):
        emoji = CROP_EMOJIS.get(entry['crop'], "🌱")
        lines.append(
            f"<div class='history-item'>"
            f"  <div class='hist-icon'>{emoji}</div>"
            f"  <div class='hist-info'>"
            f"    <div class='hist-crop'>{entry['crop']} - {entry['question_type']}</div>"
            f"    <div class='hist-date'>{entry['timestamp']}</div>"
            f"  </div>"
            f"</div>"
        )
    return "<div class='history-list'>" + "\n".join(lines) + "</div>"

def clear_all():
    return "", [], "<div style='color:#64748b; font-style:italic; text-align:center; padding:20px;'>No queries this session.</div>", []

# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
with gr.Blocks(title="RK AgriDig", theme=THEME, css=CUSTOM_CSS) as demo:
    history_state = gr.State([])
    lang_state = gr.State("en")

    # --- HEADER ---
    with gr.Row(elem_classes="rk-header-bar"):
        with gr.Column(scale=1):
            gr.HTML('<h1>RK AgriDig</h1>')
        with gr.Column(scale=2):
            gr.HTML('''
            <div class="header-controls">
                <div class="offline-badge">
                    <div class="offline-dot"></div>
                    <span>Offline Mode: Enabled</span>
                </div>
                <div class="lang-toggle">
                    <button class="lang-btn active" id="btn-lang-en">English</button>
                    <button class="lang-btn" id="btn-lang-tw">Twi</button>
                </div>
            </div>
            ''')

    # --- MAIN CONTENT ---
    with gr.Row():
        # LEFT COLUMN: INPUT & ACTIONS
        with gr.Column(scale=3, elem_classes="tactile-card"):
            gr.HTML('<div class="panel-title"><span class="step-num">1</span> Diagnose a New Problem</div>')
            
            # Crop Selector Visuals
            crop_visualizer = gr.HTML(value=get_crop_selector_html("Maize"))
            
            crop_dropdown = gr.Dropdown(
                choices=CROPS,
                value="Maize",
                label="Select Crop",
                elem_id="crop-dropdown"
            )
            
            question_type_radio = gr.Radio(
                choices=QUESTION_TYPES,
                value="Identification",
                label="Question Type",
                elem_id="question-type-radio"
            )
            
            question_input = gr.Textbox(
                label="Describe Symptoms",
                placeholder="Explain what you see on your crop leaves (in simple English or Twi)...",
                lines=4,
                max_lines=6,
                elem_id="question-input"
            )

            # NEW: Image Upload
            image_input = gr.Image(
                label="Upload Crop Image (Optional)",
                type="filepath",
                height=150,
                elem_id="image-input"
            )

            gr.HTML('<div class="panel-title" style="margin-top:24px;"><span class="step-num">2</span> Submit</div>')
            
            with gr.Row():
                submit_btn = gr.Button(
                    "Get Expert Diagnosis", 
                    variant="primary", 
                    size="lg",
                    elem_classes="submit-btn-large",
                    elem_id="submit-btn"
                )
                clear_btn = gr.Button("Clear", variant="secondary", size="lg", elem_id="clear-btn")

        # CENTER COLUMN: DIAGNOSIS & HISTORY
        with gr.Column(scale=5, elem_classes="tactile-card"):
            gr.HTML('<div class="panel-title"><span class="step-num">3</span> Expert Diagnosis & Advice</div>')
            
            response_output = gr.HTML(
                value="<div class='status-box'>Diagnosis results will appear here.</div>",
                elem_id="response-output"
            )

        # RIGHT COLUMN: STATUS & SCAN HISTORY
        with gr.Column(scale=2):
            # Status Box
            status_box = gr.HTML(
                value="<div class='status-box'>No Diagnosis running</div>",
                elem_id="status-box"
            )
            
            # History Panel
            with gr.Column(elem_classes="tactile-card"):
                gr.HTML('<div class="panel-title"><span class="step-num">4</span> Your Scan History</div>')
                
                history_search = gr.Textbox(
                    placeholder="Search...",
                    label="",
                    show_label=False,
                    elem_classes="history-search",
                    elem_id="history-search"
                )
                
                history_display = gr.HTML(
                    value="<div style='color:#64748b; font-style:italic; text-align:center; padding:20px;'>No queries this session.</div>",
                    elem_id="history-display"
                )
                
                reload_btn = gr.Button("Reload past results", variant="secondary", size="sm", elem_id="reload-btn")

    # --- FOOTER ---
    gr.HTML('''
    <div class="rk-footer">
        RK AgriDig &middot; Africa Deep Tech Challenge 2026 &middot; Offline-first crop diagnosis<br>
        Built by Aaron Baidoo (RoniKid) &amp; Firdaus Kudus &middot;
        <a href="https://github.com/KudusFirdaus" target="_blank" rel="noopener">github.com/KudusFirdaus</a>
    </div>
    ''')

    # --- LANGUAGE TOGGLE LOGIC ---
    def update_language(lang):
        t = TRANSLATIONS[lang]
        return {
            crop_dropdown: gr.update(label=t["select_crop"]),
            question_type_radio: gr.update(label=t["question_type"], choices=[t["identify"], t["treat"], t["prevent"]]),
            question_input: gr.update(label=t["describe_symptoms"], placeholder=t["symptoms_placeholder"]),
            image_input: gr.update(label=t["upload_image"]),
            submit_btn: gr.update(value=t["diagnose_btn"]),
            clear_btn: gr.update(value=t["clear_btn"]),
            history_search: gr.update(placeholder=t["search_history"]),
            reload_btn: gr.update(value=t["reload_history"]),
        }

    # Note: Gradio JS for button active state toggling
    lang_js = """
    function(lang) {
        document.querySelectorAll('.lang-btn').forEach(b => b.classList.remove('active'));
        if (lang === 'en') document.getElementById('btn-lang-en').classList.add('active');
        else document.getElementById('btn-lang-tw').classList.add('active');
        return lang;
    }
    """
    
    # We use two buttons that trigger the same logic
    btn_en = gr.Button("English", visible=False, elem_id="btn-lang-en-trigger")
    btn_tw = gr.Button("Twi", visible=False, elem_id="btn-lang-tw-trigger")

    btn_en.click(fn=lambda: "en", outputs=[lang_state]).then(
        fn=update_language, inputs=[lang_state], 
        outputs=[crop_dropdown, question_type_radio, question_input, image_input, submit_btn, clear_btn, history_search, reload_btn]
    )
    btn_tw.click(fn=lambda: "tw", outputs=[lang_state]).then(
        fn=update_language, inputs=[lang_state], 
        outputs=[crop_dropdown, question_type_radio, question_input, image_input, submit_btn, clear_btn, history_search, reload_btn]
    )

    # Inject JS for header button clicks
    demo.load(js="""
    document.getElementById('btn-lang-en').addEventListener('click', () => document.getElementById('btn-lang-en-trigger').click());
    document.getElementById('btn-lang-tw').addEventListener('click', () => document.getElementById('btn-lang-tw-trigger').click());
    """)

    # --- Wiring (UNCHANGED LOGIC) ---
    # Submit Click Chain
    submit_btn.click(
        fn=lambda: gr.update(interactive=False, value="Analyzing... (~25s)"),
        outputs=[submit_btn]
    ).then(
        fn=diagnose,
        inputs=[question_input, crop_dropdown, question_type_radio, history_state],
        outputs=[response_output, history_state, history_state],
    ).then(
        fn=format_history_display,
        inputs=[history_state],
        outputs=[history_display],
    ).then(
        fn=lambda: gr.update(interactive=True, value="Get Expert Diagnosis"),
        outputs=[submit_btn]
    )

    # Enter Key Chain (Textbox Submit)
    question_input.submit(
        fn=lambda: gr.update(interactive=False, value="Analyzing... (~25s)"),
        outputs=[submit_btn]
    ).then(
        fn=diagnose,
        inputs=[question_input, crop_dropdown, question_type_radio, history_state],
        outputs=[response_output, history_state, history_state],
    ).then(
        fn=format_history_display,
        inputs=[history_state],
        outputs=[history_display],
    ).then(
        fn=lambda: gr.update(interactive=True, value="Get Expert Diagnosis"),
        outputs=[submit_btn]
    )

    # Clear Chain
    clear_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[question_input, history_state, history_display, history_state],
    ).then(
        fn=lambda: "<div class='status-box'>Diagnosis results will appear here.</div>",
        outputs=[response_output]
    )

if __name__ == "__main__":
    print("Checking Ollama server before launch...")
    if not client.health_check():
        print(
            "⚠️  Warning: Ollama server not reachable at http://localhost:11434\n"
            "   The UI will still launch, but diagnosis requests will fail until  "
            "you run `ollama serve`."
        )
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )