"""
app.py — RK AgriDig
Gradio web interface for farmer interaction. Connects to the local
Ollama server via src/ollama_client.py — no cloud calls, fully offline.

UI matches the "PC Dashboard" mockup (index.html): a fixed sidebar with
view-switching navigation, rendered as gr.Column visibility toggles.

Every button on the page does something:
  - Sidebar nav + the two dashboard shortcut buttons switch pages.
  - Get Expert Diagnosis / Clear / Enter-to-submit run and reset diagnosis.
  - The three Knowledge Base "Open guide" buttons swap in real crop content.
  - English / Twi actually re-labels every visible string, not just a few.
  - Dashboard stat + weekly chart update live from this session's real scans.

Session state (scan history, weekly counts, language) lives in gr.State,
not module globals — each visitor gets their own, so one farmer's history
never leaks into another's browser tab.

Built for the Africa Deep Tech Challenge 2026.
Team: Aaron Baidoo (RoniKid) & Firdaus Kudus (github.com/KudusFirdaus)
"""
from __future__ import annotations
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
import gradio as gr
import numpy as np

# Allow `python ui/app.py` to find the sibling src/ package regardless of cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ollama_client import OllamaClient, InferenceResult  # noqa: E402
from src.knowledge_base import render_crop_guide_html  # noqa: E402
from src.image_symptom_extractor import build_augmented_question, ImageSymptomSignals  # noqa: E402
from src.voice_output import speak_diagnosis, speech_status_message  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MAX_QUESTION_CHARS = 500

# Phase 1 profiling found sustained back-to-back inference pushes CPU temps
# to 95–98°C on the target hardware — above the ADTC 85°C disqualification
# threshold. Enforcing a minimum cooldown between requests gives the CPU time
# to drop back below safe operating temperature before the next inference run.
DEFAULT_COOLDOWN_SECONDS = 15
COOLDOWN_OPTIONS = [10, 15, 20, 25, 30]

CROPS = ["Maize", "Pepper", "Tomato"]
QUESTION_TYPES = ["Identification", "Treatment", "Prevention"]
CROP_EMOJIS = {"Maize": "🌽", "Pepper": "🌶️", "Tomato": "🍅"}

# Minimal single-color vector icons for the crop picker (Scan a Leaf tab).
# Hand-authored, no external icon library dependency — keeps the offline
# tool free of any CDN/network requirement for UI chrome. Each is a flat
# outline silhouette (not a literal photo-realistic drawing) so it reads
# clearly at card size and recolors cleanly via `currentColor`.
CROP_SVG_ICONS = {
    "Maize": """<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M32 6c6 8 9 18 9 28 0 12-4 20-9 24-5-4-9-12-9-24 0-10 3-20 9-28z" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/>
        <path d="M32 14v42" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        <path d="M25 20c2 1 4 1 7 1M39 20c-2 1-4 1-7 1M24 28c2.5 1.2 5 1.2 8 1.2M40 28c-2.5 1.2-5 1.2-8 1.2M24 36c2.5 1.2 5 1.2 8 1.2M40 36c-2.5 1.2-5 1.2-8 1.2M25 44c2 1 4.5 1 7 1M39 44c-2 1-4.5 1-7 1" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>
        <path d="M20 12c-3 3-4 8-3 12M44 12c3 3 4 8 3 12" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
    </svg>""",
    "Pepper": """<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M30 10c-1-3-4-4-7-3" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
        <path d="M29 9c2 2 2 5 1 7" stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
        <path d="M30 16c9 0 15 8 14 19-1 11-8 21-16 21-9 0-14-9-14-19 0-11 7-21 16-21z" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/>
        <path d="M22 24c-2 6-2 14 1 21" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>
    </svg>""",
    "Tomato": """<svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M32 20c10 0 17 8 17 18s-7 18-17 18-17-8-17-18 7-18 17-18z" stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/>
        <path d="M32 20v-4M26 17l-3-5M38 17l3-5M22 19c-2-2-2-4-1-6M42 19c2-2 2-4 1-6" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M20 34c1-5 6-8 12-8s11 3 12 8" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" opacity="0.6"/>
    </svg>""",
}

VIEWS = ["dashboard", "scan", "treatments", "knowledge", "history", "settings"]
NAV_ICONS = {
    "dashboard": "⌂",
    "scan": "⌁",
    "treatments": "✚",
    "knowledge": "▤",
    "history": "◷",
    "settings": "⚙",
}

client = OllamaClient()

# ---------------------------------------------------------------------------
# Language content — full bilingual coverage (English / Twi), ported and
# extended from the project's original prompt-engineered strings.
# ---------------------------------------------------------------------------
TEXTS = {
    "en": {
        "nav": {
            "dashboard": "Dashboard", "scan": "Scan a Leaf", "treatments": "Treatments",
            "knowledge": "Knowledge Base", "history": "Scan History", "settings": "Settings",
        },
        "crumb": {
            "dashboard": "Dashboard", "scan": "Scan a Leaf", "treatments": "Treatments",
            "knowledge": "Knowledge Base", "history": "Scan History", "settings": "Settings",
        },
        "eyebrow": "OFFLINE AI CROP INTELLIGENCE",
        "hero_head": "Good day, farmer.",
        "hero_desc": (
            "Describe what you're seeing on maize, pepper, or tomato leaves — in "
            "English or Twi — and get a structured diagnosis with treatment and "
            "prevention steps, no internet required."
        ),
        "btn_scan": "＋ Start New Scan",
        "btn_history": "View History",
        "stat_scans": "Scans this session",
        "stat_local": "Local to this device",
        "stat_crops": "Crops supported",
        "stat_crops_sub": "Maize · Pepper · Tomato",
        "stat_cooldown": "Cooldown",
        "stat_cooldown_sub": "Between requests",
        "stat_offline": "Offline status",
        "stat_ready": "Ready",
        "stat_no_internet": "No internet required",
        "recent_scans": "Recent scans",
        "view_all": "View all",
        "no_scans_yet": "No scans yet this session — start one from the Scan Leaf tab.",
        "scan_title": "Scan a Leaf",
        "scan_sub": "Select a crop, describe the symptoms, and run the offline model.",
        "select_crop": "Select Crop",
        "question_type": "Question Type",
        "q_id": "Identification", "q_treat": "Treatment", "q_prev": "Prevention",
        "symptom_lbl": "Describe Symptoms",
        "symptom_ph": "E.g., My maize leaves have brown spots with yellow halos...",
        "image_lbl": "Add a leaf photo (optional)",
        "btn_submit": "Get Expert Diagnosis",
        "btn_analyzing": "Analyzing... (~25s)",
        "btn_clear": "Clear",
        "btn_voice": "🔊 Listen to diagnosis",
        "btn_voice_loading": "Generating audio...",
        "voice_unsupported_tw": "🔇 Voice narration isn't available in Twi yet — the offline speech engine has no Twi voice data.",
        "ai_diag": "AI Diagnosis",
        "local_model": "Local model",
        "empty_diag_title": "Your diagnosis will appear here",
        "empty_diag_sub": "Describe your crop's symptoms and run the offline model.",
        "treat_title": "Treatment Guide",
        "treat_sub": "Simple, field-ready actions linked to common diagnoses.",
        "know_title": "Knowledge Base",
        "know_sub": "Offline reference cards for crops, symptoms and diagnosis.",
        "open_guide": "Open guide",
        "hist_title": "Scan History",
        "hist_sub": "Your locally stored diagnosis records for this session.",
        "empty_hist": "No queries this session.",
        "sett_title": "Settings",
        "sett_sub": "Configure the desktop experience.",
        "sett_offline_h": "Offline-first mode",
        "sett_offline_p": "Keep diagnosis and history available without internet.",
        "sett_save_h": "Save scan history",
        "sett_save_p": "Store recent diagnoses on this device for this session.",
        "sett_cooldown_h": "Thermal cooldown",
        "sett_cooldown_p": "Minimum gap enforced between diagnosis requests to protect your hardware.",
        "sett_lang_h": "Language",
        "sett_lang_p": "Diagnosis output language.",
        "footer_grounding": "Grounded in GhanaAgricVQA dataset — verify with a local extension officer.",
        "weekly_activity": "Weekly activity",
        "scans_label": "Scans",
        "offline_banner_title": "100% Offline",
        "offline_banner_sub": "All core diagnosis features run locally.",
        "warn_empty": "⚠️ Please describe the symptoms before submitting.",
        "err_server_title": "⚠️ Model server not reachable",
        "err_server_body": "The on-device AI model isn't running. Please start it with: <code>ollama serve</code>",
        "guide_maize_h": "🌽 Maize Disease Reference",
        "guide_maize_common": "<b>Common Issues:</b> Northern Leaf Blight (long grayish streaks), Gray Leaf Spot (rectangular lesions).",
        "guide_maize_tip": "<b>Farmer Tip:</b> Ensure plant spacing allows proper airflow. Remove infected stalks after harvest to reduce spore carryover.",
        "guide_pepper_h": "🌶️ Pepper Disease Reference",
        "guide_pepper_common": "<b>Common Issues:</b> Bacterial Leaf Spot (dark water-soaked spots), Anthracnose (sunken fruit lesions).",
        "guide_pepper_tip": "<b>Farmer Tip:</b> Irrigate at the base rather than spraying foliage. Rotate with non-solanaceous crops.",
        "guide_tomato_h": "🍅 Tomato Disease Reference",
        "guide_tomato_common": "<b>Common Issues:</b> Early Blight (concentric-ring lesions), Late Blight (large dark water-soaked patches).",
        "guide_tomato_tip": "<b>Farmer Tip:</b> Mulch around plants to stop soil-borne spores splashing onto lower leaves during rain.",
    },
    "tw": {
        "nav": {
            "dashboard": "Dashboard", "scan": "Hwehwɛ Ahaban", "treatments": "Adwuma Akwan",
            "knowledge": "Nimdeɛ Adaka", "history": "Hwehwɛ Abakɔsɛm", "settings": "Nhyehyɛe",
        },
        "crumb": {
            "dashboard": "Dashboard", "scan": "Hwehwɛ Ahaban", "treatments": "Adwuma Akwan",
            "knowledge": "Nimdeɛ Adaka", "history": "Hwehwɛ Abakɔsɛm", "settings": "Nhyehyɛe",
        },
        "eyebrow": "ƆMAN MU NHYƐHYƐ A ƐYƐ ADWUMA",
        "hero_head": "Akwaaba, okuafo.",
        "hero_desc": (
            "Kyerɛkyerɛ nea wuhu wɔ aburow, mako, anaa ntoosi nhaban so — wɔ "
            "Borɔfo anaa Twi mu — na nya ɔyare hunhu ne adwuma ne akwan a wɔfa "
            "so siw ano, ɛnhia intanet."
        ),
        "btn_scan": "＋ Fi Hwehwɛ Foforo Ase",
        "btn_history": "Hwɛ Abakɔsɛm",
        "stat_scans": "Nhwehwɛmu wɔ saa berɛ yi",
        "stat_local": "Ɛwɔ saa dɛvaes yi so",
        "stat_crops": "Nnɔbaeɛ a wɔfa boa",
        "stat_crops_sub": "Aburow · Mako · Ntoosi",
        "stat_cooldown": "Nnɔso berɛ",
        "stat_cooldown_sub": "Ntam abisa",
        "stat_offline": "Ɔman mu tebea",
        "stat_ready": "Ɛyɛ",
        "stat_no_internet": "Enhia intanet",
        "recent_scans": "Nhwehwɛmu foforo",
        "view_all": "Hwɛ nyinaa",
        "no_scans_yet": "Nhwehwɛmu biara nni hɔ wɔ saa berɛ yi — fi ase wɔ Hwehwɛ Ahaban tab so.",
        "scan_title": "Hwehwɛ Ahaban",
        "scan_sub": "Paw ɔdua, kyerɛkyerɛ nsɛnkyerɛnne, na ma model no yɛ adwuma.",
        "select_crop": "Paw Ɔdua",
        "question_type": "Abisa Kwan",
        "q_id": "Hunhu", "q_treat": "Adwuma", "q_prev": "Akwan a wɔfa so siw ano",
        "symptom_lbl": "Kyerɛkyerɛ Nsɛnkyerɛnne",
        "symptom_ph": "Sɛ nhwɛso: Me aburow nhaban wɔ nsensanee tuntum a akokɔsradeɛ atwa ho hyia...",
        "image_lbl": "Fa ahaban mfonini bi ka ho (ɛnhia)",
        "btn_submit": "Hwehwɛ Ɔyare",
        "btn_analyzing": "Ɛrehwehwɛ... (~25s)",
        "btn_clear": "Pepa",
        "btn_voice": "🔇 Nne kenkan nni ha koraa Twi mu",
        "btn_voice_loading": "Ɛreyɛ nne...",
        "voice_unsupported_tw": "🔇 Nne kenkan nni ha koraa Twi mu — dɛvaes no nni Twi nne data.",
        "kb_tw_pending": "Nkyerɛmu a emu dɔ no wɔ Borɔfo kasa mu nko ara seesei — ɔyare nsɛmfua nkyerɛase retwɛn nhwehwɛmu.",
        "ai_diag": "AI Ɔyare Hunhu",
        "local_model": "Ɛwɔ dɛvaes so",
        "empty_diag_title": "Wo ɔyare hunhu bɛba ha",
        "empty_diag_sub": "Kyerɛkyerɛ wo ɔdua nsɛnkyerɛnne na ma model no yɛ adwuma.",
        "treat_title": "Adwuma Akwan Kwankyerɛ",
        "treat_sub": "Nneɛma a wobɛtumi ayɛ ntɛm a ɛfa ɔyare ahodoɔ ho.",
        "know_title": "Nimdeɛ Adaka",
        "know_sub": "Nsɛm a ɛfa nnɔbaeɛ, nsɛnkyerɛnne ne ɔyare hunhu ho — ɛwɔ hɔ ɛnhia intanet.",
        "open_guide": "Bue kwankyerɛ",
        "hist_title": "Hwehwɛ Abakɔsɛm",
        "hist_sub": "Wo ɔyare hunhu abakɔsɛm a ɛwɔ dɛvaes yi so wɔ saa berɛ yi.",
        "empty_hist": "Abisa biara nni hɔ wɔ saa berɛ yi.",
        "sett_title": "Nhyehyɛe",
        "sett_sub": "Sesa desktop no adwumayɛ ho nsɛm.",
        "sett_offline_h": "Ɔman mu nhyehyɛe a ɛdi kan",
        "sett_offline_p": "Ma ɔyare hunhu ne abakɔsɛm nyɛ adwuma a ɛnhia intanet.",
        "sett_save_h": "Sie hwehwɛ abakɔsɛm",
        "sett_save_p": "Sie ɔyare hunhu foforo wɔ dɛvaes yi so wɔ saa berɛ yi mu.",
        "sett_cooldown_h": "Ɔhyew nnɔso berɛ",
        "sett_cooldown_p": "Berɛ ketewa a ɛda ɔyare hunhu abisa ntam sɛ ɛbɛbɔ wo dɛvaes ho ban.",
        "sett_lang_h": "Kasa",
        "sett_lang_p": "Ɔyare hunhu nsɛm kasa.",
        "footer_grounding": "Yɛde GhanaAgricVQA data na yɛyɛ — kɔsra ɔyare ho ɔbenfo.",
        "weekly_activity": "Dapɛn mu adwuma",
        "scans_label": "Nhwehwɛmu",
        "offline_banner_title": "Ɔman mu 100%",
        "offline_banner_sub": "Nneɛma nyinaa yɛ adwuma wɔ dɛvaes yi so nko ara.",
        "warn_empty": "⚠️ Yɛsrɛ kyerɛkyerɛ nsɛnkyerɛnne no ansa na wode akɔ.",
        "err_server_title": "⚠️ Yɛntumi nnu model server no",
        "err_server_body": "Model no nnyɛ adwuma. Yɛsrɛ fi ase wɔ ha: <code>ollama serve</code>",
        "guide_maize_h": "🌽 Aburow Ɔyare Nimdeɛ",
        "guide_maize_common": "<b>Ɔyare a ɛtaa ba:</b> Northern Leaf Blight (nsensanee tuntum atenten), Gray Leaf Spot (ahyɛn ntetareɛ).",
        "guide_maize_tip": "<b>Afotusɛm:</b> Ma nnɔbaeɛ no ntam kwan mu yɛ da mframa nkɔ mu yiye. Yi nnɔbaeɛ a ɔyare aka no fi mfuo no mu wɔ otwa akyi.",
        "guide_pepper_h": "🌶️ Mako Ɔyare Nimdeɛ",
        "guide_pepper_common": "<b>Ɔyare a ɛtaa ba:</b> Bacterial Leaf Spot (nsensanee tuntum a nsuo wɔ mu), Anthracnose (aba ho apirakuro).",
        "guide_pepper_tip": "<b>Afotusɛm:</b> Gugu nsuo wɔ ase na mmfa ngu nhaban so. Sesa nnɔbaeɛ a ɛnyɛ Solanaceae abusua mufoɔ.",
        "guide_tomato_h": "🍅 Ntoosi Ɔyare Nimdeɛ",
        "guide_tomato_common": "<b>Ɔyare a ɛtaa ba:</b> Early Blight (ahyɛn a ɛda so so), Late Blight (nsuo wɔ mu ntetareɛ akɛseɛ tuntum).",
        "guide_tomato_tip": "<b>Afotusɛm:</b> Fa mmerɛ kata nnɔbaeɛ no ase kwan so na ɛnnyɛ osuo mu bio nsɛnkyerɛnne ntow nkɔ nhaban a ɛwɔ ase no so.",
    },
}

# ---------------------------------------------------------------------------
# Styling — index.html tokens, plus a few dark-card utility classes carried
# over cleanly from the comparison file where they were simpler than mine.
# ---------------------------------------------------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Sora:wght@600;700;800&display=swap');
:root{
  --navy:#061d32; --navy2:#092941; --panel:#0d3048; --panel2:#123a53;
  --green:#83c62d; --green2:#4f9f2a; --lime:#a8dd43; --blue:#26b7ee;
  --orange:#f5a817; --text:#f3f7f9; --muted:#91a7b6; --line:rgba(255,255,255,.09);
  --danger:#ed6b5d; --shadow:0 20px 60px rgba(0,0,0,.24);
}
body, .gradio-container {
    background: linear-gradient(135deg,#041725,#08263b 55%,#061b2e) !important;
    color: var(--text) !important;
    font-family: 'Inter', 'Segoe UI', Arial, sans-serif !important;
    font-feature-settings: "cv11", "ss01";
}
.rk-page-title, .rk-brand b, .rk-med-disease, .rk-card-head h2, .rk-guide-panel h4 {
    font-family: 'Sora', 'Inter', 'Segoe UI', Arial, sans-serif !important;
    letter-spacing: -0.01em;
}
.gradio-container { max-width: 100% !important; width: 100% !important; margin: 0 !important; padding: 0 !important; }
.gradio-container > .main, .gradio-container .contain { max-width: 100% !important; width: 100% !important; }
footer { display: none !important; }
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,.12); border-radius: 4px; }

.rk-app { display: flex !important; gap: 0 !important; align-items: flex-start !important; width: 100% !important; min-height: 100vh; }

/* ---------- Sidebar ---------- */
.rk-sidebar {
    background: rgba(3,22,37,.92) !important;
    border-right: 1px solid var(--line);
    padding: 24px 16px !important;
    min-width: 255px !important; max-width: 255px !important;
    position: sticky; top: 0;
}
.rk-brand { display:flex; align-items:center; gap:12px; padding:4px 12px 28px; }
.rk-logo { width:42px; height:42px; border:2px solid var(--orange); border-radius:14px; display:grid; place-items:center; color:var(--green); font-weight:900; font-size:19px; flex-shrink:0; }
.rk-brand b { font-size:20px; letter-spacing:-.5px; display:block; }
.rk-brand span { display:block; color:var(--muted); font-size:11px; margin-top:2px; }

.rk-sidebar .gr-button, .rk-sidebar button {
    border: 0 !important; background: transparent !important; color: #b9c9d2 !important;
    text-align: left !important; padding: 13px 14px !important; border-radius: 12px !important;
    font-size: 14px !important; font-weight: 500 !important; justify-content: flex-start !important;
    box-shadow: none !important;
}
.rk-sidebar button:hover {
    background: linear-gradient(90deg, rgba(132,198,45,.18), rgba(38,183,238,.06)) !important;
    color: white !important;
}
.rk-nav-btn.rk-nav-active {
    background: linear-gradient(90deg, rgba(132,198,45,.18), rgba(38,183,238,.06)) !important;
    color: white !important; box-shadow: inset 3px 0 var(--green) !important;
}

.rk-side-bottom {
    margin-top: 22px;
    background: linear-gradient(145deg, rgba(131,198,45,.14), rgba(245,168,23,.08));
    border: 1px solid rgba(131,198,45,.25); border-radius: 16px; padding: 16px;
}
.rk-side-bottom strong { font-size:13px; }
.rk-side-bottom p { color:var(--muted); font-size:11px; line-height:1.5; margin:7px 0 12px; }
.rk-status { display:flex; align-items:center; gap:8px; color:#b8dc78; font-size:11px; }
.rk-dot { width:7px; height:7px; border-radius:50%; background:var(--green); box-shadow:0 0 10px var(--green); display:inline-block; }

/* ---------- Main / topbar ---------- */
.rk-main { flex: 1 !important; min-width: 0 !important; width: 100% !important; padding: 24px 36px 36px !important; }
/* Content stays a readable width and centers within whatever space rk-main has, so
   text and cards don't stretch edge-to-edge on ultrawide screens — but the sidebar
   and dark background still fill the full viewport instead of leaving dead space
   at the sides on wide monitors. */
.rk-main > * { max-width: 1240px; margin-left: auto; margin-right: auto; }
.rk-topbar { display:flex !important; align-items:center; justify-content:space-between; margin-bottom: 22px !important; }
.rk-crumb { color: var(--muted); font-size: 13px; }
.rk-crumb b { color: white; }
.rk-topbar-right { display:flex; align-items:center; gap:10px; }
.rk-avatar {
    width:38px; height:38px; border-radius:12px;
    background: linear-gradient(135deg, var(--green), #2c7e30);
    display:grid; place-items:center; font-weight:800; color:#071c10;
}

/* ---------- Language toggle ---------- */
.rk-lang-toggle { display:flex; gap:4px; background: rgba(255,255,255,.04); border:1px solid var(--line); padding:4px; border-radius:10px; }
.rk-lang-toggle button {
    background: transparent !important; border:none !important; color: var(--muted) !important;
    padding: 7px 14px !important; border-radius:7px !important; font-weight:700 !important; font-size:.82em !important;
    box-shadow: none !important;
}
.rk-lang-btn-active {
    background: linear-gradient(135deg, var(--green), var(--green2)) !important; color:#071c10 !important;
}

/* ---------- Hero ---------- */
.rk-hero-card {
    border: 1px solid var(--line); border-radius: 22px; padding: 28px 30px;
    background: radial-gradient(circle at 75% 20%, rgba(38,183,238,.16), transparent 30%), linear-gradient(135deg,#0c344d,#0b283f);
    box-shadow: var(--shadow);
}
.rk-eyebrow { color: var(--green); font-size:12px; font-weight:800; letter-spacing:1.4px; text-transform:uppercase; }
.rk-hero-card h1 { font-size:32px; line-height:1.05; margin:9px 0 12px; letter-spacing:-1.2px; color: var(--text); }
.rk-hero-card p { color:#a8bac5; max-width:590px; line-height:1.6; font-size:14px; }

/* ---------- Buttons ---------- */
.rk-primary-btn, .rk-primary-btn button {
    border: 0 !important; border-radius: 11px !important; padding: 12px 17px !important;
    background: linear-gradient(135deg, var(--green), #5ea52d) !important;
    color: #071c10 !important; font-weight: 800 !important;
    box-shadow: 0 10px 25px rgba(131,198,45,.2) !important;
}
.rk-secondary-btn, .rk-secondary-btn button {
    border: 1px solid var(--line) !important; border-radius: 11px !important; padding: 11px 16px !important;
    background: rgba(255,255,255,.04) !important; color: white !important; font-weight: 700 !important;
}

/* ---------- Stats ---------- */
.rk-stat { background: rgba(255,255,255,.035); border: 1px solid var(--line); border-radius: 16px; padding: 18px; }
.rk-stat-top { display:flex; justify-content:space-between; color:var(--muted); font-size:12px; }
.rk-stat strong { display:block; font-size:27px; margin-top:9px; color: var(--text); }
.rk-trend { font-size:11px; color:#9ed45c; margin-top:6px; }

/* ---------- Generic cards ---------- */
.rk-card {
    background: rgba(255,255,255,.035) !important; border: 1px solid var(--line) !important;
    border-radius: 18px !important; padding: 20px !important;
}
.rk-card-head { display:flex; justify-content:space-between; align-items:center; margin-bottom:16px; }
.rk-card h2 { font-size:16px; margin:0; color: var(--text); }
.rk-muted { color: var(--muted); font-size:12px; }

.rk-scan-item {
    display:grid; grid-template-columns:46px 1fr auto; gap:12px; align-items:center;
    padding:10px; border-radius:12px; background: rgba(255,255,255,.025); margin-bottom: 9px;
}
.rk-thumb { width:46px; height:46px; border-radius:10px; font-size:22px; display:flex; align-items:center; justify-content:center; background: rgba(131,198,45,.1); }
.rk-scan-item b { font-size:13px; color: var(--text); }
.rk-scan-item small { display:block; color:var(--muted); margin-top:4px; }
.rk-badge { padding:5px 8px; border-radius:99px; font-size:10px; font-weight:800; background:rgba(131,198,45,.14); color:#a8dd43; white-space: nowrap; }
.rk-badge.warn { background:rgba(245,168,23,.13); color:#ffc75d; }

/* Weekly bar chart — heights are set inline per-bar from real scan counts */
.rk-bar-chart { height:150px; display:flex; align-items:flex-end; gap:10px; padding-top:15px; border-bottom: 1px solid var(--line); padding-bottom: 10px; }
.rk-bar-col { flex:1; display:flex; flex-direction:column; align-items:center; height:100%; justify-content:flex-end; }
.rk-bar-count { font-size:10px; color:var(--muted); margin-bottom:4px; }
.rk-bar { width:100%; background: linear-gradient(180deg, var(--blue), #155d86); border-radius:7px 7px 3px 3px; min-height:6px; }
.rk-bar-labels { display:flex; gap:10px; margin-top:6px; }
.rk-bar-labels span { flex:1; text-align:center; font-size:10px; color:var(--muted); }

.rk-offline {
    display:flex; align-items:center; gap:13px; padding:14px; border-radius:13px;
    background: rgba(131,198,45,.08); border: 1px solid rgba(131,198,45,.2); margin-top:16px;
}
.rk-offline-icon { width:38px; height:38px; border-radius:50%; display:grid; place-items:center; background:rgba(131,198,45,.13); color:var(--green); font-size:18px; flex-shrink:0; }

/* ---------- Page headers ---------- */
.rk-page-title { font-size:29px; margin:4px 0 7px; color: var(--text); }
.rk-page-sub { color:var(--muted); font-size:13px; margin-bottom:22px; }

/* ---------- Diagnosis result ---------- */
.rk-result-empty { min-height: 320px; display:flex; align-items:center; justify-content:center; text-align:center; color: var(--muted); padding: 40px; }
.rk-result-empty div { max-width: 270px; }
.rk-med-header { display:flex; justify-content:space-between; align-items:center; border-bottom: 2px solid var(--green); padding-bottom:16px; margin-bottom:20px; flex-wrap: wrap; gap: 10px; }
.rk-med-disease { font-size:1.3em; font-weight:800; color: var(--text); }
.rk-pill { padding:6px 14px; border-radius:99px; font-size:.85em; font-weight:800; white-space: nowrap; }
.rk-pill-conf { background: linear-gradient(135deg, var(--green), var(--green2)); color:#071c10; }
.rk-pill-time { background: rgba(38,183,238,.15); color: var(--blue); border:1px solid rgba(38,183,238,.3); }
.rk-med-section { background: rgba(255,255,255,.03); border-radius:12px; padding:16px; margin-bottom:16px; border-left:4px solid var(--muted); }
.rk-sec-symptoms { border-left-color:#a855f7; }
.rk-sec-treatment { border-left-color: var(--orange); }
.rk-sec-prevention { border-left-color: var(--green); }
.rk-sec-title { font-size:.85em; text-transform:uppercase; letter-spacing:.6px; font-weight:800; margin-bottom:8px; color:var(--muted); }
.rk-sec-content { color: var(--text); line-height:1.6; font-size:.95em; }
.rk-meter { height:7px; background:#173548; border-radius:99px; overflow:hidden; margin:8px 0 16px; }
.rk-meter i { display:block; height:100%; background: linear-gradient(90deg, var(--green), var(--lime)); border-radius:99px; }
.rk-confidence-row { display:flex; justify-content:space-between; font-size:12px; color:var(--muted); }

/* ---------- Tables ---------- */
.rk-card table { width:100%; border-collapse: collapse; }
.rk-card table th, .rk-card table td { text-align:left; padding:13px 9px; border-bottom:1px solid var(--line); font-size:12px; color: var(--text); }
.rk-card table th { color: var(--muted); font-weight:600; }

/* ---------- Crop icon picker (Scan a Leaf tab) ---------- */
.rk-crop-grid { display:flex; gap:12px; margin-bottom: 6px; }
.rk-crop-card, .rk-crop-card button {
    flex: 1 !important; display:flex !important; flex-direction:column !important; align-items:center !important;
    gap:8px !important; padding:16px 8px !important; border-radius:14px !important;
    border:1.5px solid var(--line) !important; background: rgba(255,255,255,.03) !important;
    color: var(--muted) !important; font-size:13px !important; font-weight:700 !important;
    box-shadow:none !important; transition: border-color .15s, background .15s, color .15s;
}
.rk-crop-card svg { width:34px; height:34px; }
.rk-crop-card button:hover { border-color: rgba(131,198,45,.4) !important; color: var(--text) !important; }
.rk-crop-card-active, .rk-crop-card-active button {
    border-color: var(--green) !important;
    background: linear-gradient(160deg, rgba(131,198,45,.16), rgba(38,183,238,.05)) !important;
    color: var(--text) !important;
}
.rk-crop-card-active svg { color: var(--green) !important; }
.rk-crop-picker-label { font-size:.85em; font-weight:600; color:var(--muted); margin-bottom:8px; display:block; }

/* ---------- Knowledge cards ---------- */
.rk-article { padding:18px; border:1px solid var(--line); border-radius:15px; background: rgba(255,255,255,.03); height: 100%; }
.rk-article .rk-articon { font-size:27px; }
.rk-article h3 { font-size:14px; margin: 8px 0 6px; color: var(--text); }
.rk-article p { font-size:11px; line-height:1.6; color:var(--muted); margin-bottom: 14px; }
.rk-guide-panel { background: rgba(255,255,255,.03); border: 1px solid rgba(131,198,45,.25); border-radius: 12px; padding: 16px; margin-top: 16px; }
.rk-guide-panel h4 { color: var(--green); margin: 0 0 10px; font-size: 14px; }
.rk-guide-panel p { font-size: 12px; line-height: 1.6; color: var(--text); margin: 6px 0; }

/* ---------- Settings ---------- */
.rk-setting { display:flex; justify-content:space-between; gap:20px; align-items:center; padding:18px 0; border-bottom:1px solid var(--line); }
.rk-setting h3 { font-size:14px; margin:0 0 5px; color: var(--text); }
.rk-setting p { margin:0; color:var(--muted); font-size:11px; }
#settings-card { max-width: 760px; }
#settings-card .row { border-bottom: 1px solid var(--line); padding: 4px 0; align-items: center !important; }
#settings-card .row:last-of-type { border-bottom: none; }
#settings-card h3 { font-size:14px; margin:0 0 5px; color: var(--text); }
#settings-card p { margin:0; color:var(--muted); font-size:11px; line-height: 1.5; }
.rk-toggle-btn, .rk-toggle-btn button {
    border-radius: 99px !important; font-weight: 800 !important; font-size: .8em !important;
    padding: 8px 4px !important; border: 1px solid var(--line) !important; box-shadow: none !important;
}
.rk-toggle-on, .rk-toggle-on button {
    background: linear-gradient(135deg, var(--green), var(--green2)) !important; color: #071c10 !important; border: none !important;
}
.rk-toggle-off, .rk-toggle-off button {
    background: rgba(255,255,255,.05) !important; color: var(--muted) !important;
}
#settings-card .gr-dropdown, #settings-card [data-testid="dropdown"] {
    background: rgba(255,255,255,.04) !important; border: 1px solid var(--line) !important; border-radius: 10px !important;
}

/* ---------- Status / cooldown box ---------- */
.status-box {
    background: rgba(255,255,255,.025); border: 1px dashed rgba(131,198,45,.35);
    border-radius: 13px; padding: 20px; text-align: center; color: var(--muted);
}

/* ---------- Gradio form-control overrides so they inherit the dark theme ---------- */
.gradio-container input, .gradio-container textarea, .gradio-container select {
    background: rgba(255,255,255,.04) !important; border: 1px solid var(--line) !important;
    color: var(--text) !important; border-radius: 10px !important;
}
.gradio-container label { color: var(--muted) !important; font-weight: 600 !important; font-size: .85em !important; }
.gradio-container .form, .gradio-container div.form { background: transparent !important; }
"""

THEME = gr.themes.Soft(primary_hue="lime", secondary_hue="cyan").set(
    body_background_fill="#08263b",
    block_background_fill="transparent",
    block_border_width="0px",
    body_text_color="#f3f7f9",
    button_primary_background_fill="#83c62d",
    button_primary_text_color="#071c10",
)

# ---------------------------------------------------------------------------
# Helpers — dynamic weekly chart (real per-session scan timestamps, ported
# from fapp.py's approach since it computes real data instead of hardcoding
# bar heights) and other pure-render functions.
# ---------------------------------------------------------------------------
def _weekly_counts(scan_times: list) -> dict:
    today = datetime.now().date()
    days = [today - timedelta(days=i) for i in range(6, -1, -1)]
    counts = {d.strftime("%a"): 0 for d in days}
    for t in scan_times:
        day_str = t.strftime("%a")
        if day_str in counts:
            counts[day_str] += 1
    return counts


def render_weekly_chart(scan_times: list, t: dict) -> str:
    counts = _weekly_counts(scan_times)
    max_val = max(counts.values()) or 1
    bars, labels = [], []
    for day, count in counts.items():
        height_pct = max(8, int((count / max_val) * 100)) if count else 8
        bars.append(
            f"<div class='rk-bar-col'><div class='rk-bar-count'>{count}</div>"
            f"<div class='rk-bar' style='height:{height_pct}%;'></div></div>"
        )
        labels.append(f"<span>{day}</span>")
    return (
        "<div class='rk-card-head'><h2>%s</h2><span class='rk-muted'>%s</span></div>"
        "<div class='rk-bar-chart'>%s</div>"
        "<div class='rk-bar-labels'>%s</div>"
        "<div class='rk-offline'><div class='rk-offline-icon'>⌁</div>"
        "<div><b>%s</b><div class='rk-muted'>%s</div></div></div>"
    ) % (
        t["weekly_activity"], t["scans_label"], "".join(bars), "".join(labels),
        t["offline_banner_title"], t["offline_banner_sub"],
    )


def _empty_history_html(t: dict) -> str:
    return f"<div style='color:#64748b; font-style:italic; text-align:center; padding:20px;'>{t['empty_hist']}</div>"


def dashboard_stat_html(history: list, t: dict) -> str:
    return (
        "<div class='rk-stat'><div class='rk-stat-top'>%s <span>⌁</span></div>"
        "<strong>%d</strong><div class='rk-trend'>%s</div></div>"
    ) % (t["stat_scans"], len(history), t["stat_local"])


def recent_scans_html(history: list, t: dict) -> str:
    recent = list(reversed(history[-3:]))
    if not recent:
        return f"<div class='rk-muted' style='padding:14px 0;'>{t['no_scans_yet']}</div>"
    return "".join(
        f"<div class='rk-scan-item'><div class='rk-thumb'>{CROP_EMOJIS.get(e['crop'], '🌱')}</div>"
        f"<div><b>{e['crop']} — {e['question_type']}</b><small>{e['timestamp']}</small></div>"
        f"<span class='rk-badge'>Logged</span></div>"
        for e in recent
    )


def dashboard_html(history: list, t: dict) -> str:
    return f"""
    <div style="display:flex; gap:25px; align-items:stretch; margin-bottom:22px; flex-wrap:wrap;">
      <div class="rk-hero-card" style="flex:1;">
        <div class="rk-eyebrow">{t['eyebrow']}</div>
        <h1>{t['hero_head']}</h1>
        <p>{t['hero_desc']}</p>
      </div>
    </div>
    """


def stats_row_html(history: list, t: dict, cooldown_seconds: int = DEFAULT_COOLDOWN_SECONDS) -> str:
    return f"""
    <div style="display:grid; grid-template-columns:repeat(4,1fr); gap:14px;">
      {dashboard_stat_html(history, t)}
      <div class="rk-stat"><div class="rk-stat-top">{t['stat_crops']} <span>◌</span></div><strong>3</strong><div class="rk-trend">{t['stat_crops_sub']}</div></div>
      <div class="rk-stat"><div class="rk-stat-top">{t['stat_cooldown']} <span>🌡</span></div><strong>{cooldown_seconds}s</strong><div class="rk-trend">{t['stat_cooldown_sub']}</div></div>
      <div class="rk-stat"><div class="rk-stat-top">{t['stat_offline']} <span>●</span></div><strong>{t['stat_ready']}</strong><div class="rk-trend">{t['stat_no_internet']}</div></div>
    </div>
    """


def recent_scans_card_html(history: list, t: dict) -> str:
    return f"""
    <div class="rk-card-head"><h2>{t['recent_scans']}</h2><span class="rk-muted">{t['view_all']}</span></div>
    {recent_scans_html(history, t)}
    """


TREATMENTS_ROWS = [
    ("🌽", "Maize", "Northern Leaf Blight", "Moderate", "Remove heavily affected leaves, improve field airflow, and apply an approved fungicide as directed."),
    ("🌶️", "Pepper", "Bacterial Leaf Spot", "High", "Remove infected foliage, avoid overhead watering, and rotate crops next season."),
    ("🍅", "Tomato", "Early Blight", "High", "Remove infected leaves, avoid overhead watering, and follow approved disease-control guidance."),
    ("🌽", "Maize", "Gray Leaf Spot", "Moderate", "Rotate crops, remove crop debris after harvest, and monitor neighboring plants."),
    ("🍅", "Tomato", "Late Blight", "High", "Remove and destroy affected plants promptly; do not compost infected material."),
]


def treatments_table_html() -> str:
    rows = "".join(
        f"<tr><td>{emoji} {crop}</td><td>{cond}</td>"
        f"<td><span class='rk-badge{' warn' if pri == 'High' else ''}'>{pri}</span></td><td>{action}</td></tr>"
        for emoji, crop, cond, pri, action in TREATMENTS_ROWS
    )
    return (
        "<div class='rk-card'><table>"
        "<thead><tr><th>Crop</th><th>Condition</th><th>Priority</th><th>Recommended action</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def knowledge_cards_html(t: dict) -> str:
    return f"""
    <div style="display:grid; grid-template-columns:repeat(3,1fr); gap:15px;">
      <div class="rk-article"><div class="rk-articon">🌽</div><h3>Maize diseases</h3>
        <p>Recognize Northern Leaf Blight and Gray Leaf Spot, and know what to inspect next.</p></div>
      <div class="rk-article"><div class="rk-articon">🌶️</div><h3>Pepper diseases</h3>
        <p>Compare bacterial leaf spot and anthracnose before deciding on treatment.</p></div>
      <div class="rk-article"><div class="rk-articon">🍅</div><h3>Tomato diseases</h3>
        <p>Distinguish early blight from late blight and other common leaf conditions.</p></div>
    </div>
    """


def guide_html(crop: str, t: dict) -> str:
    # Structured per-disease reference (symptoms/causes/treatment/prevention)
    # from src/knowledge_base.py, replacing the earlier two-sentence summary.
    # English-only for now — see knowledge_base.py docstring on why the Twi
    # translation of disease terminology needs a real review pass before
    # shipping rather than a rough guess.
    if t is TEXTS.get("tw"):
        return (
            f"<div class='rk-guide-panel'><p style='color:var(--muted); font-style:italic;'>"
            f"{t.get('kb_tw_pending', 'Detailed guide is English-only for now — translation of medical terms is pending review.')}"
            f"</p>{render_crop_guide_html(crop)}</div>"
        )
    return render_crop_guide_html(crop)


def settings_offline_row_html(t: dict) -> str:
    return (
        f"<div class='rk-setting'><div><h3>{t['sett_offline_h']}</h3>"
        f"<p>{t['sett_offline_p']}</p></div>"
        f"<span class='rk-muted' style='font-weight:700;'>Enabled</span></div>"
    )


def history_table_html(history: list, t: dict) -> str:
    if not history:
        return f"<div class='rk-card'><div class='rk-muted' style='padding:14px 0;'>{t['empty_hist']}</div></div>"
    rows = "".join(
        f"<tr><td>{e['timestamp']}</td><td>{e['crop']}</td><td>{e['question_type']}</td>"
        f"<td><span class='rk-badge'>Saved</span></td></tr>"
        for e in reversed(history)
    )
    return (
        "<div class='rk-card'><table>"
        "<thead><tr><th>Time</th><th>Crop</th><th>Question Type</th><th>Status</th></tr></thead>"
        f"<tbody>{rows}</tbody></table></div>"
    )


def empty_result_html(t: dict) -> str:
    return (
        "<div class='rk-result-empty'><div>"
        f"<div style='font-size:36px'>◉</div><b>{t['empty_diag_title']}</b>"
        f"<p>{t['empty_diag_sub']}</p></div></div>"
    )


# ---------------------------------------------------------------------------
# Core diagnosis handler — real Ollama call preserved from the working
# implementation (fapp.py's diagnosis is a hardcoded template with no model
# call at all, so that part of it is NOT adopted here).
# ---------------------------------------------------------------------------
def diagnose(question: str, crop: str, question_type: str, history: list,
             scan_times: list, last_request_time: float, lang: str,
             cooldown_seconds: int, save_history: bool, leaf_image: "np.ndarray | None" = None):
    t = TEXTS[lang]

    now = time.time()
    since_last = now - last_request_time
    if last_request_time > 0 and since_last < cooldown_seconds:
        wait = cooldown_seconds - since_last
        cooldown_html = (
            "<div class='status-box' style='border-color: rgba(245,168,23,.4);'>"
            "<span style='font-size:1.3em;'>🌡️</span>"
            f"<div style='color:#f5a817; font-weight:700; margin-top:8px;'>Cooling down ({wait:.0f}s remaining)</div>"
            "<div style='color:#91a7b6; font-size:0.85em; margin-top:4px;'>Protecting hardware — please wait between requests.</div>"
            "</div>"
        )
        return cooldown_html, history, scan_times, last_request_time

    question = (question or "").strip()

    # Combine typed symptoms with photo-derived signals, if a leaf photo was
    # provided. This runs classical CV heuristics (color/lesion analysis),
    # NOT the LLM — Phi-3-mini-4k-instruct has no vision tower. See
    # src/image_symptom_extractor.py for why this design was chosen over a
    # vision-model swap. image_bgr conversion (RGB->BGR) happens here since
    # Gradio's numpy Image output is RGB but OpenCV heuristics expect BGR.
    image_note_html = ""
    if leaf_image is not None:
        try:
            import cv2
            image_bgr = cv2.cvtColor(leaf_image, cv2.COLOR_RGB2BGR)
        except Exception:
            image_bgr = None
        augmented_question, signals = build_augmented_question(question, image_bgr)
        question = augmented_question
        if signals is not None:
            if signals.analyzed:
                image_note_html = (
                    "<div class='rk-med-section' style='border-left-color:#a855f7; margin-top:10px;'>"
                    "<div class='rk-sec-title'>📷 Photo analysis</div>"
                    f"<div class='rk-sec-content'>{signals.to_symptom_text()}</div></div>"
                )
            elif signals.error:
                # Honest degrade — image didn't yield usable signals, but we
                # still proceed with whatever text the farmer typed.
                image_note_html = (
                    "<div class='rk-med-section' style='border-left-color:#f59e0b; margin-top:10px;'>"
                    "<div class='rk-sec-title'>📷 Photo analysis</div>"
                    f"<div class='rk-sec-content'>{signals.error}</div></div>"
                )

    if not question:
        warning = f"<div class='rk-card' style='text-align:center; color:#f59e0b;'>{t['warn_empty']}</div>"
        return warning, history, scan_times, last_request_time

    if len(question) > MAX_QUESTION_CHARS:
        question = question[:MAX_QUESTION_CHARS]

    if not client.health_check():
        error_html = (
            "<div class='rk-card' style='border: 1px solid #ef4444;'>"
            f"<h3 style='color:#ef4444; margin-top:0;'>{t['err_server_title']}</h3>"
            f"<p>{t['err_server_body']}</p></div>"
        )
        return error_html, history, scan_times, last_request_time

    start = time.time()
    result: InferenceResult = client.infer(question=question, crop=crop, question_type=question_type)
    elapsed = time.time() - start
    new_last_request_time = time.time()
    response_html = _format_response(result, elapsed)

    if image_note_html:
        # Append the photo-analysis note after the model's own response so
        # the farmer can see exactly what the image contributed, distinct
        # from what the LLM reasoned over.
        response_html = response_html.replace("</div>\n", "</div>" + image_note_html + "\n", 1) \
            if "</div>\n" in response_html else response_html + image_note_html

    if not save_history:
        # Diagnosis still ran, but the person turned off session history —
        # respect that immediately rather than saving then hiding it.
        return response_html, history, scan_times, new_last_request_time

    now_dt = datetime.now()
    entry = {
        "question": question, "crop": crop, "question_type": question_type,
        "timestamp": now_dt.strftime("%H:%M:%S"),
    }
    new_history = history + [entry]
    new_scan_times = scan_times + [now_dt]
    return response_html, new_history, new_scan_times, new_last_request_time


def _format_response(result: InferenceResult, elapsed: float) -> str:
    if not result.success:
        return (
            "<div class='rk-card'>"
            "<h3 style='color:#ef4444; margin-top:0;'>⚠️ Diagnosis Failed</h3>"
            f"<p>{result.error or 'Unknown error.'}</p><pre>{result.raw_text}</pre>"
            "</div>"
        )

    confidence_pct = int(result.structured_confidence * 100)
    if result.structured_confidence < 0.5:
        return (
            "<div class='rk-card'>"
            f"<div style='margin-bottom:12px;'><span class='rk-pill rk-pill-time'>⏱ {elapsed:.1f}s</span></div>"
            f"<div class='rk-sec-content'>{result.raw_text}</div>"
            "<p style='color:#64748b; font-size:0.8em; font-style:italic; margin-top:16px;'>Response shown as-is — structure matching failed.</p>"
            "</div>"
        )

    disease_name = result.disease if result.disease else "Diagnosis Complete"
    parts = [
        "<div class='rk-card'>",
        "<div class='rk-med-header'>",
        f"<div class='rk-med-disease'>🦠 {disease_name}</div>",
        "<div style='display:flex; gap:8px;'>",
        f"<span class='rk-pill rk-pill-conf'>{confidence_pct}% Confidence</span>",
        f"<span class='rk-pill rk-pill-time'>⏱ {elapsed:.1f}s</span>",
        "</div></div>",
        f"<div class='rk-confidence-row'><span>Confidence</span><b>{confidence_pct}%</b></div>",
        f"<div class='rk-meter'><i style='width:{confidence_pct}%;'></i></div>",
    ]

    if result.symptoms:
        parts.append(
            "<div class='rk-med-section rk-sec-symptoms'><div class='rk-sec-title'>🩺 Symptoms</div>"
            f"<div class='rk-sec-content'>{result.symptoms}</div></div>"
        )
    if result.treatment:
        parts.append(
            "<div class='rk-med-section rk-sec-treatment'><div class='rk-sec-title'>🛠️ Treatment Plan</div>"
            f"<div class='rk-sec-content'>{result.treatment}</div></div>"
        )
    if result.prevention:
        parts.append(
            "<div class='rk-med-section rk-sec-prevention'><div class='rk-sec-title'>🛡️ Prevention Guide</div>"
            f"<div class='rk-sec-content'>{result.prevention}</div></div>"
        )

    parts.append(
        "<div style='text-align:center; color:#64748b; font-size:0.8em; margin-top:20px; font-style:italic;'>"
        "Grounded in GhanaAgricVQA dataset — verify with a local extension officer.</div></div>"
    )
    return "".join(parts)


def clear_all(lang: str):
    t = TEXTS[lang]
    return "", empty_result_html(t), None


# ---------------------------------------------------------------------------
# UI Layout
# ---------------------------------------------------------------------------
# FIXED
with gr.Blocks(title="RK AgriDig", theme=THEME, css=CUSTOM_CSS) as demo:
    # Per-session state — NOT module globals, so one visitor's scans never
    # leak into another visitor's browser tab (fapp.py used module-level
    # lists for this, which is a real bug under concurrent users).
    history_state = gr.State([])
    scan_times_state = gr.State([])
    last_request_state = gr.State(0.0)
    lang_state = gr.State("en")
    cooldown_state = gr.State(DEFAULT_COOLDOWN_SECONDS)
    save_history_state = gr.State(True)

    with gr.Row(elem_classes="rk-app"):
        # ------------------------------------------------------------ SIDEBAR
        with gr.Column(elem_classes="rk-sidebar", min_width=255):
            gr.HTML(
                '<div class="rk-brand"><div class="rk-logo">✦</div>'
                '<div><b>RK AgriDig</b><span>Offline Crop Intelligence</span></div></div>'
            )

            nav_buttons = {}
            for v in VIEWS:
                nav_buttons[v] = gr.Button(
                    f"{NAV_ICONS[v]}  {TEXTS['en']['nav'][v]}",
                    elem_id=f"nav-{v}",
                    elem_classes=("rk-nav-btn rk-nav-active" if v == "dashboard" else "rk-nav-btn"),
                )

            side_bottom = gr.HTML(
                '<div class="rk-side-bottom"><strong>Designed for the field.</strong>'
                '<p>Local-first diagnosis means farmers can keep working even when connectivity is unavailable.</p>'
                '<div class="rk-status"><span class="rk-dot"></span> AI model ready offline</div></div>'
            )

        # --------------------------------------------------------------- MAIN
        with gr.Column(elem_classes="rk-main"):
            with gr.Row(elem_classes="rk-topbar"):
                crumb = gr.HTML('<div class="rk-crumb">RK AgriDig / <b>Dashboard</b></div>')
                with gr.Row(elem_classes="rk-topbar-right"):
                    with gr.Row(elem_classes="rk-lang-toggle"):
                        btn_lang_en = gr.Button("English", elem_classes="rk-lang-btn-active", size="sm")
                        btn_lang_tw = gr.Button("Twi", size="sm")
                    gr.HTML('<div class="rk-avatar">AD</div>')

            # ---- Dashboard view ----
            with gr.Column(visible=True) as page_dashboard:
                dashboard_view = gr.HTML(dashboard_html([], TEXTS["en"]))
                with gr.Row():
                    dash_scan_btn = gr.Button("＋ Start New Scan", elem_classes="rk-primary-btn")
                    dash_history_btn = gr.Button("View History", elem_classes="rk-secondary-btn")
                stats_view = gr.HTML(stats_row_html([], TEXTS["en"]))
                with gr.Row():
                    with gr.Column(scale=7, elem_classes="rk-card"):
                        recent_scans_view = gr.HTML(recent_scans_card_html([], TEXTS["en"]))
                    with gr.Column(scale=5, elem_classes="rk-card"):
                        weekly_chart_view = gr.HTML(render_weekly_chart([], TEXTS["en"]))

            # ---- Scan Leaf view (wired to real inference) ----
            with gr.Column(visible=False) as page_scan:
                scan_title_html = gr.HTML(
                    f'<div class="rk-page-title">{TEXTS["en"]["scan_title"]}</div>'
                    f'<div class="rk-page-sub">{TEXTS["en"]["scan_sub"]}</div>'
                )
                with gr.Row():
                    with gr.Column(scale=5, elem_classes="rk-card"):
                        crop_picker_label = gr.HTML(
                            f"<span class='rk-crop-picker-label'>{TEXTS['en']['select_crop']}</span>"
                        )
                        # Vector crop icons rendered via gr.HTML (which renders raw
                        # markup, unlike gr.Button's text-escaped label) so the SVGs
                        # actually paint instead of printing as literal <svg> text.
                        # Each icon's onclick calls a hidden gr.Button's DOM element
                        # to fire the real Gradio click event — this indirection is
                        # necessary because gr.HTML has no server-side .click() event
                        # of its own; a hidden same-purpose Button gives us one.
                        crop_hidden_buttons = {}

                        def _crop_grid_html(selected: str) -> str:
                            cards = []
                            for crop_name in CROPS:
                                active = " rk-crop-card-active" if crop_name == selected else ""
                                cards.append(
                                    f"<div class='rk-crop-card{active}' onclick=\"document.getElementById('rk-crop-btn-{crop_name}').click()\">"
                                    f"{CROP_SVG_ICONS[crop_name]}<span>{crop_name}</span></div>"
                                )
                            return f"<div class='rk-crop-grid'>{''.join(cards)}</div>"

                        crop_grid_display = gr.HTML(_crop_grid_html("Maize"))

                        with gr.Row(visible=False):
                            for crop_name in CROPS:
                                crop_hidden_buttons[crop_name] = gr.Button(
                                    crop_name, elem_id=f"rk-crop-btn-{crop_name}"
                                )

                        # Hidden dropdown remains the single source of truth for
                        # crop selection — every existing event binding below
                        # (diagnose, switch_lang, etc.) reads this value unchanged.
                        # The icon cards above just write into it on click.
                        crop_dropdown = gr.Dropdown(
                            choices=CROPS, value="Maize", label=TEXTS["en"]["select_crop"], visible=False
                        )
                        question_type_radio = gr.Radio(
                            choices=QUESTION_TYPES, value="Identification", label=TEXTS["en"]["question_type"]
                        )
                        question_input = gr.Textbox(
                            label=TEXTS["en"]["symptom_lbl"],
                            placeholder=TEXTS["en"]["symptom_ph"],
                            lines=5, max_lines=8,
                        )
                        # Optional leaf photo — analyzed by classical CV heuristics
                        # (src/image_symptom_extractor.py), NOT by Phi-3-mini-4k
                        # itself, since that model has no vision tower. The photo's
                        # extracted symptom signals get appended to whatever text
                        # the farmer typed, and the existing text-only inference
                        # pipeline handles the combined string unchanged.
                        image_input = gr.Image(
                            label=TEXTS["en"]["image_lbl"], type="numpy", sources=["upload", "webcam"], height=180,
                        )
                        image_signals_display = gr.HTML(value="", visible=False)
                        with gr.Row():
                            submit_btn = gr.Button(TEXTS["en"]["btn_submit"], elem_classes="rk-primary-btn")
                            clear_btn = gr.Button(TEXTS["en"]["btn_clear"], elem_classes="rk-secondary-btn")

                    with gr.Column(scale=5, elem_classes="rk-card"):
                        ai_diag_head = gr.HTML(
                            f'<div class="rk-card-head"><h2>{TEXTS["en"]["ai_diag"]}</h2>'
                            f'<span class="rk-badge">{TEXTS["en"]["local_model"]}</span></div>'
                        )
                        response_output = gr.HTML(value=empty_result_html(TEXTS["en"]))
                        with gr.Row():
                            voice_btn = gr.Button(TEXTS["en"]["btn_voice"], elem_classes="rk-secondary-btn", size="sm")
                        voice_status_html = gr.HTML(value="", visible=False)
                        voice_audio_output = gr.Audio(label="", type="filepath", visible=False, autoplay=True)

            # ---- Treatments view ----
            with gr.Column(visible=False) as page_treatments:
                treat_title_html = gr.HTML(
                    f'<div class="rk-page-title">{TEXTS["en"]["treat_title"]}</div>'
                    f'<div class="rk-page-sub">{TEXTS["en"]["treat_sub"]}</div>'
                )
                gr.HTML(treatments_table_html())

            # ---- Knowledge Base view (guide buttons actually work) ----
            with gr.Column(visible=False) as page_knowledge:
                know_title_html = gr.HTML(
                    f'<div class="rk-page-title">{TEXTS["en"]["know_title"]}</div>'
                    f'<div class="rk-page-sub">{TEXTS["en"]["know_sub"]}</div>'
                )
                with gr.Row():
                    with gr.Column(elem_classes="rk-article"):
                        gr.HTML('<div class="rk-articon">🌽</div><h3>Maize diseases</h3>'
                                '<p>Recognize Northern Leaf Blight and Gray Leaf Spot, and know what to inspect next.</p>')
                        btn_guide_maize = gr.Button(TEXTS["en"]["open_guide"], elem_classes="rk-secondary-btn")
                    with gr.Column(elem_classes="rk-article"):
                        gr.HTML('<div class="rk-articon">🌶️</div><h3>Pepper diseases</h3>'
                                '<p>Compare bacterial leaf spot and anthracnose before deciding on treatment.</p>')
                        btn_guide_pepper = gr.Button(TEXTS["en"]["open_guide"], elem_classes="rk-secondary-btn")
                    with gr.Column(elem_classes="rk-article"):
                        gr.HTML('<div class="rk-articon">🍅</div><h3>Tomato diseases</h3>'
                                '<p>Distinguish early blight from late blight and other common leaf conditions.</p>')
                        btn_guide_tomato = gr.Button(TEXTS["en"]["open_guide"], elem_classes="rk-secondary-btn")
                guide_display = gr.HTML("")

            # ---- Scan History view ----
            with gr.Column(visible=False) as page_history:
                hist_title_html = gr.HTML(
                    f'<div class="rk-page-title">{TEXTS["en"]["hist_title"]}</div>'
                    f'<div class="rk-page-sub">{TEXTS["en"]["hist_sub"]}</div>'
                )
                history_table_view = gr.HTML(history_table_html([], TEXTS["en"]))

            # ---- Settings view ----
            with gr.Column(visible=False) as page_settings:
                sett_title_html = gr.HTML(
                    f'<div class="rk-page-title">{TEXTS["en"]["sett_title"]}</div>'
                    f'<div class="rk-page-sub">{TEXTS["en"]["sett_sub"]}</div>'
                )
                with gr.Column(elem_classes="rk-card", elem_id="settings-card"):
                    # Offline-first mode — plain text, not a toggle: there is no
                    # cloud-mode code path in this app, so nothing here could
                    # actually be switched off.
                    offline_row_html = gr.HTML(settings_offline_row_html(TEXTS["en"]))

                    with gr.Row(elem_classes="rk-setting"):
                        with gr.Column(scale=4):
                            save_hist_label = gr.HTML(
                                f"<h3>{TEXTS['en']['sett_save_h']}</h3><p>{TEXTS['en']['sett_save_p']}</p>"
                            )
                        with gr.Column(scale=1, min_width=90):
                            save_history_btn = gr.Button("On", elem_classes="rk-toggle-btn rk-toggle-on")

                    with gr.Row(elem_classes="rk-setting"):
                        with gr.Column(scale=4):
                            cooldown_label = gr.HTML(
                                f"<h3>{TEXTS['en']['sett_cooldown_h']}</h3><p>{TEXTS['en']['sett_cooldown_p']}</p>"
                            )
                        with gr.Column(scale=1, min_width=110):
                            cooldown_dropdown = gr.Dropdown(
                                choices=[f"{s}s" for s in COOLDOWN_OPTIONS],
                                value=f"{DEFAULT_COOLDOWN_SECONDS}s",
                                show_label=False, container=False,
                            )

                    lang_row_html = gr.HTML(
                        f"<div class='rk-setting' style='border-bottom:none;'><div><h3>{TEXTS['en']['sett_lang_h']}</h3>"
                        f"<p>{TEXTS['en']['sett_lang_p']}</p></div>"
                        f"<span class='rk-muted'>English / Twi</span></div>"
                    )

            footer_html = gr.HTML(
                '<div style="text-align:center; color:var(--muted); font-size:.78em; padding:22px 0 6px;'
                ' border-top:1px solid var(--line); margin-top:28px; line-height:1.7;">'
                "RK AgriDig &middot; Offline Crop Diagnosis AI"
                "</div>"
            )

    page_columns = {
        "dashboard": page_dashboard, "scan": page_scan, "treatments": page_treatments,
        "knowledge": page_knowledge, "history": page_history, "settings": page_settings,
    }
    page_outputs = [page_columns[v] for v in VIEWS]

    nav_active_js = """
    () => {
        document.querySelectorAll('.rk-nav-btn').forEach(b => b.classList.remove('rk-nav-active'));
        const clicked = document.activeElement;
        if (clicked && clicked.classList.contains('rk-nav-btn')) clicked.classList.add('rk-nav-active');
    }
    """

    # --- Navigation: every sidebar item + both dashboard shortcut buttons ---
    def make_nav_handler(target_view):
        def _handler(history, scan_times, lang):
            t = TEXTS[lang]
            visibility = [gr.update(visible=(v == target_view)) for v in VIEWS]
            crumb_html = f'<div class="rk-crumb">RK AgriDig / <b>{t["crumb"][target_view]}</b></div>'
            dash = dashboard_html(history, t) if target_view == "dashboard" else gr.update()
            stats = stats_row_html(history, t) if target_view == "dashboard" else gr.update()
            recent = recent_scans_card_html(history, t) if target_view == "dashboard" else gr.update()
            chart = render_weekly_chart(scan_times, t) if target_view == "dashboard" else gr.update()
            hist = history_table_html(history, t) if target_view == "history" else gr.update()
            return [crumb_html, dash, stats, recent, chart, hist] + visibility
        return _handler

    nav_outputs = [crumb, dashboard_view, stats_view, recent_scans_view, weekly_chart_view, history_table_view] + page_outputs

    for v in VIEWS:
        nav_buttons[v].click(
            fn=make_nav_handler(v),
            inputs=[history_state, scan_times_state, lang_state],
            outputs=nav_outputs,
            js=nav_active_js,
        )

    dash_scan_btn_js = nav_active_js.replace("document.activeElement", "document.getElementById('nav-scan')")
    dash_scan_btn.click(
        fn=make_nav_handler("scan"),
        inputs=[history_state, scan_times_state, lang_state],
        outputs=nav_outputs,
        js=dash_scan_btn_js,
    )
    dash_history_btn_js = nav_active_js.replace("document.activeElement", "document.getElementById('nav-history')")
    dash_history_btn.click(
        fn=make_nav_handler("history"),
        inputs=[history_state, scan_times_state, lang_state],
        outputs=nav_outputs,
        js=dash_history_btn_js,
    )

    # --- Knowledge Base guide buttons ---
    def make_guide_handler(crop_key):
        def _handler(lang):
            return guide_html(crop_key, TEXTS[lang])
        return _handler

    btn_guide_maize.click(fn=make_guide_handler("maize"), inputs=[lang_state], outputs=[guide_display])
    btn_guide_pepper.click(fn=make_guide_handler("pepper"), inputs=[lang_state], outputs=[guide_display])
    btn_guide_tomato.click(fn=make_guide_handler("tomato"), inputs=[lang_state], outputs=[guide_display])

    # --- Crop icon picker: hidden button click -> updates the real
    # crop_dropdown value (source of truth, unchanged for every other
    # handler in this file) and re-renders the icon grid's active state. ---
    def make_crop_icon_handler(selected_crop: str):
        def _handler():
            return selected_crop, _crop_grid_html(selected_crop)
        return _handler

    for crop_name in CROPS:
        crop_hidden_buttons[crop_name].click(
            fn=make_crop_icon_handler(crop_name),
            inputs=[],
            outputs=[crop_dropdown, crop_grid_display],
        )

    # --- Diagnosis submit chain (Enter key + button both wired) ---
    def _submit_chain(question, crop, qtype, history, scan_times, last_req, lang, cooldown, save_hist, leaf_image):
        resp, new_hist, new_scan_times, new_last_req = diagnose(
            question, crop, qtype, history, scan_times, last_req, lang, cooldown, save_hist, leaf_image
        )
        return resp, new_hist, new_scan_times, new_last_req

    submit_btn.click(
        fn=lambda lang: gr.update(interactive=False, value=TEXTS[lang]["btn_analyzing"]),
        inputs=[lang_state], outputs=[submit_btn],
    ).then(
        fn=_submit_chain,
        inputs=[question_input, crop_dropdown, question_type_radio, history_state, scan_times_state,
                last_request_state, lang_state, cooldown_state, save_history_state, image_input],
        outputs=[response_output, history_state, scan_times_state, last_request_state],
    ).then(
        fn=lambda lang: gr.update(interactive=True, value=TEXTS[lang]["btn_submit"]),
        inputs=[lang_state], outputs=[submit_btn],
    )

    question_input.submit(
        fn=lambda lang: gr.update(interactive=False, value=TEXTS[lang]["btn_analyzing"]),
        inputs=[lang_state], outputs=[submit_btn],
    ).then(
        fn=_submit_chain,
        inputs=[question_input, crop_dropdown, question_type_radio, history_state, scan_times_state,
                last_request_state, lang_state, cooldown_state, save_history_state, image_input],
        outputs=[response_output, history_state, scan_times_state, last_request_state],
    ).then(
        fn=lambda lang: gr.update(interactive=True, value=TEXTS[lang]["btn_submit"]),
        inputs=[lang_state], outputs=[submit_btn],
    )

    clear_btn.click(
        fn=clear_all, inputs=[lang_state], outputs=[question_input, response_output, image_input],
    )

    # --- Voice playback: reads the current diagnosis aloud (English only).
    # Twi is intentionally unsupported — see src/voice_output.py docstring
    # for why an honest refusal beats a mispronounced attempt. ---
    def _voice_handler(response_html: str, lang: str):
        t = TEXTS[lang]
        result = speak_diagnosis(response_html, lang)
        if result.success:
            return (
                gr.update(value=result.audio_path, visible=True),
                gr.update(value="", visible=False),
            )
        if result.reason == "unsupported_language":
            msg = t.get("voice_unsupported_tw", "Voice narration isn't available in this language yet.")
        elif result.reason == "empty_text":
            msg = ""  # nothing to read yet — no diagnosis has been generated
        else:
            msg = "Voice narration couldn't be generated right now."
        return (
            gr.update(value=None, visible=False),
            gr.update(value=f"<div class='rk-muted' style='font-size:0.85em; margin-top:4px;'>{msg}</div>" if msg else "", visible=bool(msg)),
        )

    voice_btn.click(
        fn=_voice_handler,
        inputs=[response_output, lang_state],
        outputs=[voice_audio_output, voice_status_html],
    )

    # --- Settings: Save scan history On/Off toggle ---
    def toggle_save_history(current):
        new_val = not current
        label = "On" if new_val else "Off"
        css_class = "rk-toggle-btn rk-toggle-on" if new_val else "rk-toggle-btn rk-toggle-off"
        return new_val, gr.update(value=label, elem_classes=css_class)

    save_history_btn.click(
        fn=toggle_save_history,
        inputs=[save_history_state],
        outputs=[save_history_state, save_history_btn],
    )

    # --- Settings: Thermal cooldown seconds dropdown ---
    def set_cooldown(choice, history, lang):
        seconds = int(choice.rstrip("s"))
        t = TEXTS[lang]
        return seconds, stats_row_html(history, t, seconds)

    cooldown_dropdown.change(
        fn=set_cooldown,
        inputs=[cooldown_dropdown, history_state, lang_state],
        outputs=[cooldown_state, stats_view],
    )


    # --- Language toggle: re-labels every visible string, not just a few ---
    def switch_lang(lang, history, scan_times, cooldown):
        t = TEXTS[lang]
        return (
            lang,
            gr.update(elem_classes="rk-lang-btn-active" if lang == "en" else ""),
            gr.update(elem_classes="rk-lang-btn-active" if lang == "tw" else ""),
            # sidebar nav labels
            gr.update(value=f"{NAV_ICONS['dashboard']}  {t['nav']['dashboard']}"),
            gr.update(value=f"{NAV_ICONS['scan']}  {t['nav']['scan']}"),
            gr.update(value=f"{NAV_ICONS['treatments']}  {t['nav']['treatments']}"),
            gr.update(value=f"{NAV_ICONS['knowledge']}  {t['nav']['knowledge']}"),
            gr.update(value=f"{NAV_ICONS['history']}  {t['nav']['history']}"),
            gr.update(value=f"{NAV_ICONS['settings']}  {t['nav']['settings']}"),
            # dashboard
            dashboard_html(history, t),
            stats_row_html(history, t, cooldown),
            recent_scans_card_html(history, t),
            render_weekly_chart(scan_times, t),
            gr.update(value=t["btn_scan"]),
            gr.update(value=t["btn_history"]),
            # scan page
            f'<div class="rk-page-title">{t["scan_title"]}</div><div class="rk-page-sub">{t["scan_sub"]}</div>',
            gr.update(label=t["select_crop"]),
            gr.update(label=t["question_type"], choices=[t["q_id"], t["q_treat"], t["q_prev"]], value=t["q_id"]),
            gr.update(label=t["symptom_lbl"], placeholder=t["symptom_ph"]),
            gr.update(value=t["btn_submit"]),
            gr.update(value=t["btn_clear"]),
            f'<div class="rk-card-head"><h2>{t["ai_diag"]}</h2><span class="rk-badge">{t["local_model"]}</span></div>',
            # treatments / knowledge / history / settings headers
            f'<div class="rk-page-title">{t["treat_title"]}</div><div class="rk-page-sub">{t["treat_sub"]}</div>',
            f'<div class="rk-page-title">{t["know_title"]}</div><div class="rk-page-sub">{t["know_sub"]}</div>',
            gr.update(value=t["open_guide"]),
            gr.update(value=t["open_guide"]),
            gr.update(value=t["open_guide"]),
            f'<div class="rk-page-title">{t["hist_title"]}</div><div class="rk-page-sub">{t["hist_sub"]}</div>',
            history_table_html(history, t),
            f'<div class="rk-page-title">{t["sett_title"]}</div><div class="rk-page-sub">{t["sett_sub"]}</div>',
            settings_offline_row_html(t),
            f"<h3>{t['sett_save_h']}</h3><p>{t['sett_save_p']}</p>",
            f"<h3>{t['sett_cooldown_h']}</h3><p>{t['sett_cooldown_p']}</p>",
            f"<div class='rk-setting' style='border-bottom:none;'><div><h3>{t['sett_lang_h']}</h3>"
            f"<p>{t['sett_lang_p']}</p></div><span class='rk-muted'>English / Twi</span></div>",
            (
                '<div class="rk-side-bottom"><strong>Designed for the field.</strong>'
                '<p>Local-first diagnosis means farmers can keep working even when connectivity is unavailable.</p>'
                '<div class="rk-status"><span class="rk-dot"></span> AI model ready offline</div></div>'
                if lang == "en" else
                '<div class="rk-side-bottom"><strong>Wɔyɛ ma mfuo mu adwuma.</strong>'
                '<p>Ɔman mu nhyehyɛe a ɛdi kan ma akuafo tumi kɔ so yɛ adwuma bere a intanet nni hɔ.</p>'
                '<div class="rk-status"><span class="rk-dot"></span> AI model ayɛ krado ɛnhia intanet</div></div>'
            ),
        )

    lang_outputs = [
        lang_state, btn_lang_en, btn_lang_tw,
        nav_buttons["dashboard"], nav_buttons["scan"], nav_buttons["treatments"],
        nav_buttons["knowledge"], nav_buttons["history"], nav_buttons["settings"],
        dashboard_view, stats_view, recent_scans_view, weekly_chart_view,
        dash_scan_btn, dash_history_btn,
        scan_title_html, crop_dropdown, question_type_radio, question_input,
        submit_btn, clear_btn, ai_diag_head,
        treat_title_html, know_title_html,
        btn_guide_maize, btn_guide_pepper, btn_guide_tomato,
        hist_title_html, history_table_view,
        sett_title_html, offline_row_html, save_hist_label, cooldown_label, lang_row_html,
        side_bottom,
    ]

    btn_lang_en.click(
        fn=lambda h, s, c: switch_lang("en", h, s, c),
        inputs=[history_state, scan_times_state, cooldown_state],
        outputs=lang_outputs,
    )
    btn_lang_tw.click(
        fn=lambda h, s, c: switch_lang("tw", h, s, c),
        inputs=[history_state, scan_times_state, cooldown_state],
        outputs=lang_outputs,
    )

if __name__ == "__main__":
    print("Checking Ollama server before launch...")
    if not client.health_check():
        print(
            "⚠️  Warning: Ollama server not reachable at http://localhost:11434\n"
            "   The UI will still launch, but diagnosis requests will fail until "
            "you run `ollama serve`."
        )
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True)