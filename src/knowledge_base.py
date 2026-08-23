"""
knowledge_base.py — RK AgriDig

Structured offline disease reference for the Knowledge Base tab. Replaces
the earlier "Common Issues: X, Y" + "Farmer Tip: Z" two-sentence summary
per crop with real per-disease entries: symptoms, likely causes, treatment
steps, and prevention steps — the same structure the live diagnosis output
already uses, so the KB and the AI diagnosis read consistently.

Content is grounded in general agricultural-extension knowledge for these
three crops (maize, pepper, tomato) as commonly documented by sources like
CABI Plantwise, FAO plant health guides, and Ghana's own extension
literature; it intentionally stays at the level of widely-corroborated
symptom/treatment patterns rather than citing any single copyrighted source
verbatim. As with the AI diagnosis output, this is a reference aid, not a
substitute for a local agricultural extension officer — especially for
correct fungicide/pesticide product selection and dosing, which vary by
region and are out of scope for this offline tool to specify.

This is deliberately kept separate from TEXTS (the UI-chrome i18n dict) in
app.py: disease facts are content, not interface strings, and mixing the
two made the KB harder to extend (three crops means three sentences,
whereas real entries mean real growth).

Twi translation status: entries below are English-only for now. Translating
disease terminology (lesion, necrosis, chlorosis, etc.) accurately into Twi
needs review with a Twi-speaking agricultural source before it goes in
front of farmers — publishing a rough/uncertain translation of medical-
adjacent content is worse than clearly marking it English-only until that
review happens. The Twi UI shows an honest note instead of translated text.

Team: Aaron Baidoo (RoniKid) & Firdaus Kudus (github.com/KudusFirdaus)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class DiseaseEntry:
    name: str
    icon: str
    symptoms: str
    causes: str
    treatment: List[str]
    prevention: List[str]


@dataclass
class CropGuide:
    crop_key: str
    display_name: str
    icon: str
    intro: str
    diseases: List[DiseaseEntry] = field(default_factory=list)


CROP_GUIDES: dict[str, CropGuide] = {
    "maize": CropGuide(
        crop_key="maize",
        display_name="Maize",
        icon="🌽",
        intro="Maize leaf disease is easiest to catch early — most treatable "
              "issues below start as small lesions before spreading.",
        diseases=[
            DiseaseEntry(
                name="Northern Leaf Blight",
                icon="🟤",
                symptoms="Long, cigar-shaped grayish-green to tan lesions (2.5–15cm) running "
                         "parallel to the leaf veins, usually starting on lower/older leaves first "
                         "and moving upward.",
                causes="Fungus (Exserohilum turcicum). Favored by cool, humid weather, heavy dew, "
                       "and dense planting that traps moisture on leaves.",
                treatment=[
                    "Remove and destroy heavily infected lower leaves early to slow upward spread.",
                    "Apply a fungicide labeled for Northern Leaf Blight if lesions appear before "
                    "tasseling — timing matters more than product choice.",
                    "If already widespread near harvest, focus on next-season prevention instead "
                    "of treating this crop.",
                ],
                prevention=[
                    "Plant resistant/tolerant maize varieties where available.",
                    "Rotate with a non-host crop (e.g. legumes) for at least one season.",
                    "Space plants for airflow and avoid overhead irrigation late in the day.",
                    "Clear or bury old maize residue after harvest — it carries the fungus over.",
                ],
            ),
            DiseaseEntry(
                name="Gray Leaf Spot",
                icon="⬜",
                symptoms="Small, rectangular tan-to-gray lesions bound by leaf veins, giving a "
                         "distinctive 'ruler-drawn' straight-edged look. Lesions merge and can "
                         "cause large sections of leaf to die in severe cases.",
                causes="Fungus (Cercospora zeae-maydis). Spreads fastest in warm, humid conditions "
                       "with extended leaf wetness, and builds up in continuous maize fields.",
                treatment=[
                    "Fungicide application is most effective applied preventively or at first "
                    "sign of lesions, not after heavy spread.",
                    "Remove badly affected leaves where practical on smaller plots.",
                ],
                prevention=[
                    "Rotate away from maize for at least one season in affected fields.",
                    "Choose resistant hybrid varieties if reinfection is a recurring problem.",
                    "Till under or remove maize residue, since the fungus survives on old debris.",
                ],
            ),
        ],
    ),
    "pepper": CropGuide(
        crop_key="pepper",
        display_name="Pepper",
        icon="🌶️",
        intro="Pepper disease often shows on fruit as well as leaves — check both when "
              "comparing the entries below.",
        diseases=[
            DiseaseEntry(
                name="Bacterial Leaf Spot",
                icon="🟢",
                symptoms="Small, dark, water-soaked spots on leaves that turn brown with a "
                         "yellow halo; spots may merge and cause leaves to yellow and drop. Can "
                         "also produce raised, scab-like spots on fruit.",
                causes="Bacteria (Xanthomonas species). Spreads via splashing water, contaminated "
                       "tools, and infected seed — worse in warm, wet, or overhead-irrigated fields.",
                treatment=[
                    "Remove and destroy infected plant debris — bacterial diseases don't respond "
                    "to standard fungicides, so copper-based bactericides are the relevant product "
                    "class if treating.",
                    "Avoid working in the field while leaves are wet, to prevent spreading bacteria "
                    "on hands, tools, and clothing.",
                ],
                prevention=[
                    "Use certified disease-free seed or transplants.",
                    "Irrigate at the base of the plant rather than overhead.",
                    "Rotate with non-solanaceous crops (avoid tomato, eggplant in the same rotation).",
                    "Disinfect tools between plants if disease is present in the field.",
                ],
            ),
            DiseaseEntry(
                name="Anthracnose",
                icon="🍂",
                symptoms="Sunken, circular lesions on ripening fruit, often with concentric rings "
                         "and pink-to-orange spore masses in the center under humid conditions. "
                         "Leaf symptoms are less common than fruit symptoms.",
                causes="Fungus (Colletotrichum species). Thrives in warm, wet weather and spreads "
                       "via splashing rain or irrigation water, especially onto ripening fruit.",
                treatment=[
                    "Remove and destroy infected fruit promptly to reduce spore spread to healthy fruit.",
                    "Apply a fungicide labeled for anthracnose starting at early fruit set in "
                    "fields with a history of this disease.",
                ],
                prevention=[
                    "Stake or trellis plants to keep fruit off the ground and improve airflow.",
                    "Avoid overhead irrigation, especially as fruit begins to ripen.",
                    "Rotate crops and avoid planting into fields with recent anthracnose history.",
                ],
            ),
        ],
    ),
    "tomato": CropGuide(
        crop_key="tomato",
        display_name="Tomato",
        icon="🍅",
        intro="Early and late blight are often confused — the ring pattern and speed of "
              "spread are the clearest way to tell them apart.",
        diseases=[
            DiseaseEntry(
                name="Early Blight",
                icon="🎯",
                symptoms="Dark brown spots with visible concentric rings (a 'target' or "
                         "bullseye pattern), usually starting on older, lower leaves. Yellowing "
                         "often develops around each spot before the leaf drops.",
                causes="Fungus (Alternaria solani). Favored by warm temperatures with alternating "
                       "wet and dry periods, and often worse on stressed or nutrient-deficient plants.",
                treatment=[
                    "Remove and destroy affected lower leaves as soon as spots appear.",
                    "Apply a fungicide labeled for early blight, focusing coverage on lower "
                    "foliage where infection usually starts.",
                    "Improve plant nutrition — stressed plants are more susceptible to worsening spread.",
                ],
                prevention=[
                    "Mulch around the base of plants to stop soil-borne spores splashing onto "
                    "lower leaves during rain.",
                    "Stake or cage plants to keep foliage off the ground.",
                    "Rotate with non-solanaceous crops for at least one season.",
                    "Water at the base rather than overhead, and avoid watering late in the day.",
                ],
            ),
            DiseaseEntry(
                name="Late Blight",
                icon="⚠️",
                symptoms="Large, irregular, dark water-soaked patches that spread rapidly, often "
                         "with a pale green-to-yellow border. White fuzzy fungal growth may appear "
                         "on the underside of leaves in humid conditions. Can destroy a field within "
                         "days under favorable weather — treated as more urgent than early blight.",
                causes="Oomycete/water mold (Phytophthora infestans). Spreads explosively in cool, "
                       "wet weather; the same pathogen responsible for historic potato famines.",
                treatment=[
                    "Act immediately — late blight spreads far faster than early blight. Remove "
                    "and destroy (don't compost) infected plants where the field is not already "
                    "widely affected.",
                    "Apply a fungicide labeled specifically for late blight; general-purpose "
                    "fungicides may not be effective against this pathogen.",
                    "If spread is already severe, focus on protecting unaffected neighboring plants "
                    "rather than saving heavily infected ones.",
                ],
                prevention=[
                    "Avoid overhead irrigation and working in wet fields, since water spreads spores.",
                    "Space and stake plants for airflow to reduce leaf wetness duration.",
                    "Don't plant near old potato or tomato debris, which can carry the pathogen over.",
                    "Monitor weather — cool, wet stretches are the highest-risk period to inspect fields.",
                ],
            ),
        ],
    ),
}


def render_crop_guide_html(crop_key: str) -> str:
    """Render a full structured guide for one crop as HTML matching the
    existing rk-guide-panel / rk-med-section visual language already used
    by the live AI diagnosis output, so the KB doesn't look like a
    different app bolted on."""
    guide = CROP_GUIDES.get(crop_key.lower())
    if guide is None:
        return "<div class='rk-guide-panel'><p>No guide available for this crop yet.</p></div>"

    parts = [
        f"<div class='rk-guide-panel'>",
        f"<h4>{guide.icon} {guide.display_name} Disease Reference</h4>",
        f"<p style='color:var(--muted); margin-bottom:16px;'>{guide.intro}</p>",
    ]

    for d in guide.diseases:
        treatment_items = "".join(f"<li>{step}</li>" for step in d.treatment)
        prevention_items = "".join(f"<li>{step}</li>" for step in d.prevention)
        parts.append(
            f"<div class='rk-med-section' style='border-left-color:#a855f7; margin-bottom:14px;'>"
            f"<div class='rk-sec-title'>{d.icon} {d.name}</div>"
            f"<div class='rk-sec-content'><b>Symptoms:</b> {d.symptoms}</div>"
            f"<div class='rk-sec-content' style='margin-top:8px;'><b>Likely cause:</b> {d.causes}</div>"
            f"</div>"
        )
        parts.append(
            f"<div class='rk-med-section' style='border-left-color:var(--orange); margin-bottom:6px;'>"
            f"<div class='rk-sec-title'>🛠️ Treatment — {d.name}</div>"
            f"<ul class='rk-sec-content' style='margin:0; padding-left:18px;'>{treatment_items}</ul>"
            f"</div>"
        )
        parts.append(
            f"<div class='rk-med-section' style='border-left-color:var(--green); margin-bottom:20px;'>"
            f"<div class='rk-sec-title'>🛡️ Prevention — {d.name}</div>"
            f"<ul class='rk-sec-content' style='margin:0; padding-left:18px;'>{prevention_items}</ul>"
            f"</div>"
        )

    parts.append(
        "<p style='text-align:center; color:#64748b; font-size:0.8em; font-style:italic; margin-top:8px;'>"
        "Reference guide only — confirm treatment choices with a local agricultural extension "
        "officer, especially for product selection and dosing.</p>"
    )
    parts.append("</div>")
    return "".join(parts)