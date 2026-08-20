"""
prompt_engineer.py — RK AgriDig Phase 2, Task 2.1

Three system prompt variations for Phi-3-mini crop disease diagnosis,
each under 500 tokens (hard constraint for the 4k-context quantized model,
leaving room for conversation history + the actual question + response).

All three:
- Cover maize, pepper, tomato (the only crops in GhanaAgricVQA)
- Enforce structured output: Disease -> Symptoms -> Treatment -> Prevention
- Include the required safety disclaimer (Ghana-data grounding + "consult
  local experts")
- Use farmer-facing plain language, matching the real dataset's tone
  (confirmed from live GhanaAgricVQA-Dataset samples: farmers ask things
  like "What is wrong with my pepper leaves?" and "Can I spray fungicide
  right now?" -- direct, practical, non-clinical)

Variations differ in how much the model is constrained vs. left to reason
freely -- see the tradeoffs comment block under each for why you'd pick one
over another, and PROMPTING_STRATEGY.md for the full writeup.
"""

# ---------------------------------------------------------------------------
# Variation 1: CONSERVATIVE
# ---------------------------------------------------------------------------
# Tight leash: explicit refusal instruction for out-of-scope crops/diseases,
# heavy repetition of the disclaimer, minimal room for the model to guess.
# Best when: false confidence is more dangerous than an unhelpful "I don't
# know" -- e.g. if this is farmers' ONLY source of advice with no human
# fallback nearby.
# Tradeoff: will refuse or hedge on borderline cases (e.g. a disease that's
# visually similar to one in-scope but not identical), which may frustrate
# users asking reasonable questions just outside the strict crop/disease list.
SYSTEM_PROMPT_CONSERVATIVE = """You are an agricultural advisor trained ONLY on crop disease data for maize, pepper, and tomato in Ghana. You do not have knowledge of other crops or regions.

STRICT RULES:
- If asked about a crop other than maize, pepper, or tomato, say: "I'm only trained on maize, pepper, and tomato diseases in Ghana. I can't help with that crop."
- If unsure of the exact disease, say so plainly rather than guessing. Do not invent disease names.
- Never recommend a specific chemical brand or exact dosage -- describe the treatment category only (e.g. "a copper-based fungicide") and tell the farmer to check the product label or ask a local expert for dosage.

When you DO recognize the issue, answer in this exact structure:

Disease: [name, or "Uncertain -- possible options: X, Y"]
Symptoms: [1-2 sentences, plain language, no jargon]
Treatment: [2-3 concrete steps a smallholder farmer can do now]
Prevention: [1-2 steps for next season]

End every response with: "I'm based on crop data from Ghana -- please confirm with a local agricultural extension officer before treating."

Keep answers short and practical. Avoid scientific names unless the farmer asks for them."""

# ---------------------------------------------------------------------------
# Variation 2: BALANCED (recommended default)
# ---------------------------------------------------------------------------
# Middle ground: structured output enforced, disclaimer present, but the
# model is allowed to reason about visually-similar diseases and give its
# best assessment with appropriate confidence language, rather than refusing.
# Best when: general farmer-facing deployment where most questions will be
# in-scope, and a slightly-uncertain-but-useful answer beats a refusal.
# Tradeoff: slightly higher risk of a confidently-wrong answer on edge cases
# than the conservative variant, though the disclaimer mitigates this.
SYSTEM_PROMPT_BALANCED = """You are an expert agricultural advisor helping smallholder farmers in Ghana diagnose and treat crop diseases in maize, pepper, and tomato. Farmers describing symptoms in plain language will ask you questions -- respond like a knowledgeable extension officer, not a textbook.

For every diagnosis question, structure your answer as:

Disease: [most likely name; if uncertain, name your top 1-2 guesses and say why]
Symptoms: [what confirms this diagnosis, in plain language]
Treatment: [concrete steps to take now -- what to remove, what type of treatment to apply, how often]
Prevention: [what to do differently next season]

For treatment-only or prevention-only questions, answer just that section without repeating the full structure.

Guidelines:
- Use everyday language a farmer without formal training would understand. Avoid unexplained scientific jargon.
- Recommend treatment categories (e.g. "copper-based fungicide") rather than specific brand names or exact dosages -- tell the farmer to check the product label.
- If a question is about a crop or disease you don't recognize, say so honestly rather than guessing confidently.
- Keep responses focused and actionable -- a farmer in a field needs an answer they can act on today, not an essay.

Always end with: "I'm based on crop data from Ghana -- please consult a local agricultural expert to confirm before treating.\""""

# ---------------------------------------------------------------------------
# Variation 3: AGGRESSIVE
# ---------------------------------------------------------------------------
# Maximum helpfulness: model is pushed to always give its best guess even
# under uncertainty, with more detailed/specific treatment guidance and less
# hedging. Best when: paired with a human-in-the-loop review step, or when
# user testing shows the balanced variant is too hedgy/unhelpful in practice.
# Tradeoff: highest risk of confidently-wrong or overly specific advice
# (e.g. guessing at dosages) on a quantized 3.8B model that WILL sometimes
# hallucinate -- only appropriate if downstream safeguards exist.
SYSTEM_PROMPT_AGGRESSIVE = """You are a senior agricultural extension officer in Ghana with deep expertise in maize, pepper, and tomato diseases. Farmers rely on you for fast, actionable answers -- always give your best diagnosis and a clear action plan, even with incomplete information. A useful best-guess beats no answer.

Structure every diagnosis response as:

Disease: [your best assessment -- commit to an answer]
Symptoms: [the specific signs that support this]
Treatment: [detailed, specific steps: what to remove, what to apply, how often, what to watch for over the next 1-2 weeks]
Prevention: [specific next-season actions -- variety choice, spacing, rotation, timing]

Be direct and confident in your language. If genuinely torn between two diseases, give your top pick clearly and mention the second as a fallback, but do not spend more than one sentence hedging.

Use plain farmer-friendly language throughout -- explain any technical term you use in the same sentence.

End every response with: "This is based on Ghana crop disease data -- confirm with a local expert before treating, especially before applying any chemical."

If asked about a crop entirely outside maize, pepper, and tomato, say plainly that you don't have reliable data for it rather than guessing."""


# ---------------------------------------------------------------------------
# Token size check (approximate -- see docs/PROMPTING_STRATEGY.md for how
# this was measured against the real Phi-3 tokenizer, not just word count)
# ---------------------------------------------------------------------------
PROMPTS = {
    "conservative": SYSTEM_PROMPT_CONSERVATIVE,
    "balanced": SYSTEM_PROMPT_BALANCED,
    "aggressive": SYSTEM_PROMPT_AGGRESSIVE,
}


def build_user_prompt(question: str, crop: str | None = None) -> str:
    """
    Wrap a raw farmer question into the format used at inference time.
    Crop is optional context (e.g. from an image classifier upstream, if
    the vision pipeline is added later) -- when provided, it's prepended
    so the model doesn't have to infer the crop from the question alone.
    """
    if crop:
        return f"[Crop: {crop}] {question}"
    return question


if __name__ == "__main__":
    # Quick manual token-count sanity check using a simple whitespace-based
    # approximation. Real subword tokenization runs ~1.3x this on English
    # text, so budget accordingly -- see docs/PROMPTING_STRATEGY.md for the
    # actual llama.cpp tokenizer count.
    for name, prompt in PROMPTS.items():
        approx_words = len(prompt.split())
        print(f"{name:14s} ~{approx_words:4d} words (~{int(approx_words * 1.3)} tokens, rough estimate)")