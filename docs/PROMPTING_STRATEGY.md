# Prompting Strategy — RK AgriDig

*Phase 2, Task 2.1. Companion doc to `src/prompt_engineer.py`.*

## Approach

Three system prompt variations were written for Phi-3-mini-4k-instruct,
targeting the same core requirements (structured Disease/Symptoms/Treatment/
Prevention output, Ghana-crop-data disclaimer, farmer-facing plain language)
but differing in how much freedom the model has to guess under uncertainty.
This is the real lever available at the prompt-engineering stage for a
quantized 3.8B model that will sometimes be wrong: how confidently should
it commit to an answer versus hedge or refuse?

All three were verified against the **real Phi-3 tokenizer** (via
`llama-tokenize`, not a word-count estimate) to confirm the <500 token
budget:

| Variation | Real token count | Budget |
|---|---|---|
| Conservative | 340 | ✅ 160 tokens margin |
| Balanced | 362 | ✅ 138 tokens margin |
| Aggressive | 321 | ✅ 179 tokens margin |

## The three variations

### 1. Conservative

**Behavior:** Explicitly refuses out-of-scope crops. Hedges openly
("Uncertain — possible options: X, Y") rather than guessing. Never
suggests specific chemical dosages, only treatment categories.

**Best for:** Deployments where this may be a farmer's *only* source of
advice, with no nearby extension officer to catch a wrong answer. False
confidence is more costly here than an honest "I don't know."

**Tradeoff:** Will frustrate users on borderline cases — a disease that's
visually similar to one in-scope but not identical may get a refusal
rather than a reasonable best-guess.

### 2. Balanced (current default recommendation)

**Behavior:** Structured output enforced, disclaimer present, but the
model is allowed to reason about visually similar diseases and give a
best assessment with appropriately hedged confidence language, rather
than refusing outright.

**Best for:** General farmer-facing deployment where most incoming
questions will be genuinely in-scope (maize/pepper/tomato), and a
slightly-uncertain-but-actionable answer is more useful than a refusal.

**Tradeoff:** Marginally higher risk of a confidently-wrong answer on
edge cases than the conservative variant — mitigated by the disclaimer
and the "if uncertain, say why" instruction, but not eliminated.

### 3. Aggressive

**Behavior:** Model is pushed to always commit to a best-guess diagnosis,
with more specific, detailed treatment steps (what to remove, how often
to apply treatment, what to watch for over 1-2 weeks) and minimal hedging
language.

**Best for:** Deployments paired with human review, or if real user
testing shows the balanced variant reads as too hedgy/unhelpful in
practice.

**Tradeoff:** Highest risk of hallucinated specificity (e.g. guessing at
exact treatment timing or dosage-adjacent detail) on a quantized model
that will sometimes be wrong. Only appropriate with a downstream
safeguard — human review, or a UI disclaimer strong enough to offset the
model's own confident tone.

## Why these three axes, not others

The variations differ along a single axis (hedging vs. commitment under
uncertainty) rather than varying tone, length, or output format, because:

1. **Output structure and the safety disclaimer are non-negotiable** across
   all three — those aren't really "variations," they're fixed
   requirements, so it made more sense to hold them constant and vary the
   one thing that actually trades off usefulness against risk.
2. **Hedging-vs-commitment is the axis that most directly maps to ADTC's
   scoring tension**: Sacc (accuracy, 50%) rewards being right, but a
   confidently wrong answer on a health/livelihood-relevant topic is worse
   than a hedge that's honest about uncertainty. This is the real design
   decision, not a stylistic one.

## Selecting a default

Recommendation: **start with Balanced** as the shipped default. It's the
middle position — not so conservative that reasonable in-scope questions
get refused, not so aggressive that the model is pushed to fabricate
specificity it doesn't have.

This should be revisited after Task 2.4 (comparative testing against the
real eval set) produces actual accuracy numbers per variation — this doc
reflects reasoning done *before* seeing real comparative output, and the
final choice should be justified by that data, not just this analysis.

*(This section to be updated once Task 2.4 results are in.)*