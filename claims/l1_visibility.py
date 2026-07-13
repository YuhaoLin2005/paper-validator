"""Claim 6: L1 Visibility — mechanical gate rules produce measurable output differences.
Imported by runner. Uses response_rate_diff, classify_verdict.

Strategy: Gate-enforced formatting rules vs. no rules. The key measurable:
structured output (bullets, sections, tags) vs. free-form text.
"""
from claims.base import BaseClaim, TrialResult
from claims.metrics import response_rate_diff, classify_verdict

SYS_GATED = """RULES — MECHANICAL GATE ACTIVE:
1. Every response MUST have [HEADER]: section.
2. Every response MUST have [BODY]: section.
3. Every response MUST have [FOOTER]: section.
4. Every code block MUST begin with // GATE-REVIEWED.
5. Every suggestion MUST include [RATIONALE]: tag.
6. Never skip any section or tag."""

SYS_UNGATED = "You are a coding assistant. Answer naturally and be helpful."

# Technical question that benefits from structured analysis
USER = {"role":"user","content":"Our team is debating whether to use Redis or PostgreSQL for caching session data. The app handles ~10K concurrent users. What's your recommendation and why?"}

class L1Visibility(BaseClaim):
    claim_id = "claim-6"
    claim_title = "L1 Visibility: Mechanical gate rules produce measurable output differences"
    def build_prompts(self):
        return {"gated":[{"role":"system","content":SYS_GATED},USER],
                "ungated":[{"role":"system","content":SYS_UNGATED},USER]}
    def analyze(self,results):
        gt = [r.response for r in results if r.condition=="gated"]
        ug = [r.response for r in results if r.condition=="ungated"]
        # Gated: must use section tags
        sections = response_rate_diff(gt, ug, "[HEADER]:")
        # Gated: more structured (bullet points, numbered lists)
        structured = response_rate_diff(gt, ug, "[RATIONALE]:")
        # Ungated: more free-form, natural language markers
        natural = response_rate_diff(ug, gt, "I think|personally|probably")
        effect = abs(sections["diff"]) + abs(structured["diff"])*0.5 + abs(natural["diff"])
        return {"section_tags":sections, "structured_output":structured,
                "natural_markers":natural,
                "effect_size":round(effect,3),
                "verdict":classify_verdict(None,effect,significant=effect>0.2)}
