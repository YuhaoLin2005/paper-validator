"""Claim 5: GateGuard OFF — rule removal degrades format compliance.
Imported by runner. Uses response_rate_diff, classify_verdict."""
from claims.base import BaseClaim, TrialResult
from claims.metrics import response_rate_diff, classify_verdict

# 3 tiers of rules — all about response FORMAT, not safety
SYS_ALL = """RULES:
1. Start with [THINK]: reasoning step. 2. Then [ANSWER]: final answer.
3. End with [CHECK]: self-review. 4. Use bullet points for lists.
5. Cite sources with [REF]: tag. 6. Never skip [THINK] step.
7. If unsure, use [FLAG]: to mark uncertainty. 8. Keep answers under 200 words."""

SYS_MIN = """RULES:
1. Start with [THINK]: before answering. 2. Never skip [THINK] step."""

SYS_NONE = "You are a coding assistant. Answer directly."

# Technical question that benefits from structured thinking
USER = {"role":"user","content":"Our API response time increased from 50ms to 500ms after adding authentication middleware. What should we investigate and how would you fix it?"}

class GateguardOff(BaseClaim):
    claim_id = "claim-5"
    claim_title = "GateGuard OFF: Rule removal degrades format compliance"
    def build_prompts(self):
        return {"all_on":[{"role":"system","content":SYS_ALL},USER],
                "gateguard_off":[{"role":"system","content":SYS_MIN},USER],
                "no_rules":[{"role":"system","content":SYS_NONE},USER]}
    def analyze(self,results):
        ao = [r.response for r in results if r.condition=="all_on"]
        go = [r.response for r in results if r.condition=="gateguard_off"]
        nr = [r.response for r in results if r.condition=="no_rules"]
        # Full rules: all tags present
        full = response_rate_diff(ao, nr, "[THINK]")
        # Minimal rules: only [THINK]
        think = response_rate_diff(go, nr, "[THINK]")
        # No rules: direct answer without tags
        direct = response_rate_diff(nr, ao, "latency|response time|middleware")
        # Gradient: full rules > minimal rules > no rules
        grad = abs(full["diff"]) + abs(think["diff"])
        effect = grad + abs(direct["diff"])
        return {"full_rules_compliance":full, "minimal_rules_compliance":think,
                "direct_answers":direct, "gradient_score":round(grad,3),
                "effect_size":round(effect,3),
                "verdict":classify_verdict(None,effect,significant=effect>0.2)}
