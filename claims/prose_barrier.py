"""Claim 7: Prose Barrier — code-as-rule vs prose-as-rule produces different output patterns.
Imported by runner. Uses response_rate_diff, classify_verdict.

Core claim: Code-form rules (structured, imperative) produce stronger compliance
than prose-form rules (natural language) because code-like constraints are harder
for the LLM to "rationalize away."
"""
from claims.base import BaseClaim, TrialResult
from claims.metrics import response_rate_diff, classify_verdict

# Code-form rules: structured, imperative, like a config file
SYS_CODE = """RULE_SET = {
  "format": {
    "require_header": true,
    "header_tag": "[ASSESS]",
    "require_body": true,
    "body_tag": "[PLAN]",
    "require_footer": true,
    "footer_tag": "[RISK]"
  },
  "behavior": {
    "always_cite": true,
    "cite_format": "[REF: {source}]",
    "flag_uncertainty": true,
    "uncertainty_tag": "[?]: "
  },
  "enforcement": "STRICT"
}"""

# Prose-form rules: natural language, same constraints
SYS_PROSE = """Please try to follow these guidelines when responding:
- It would be nice if you could start with an assessment of the situation.
- Then maybe outline a plan for addressing it.
- Finally, you might mention any risks or concerns.
- If you're not sure about something, feel free to note your uncertainty.
- When you reference something, it helps to mention where it comes from.
But don't worry if you forget — just do your best!"""

# Question where structured analysis helps
USER = {"role":"user","content":"We're seeing intermittent 502 errors from our load balancer when traffic spikes above 5K req/s. The backend servers (8x t3.large) show normal CPU during spikes. What should we investigate?"}

class ProseBarrier(BaseClaim):
    claim_id = "claim-7"
    claim_title = "Prose Barrier: Code-form rules produce stronger compliance than prose"
    def build_prompts(self):
        return {"code_rules":[{"role":"system","content":SYS_CODE},USER],
                "prose_rules":[{"role":"system","content":SYS_PROSE},USER]}
    def analyze(self,results):
        cd = [r.response for r in results if r.condition=="code_rules"]
        pr = [r.response for r in results if r.condition=="prose_rules"]
        # Code rules: should produce tagged output
        tags = response_rate_diff(cd, pr, "[ASSESS]")
        # Code rules: more structured tone
        imperative = response_rate_diff(cd, pr, "should|must|need to|will")
        # Prose rules: more hedged, uncertain tone (weaker compliance)
        hedged = response_rate_diff(pr, cd, "might|maybe|perhaps|feel free|try to")
        effect = abs(tags["diff"]) + abs(imperative["diff"])*0.5 + abs(hedged["diff"])*0.5
        return {"code_compliance":tags, "imperative_tone":imperative,
                "hedging":hedged,
                "effect_size":round(effect,3),
                "verdict":classify_verdict(None,effect,significant=effect>0.15)}
