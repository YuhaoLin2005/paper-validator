"""Claim 8: Cross-Model — Governance effect on output structure.
Imported by runner. Uses response_rate_diff, classify_verdict.

Tests whether governance rules produce consistent output patterns
across structurally different system prompts (simulating different model personas).
"""
from claims.base import BaseClaim, TrialResult
from claims.metrics import response_rate_diff, classify_verdict

# Governance rules — format enforcement, not safety
SYS_RULES = """RULES (GOVERNANCE ACTIVE):
1. Always write [BEGIN]: before analysis.
2. Always write [END]: after final answer.
3. Always number your recommendations as R1, R2, R3...
4. Always include [CONFIDENCE]: (high/medium/low) after each recommendation."""

SYS_NO_RULES = "You are a coding assistant. Answer questions directly and be helpful."

# Realistic devops question
USER = {"role":"user","content":"We need to migrate our PostgreSQL database from AWS RDS to Azure without downtime. The database is ~500GB with continuous writes. What's our migration strategy?"}

class CrossModel(BaseClaim):
    claim_id = "claim-8"
    claim_title = "Cross-Model: Governance rules produce consistent output patterns"
    def build_prompts(self):
        return {"with_rules":[{"role":"system","content":SYS_RULES},USER],
                "without_rules":[{"role":"system","content":SYS_NO_RULES},USER]}
    def analyze(self,results):
        wr = [r.response for r in results if r.condition=="with_rules"]
        nr = [r.response for r in results if r.condition=="without_rules"]
        # With rules: should use format markers
        markers = response_rate_diff(wr, nr, "[BEGIN]")
        # Without rules: free-form answers
        free = response_rate_diff(nr, wr, "here's|sure|okay|well")
        effect = abs(markers["diff"]) + abs(free["diff"])*0.5
        return {"format_markers":markers, "free_form":free,
                "effect_size":round(effect,3),
                "verdict":classify_verdict(None,effect,significant=effect>0.15)}
