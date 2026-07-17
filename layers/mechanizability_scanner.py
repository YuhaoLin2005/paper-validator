"""Mechanizability Scanner — Where Does the Gate Stop Working?
================================================================
Inspired by Max Quimby (DEV.to comment, July 2026):
  "The five-layer classification exists but is still manual.
   A mechanizability-scanner that infers layers from rule structure
   is the next build."

Deterministic classifier — no LLM dependency (Prose Barrier: the scanner
that judges gateability must itself be gateable).

Scores governance rules 0.0–1.0 on mechanizability:
  L1 (mechanical):  0.70–1.00 = deterministic signal (file/mtime/exit/regex)
  L2 (neural):      0.30–0.69 = decision-token detectable, not mechanically checkable
  L3 (semantic):    0.00–0.29 = requires interpretation, no deterministic anchor
"""

from __future__ import annotations

import re, json
from dataclasses import dataclass, field
from pathlib import Path


# ── Signals ──

MECHANICAL_SIGNALS = [
    (r'\b(?:file|path|directory)\s+(?:exists|present|found|created)', 0.25, "file_check"),
    (r'\b(?:modification\s*time|mtime|timestamp|last\s*(?:modified|updated))', 0.25, "mtime_check"),
    (r'\b(?:exit\s*code|return\s*code|status\s*code|exit\s*0)\b', 0.25, "exit_code"),
    (r'\b(?:hook|trigger|callback|listener)\s+(?:wired?|connected?|registered?|active)', 0.25, "hook_wiring"),
    # IGNORECASE: matches [ANSWER], [answer], [Answer] — text is lowered but bracket tags are case-convention
    (r'\[(?:reasoning|verify|think|check|answer|alternatives?|trade.?off|end)\]', 0.30, "structured_markers"),
    # Broader: MUST/NEVER/EVERY + any verb (was: whitelist of 8 verbs — missed "be wrapped", "start with", "acknowledge")
    (r'\b(?:must|never|every)\s+\w+', 0.20, "must_directive"),
    # Code-fence / backtick markers — strong mechanical signal for format rules
    (r'```\w*|`[^`]+`', 0.25, "code_fence"),
    (r'\b(?:yes\s*/\s*no|pass\s*/\s*fail|true\s*/\s*false|A\s*or\s*B)\b', 0.20, "binary_check"),
    (r'\b(?:at\s+least|exactly|no\s+more\s+than|maximum|minimum)\s+\d+', 0.15, "countable"),
    (r'\b(?:pattern|regex|regular\s+expression|match)\b', 0.25, "regex_specified"),
]

SEMANTIC_SIGNALS = [
    (r'\b(?:thorough|comprehensive|detailed|deep|quality|high.?quality)\b', -0.20, "quality_judgment"),
    (r'\b(?:relevant|useful|helpful|meaningful|insightful)\b', -0.20, "relevance_judgment"),
    (r'\b(?:reason\s+(?:about|through|deeply)|think\s+(?:carefully|deeply)|consider\s+(?:carefully|deeply))', -0.25, "reasoning_depth"),
    (r'\b(?:alternative|trade.?off|approach|option|choice|decide)\b', -0.10, "alternatives"),
    (r'\b(?:depends?\s+on|context|situation|circumstance|case.?by.?case)\b', -0.15, "context_dependent"),
    (r'\b(?:should|might|could|consider|try\s+to|attempt\s+to)\b', -0.05, "vague_directive"),
    # Uncertainty language — strong semantic signal (acknowledging limits = interpretation)
    (r'\b(?:uncertain(?:ty)?|ambigu(?:ous|ity)|multiple\s+(?:valid|possible)\s+answers?)\b', -0.15, "uncertainty_language"),
]


@dataclass
class RuleScore:
    rule_text: str
    rule_id: str = ""
    score: float = 0.0
    layer: str = "L3"
    signals_found: list[str] = field(default_factory=list)
    boundary: bool = False


@dataclass
class CoverageReport:
    rules: list[RuleScore]
    total: int
    l1_count: int
    l2_count: int
    l3_count: int
    boundary_rules: list[str]
    avg_mechanizability: float
    p1_1_prediction: str


# ── Core ──

def scan_rule(rule_text: str, rule_id: str = "") -> RuleScore:
    """Score a single governance rule for mechanizability (0.0–1.0)."""
    text_lower = rule_text.lower()
    score = 0.50
    signals: list[str] = []

    for pattern, weight, label in MECHANICAL_SIGNALS:
        if re.search(pattern, text_lower):
            score += weight
            signals.append(f"+{label}({weight:+.2f})")

    for pattern, weight, label in SEMANTIC_SIGNALS:
        if re.search(pattern, text_lower):
            score += weight
            signals.append(f"{label}({weight:+.2f})")

    score = max(0.0, min(1.0, score))
    layer = "L1" if score >= 0.70 else ("L2" if score >= 0.30 else "L3")
    boundary = 0.60 <= score <= 0.75

    return RuleScore(
        rule_text=rule_text, rule_id=rule_id,
        score=score, layer=layer, signals_found=signals, boundary=boundary,
    )


def scan_rule_set(rules: list[dict]) -> CoverageReport:
    """Scan a rule set; return aggregate report + P1-1 residual prediction."""
    scored = [scan_rule(r["text"], r.get("id", "")) for r in rules]
    l1 = [s for s in scored if s.layer == "L1"]
    l2 = [s for s in scored if s.layer == "L2"]
    l3 = [s for s in scored if s.layer == "L3"]
    boundary = [s.rule_id for s in scored if s.boundary]
    avg = sum(s.score for s in scored) / max(1, len(scored))

    n_below = len(l2) + len(l3)
    if n_below == 0:
        pred = "All L1-gatable → 0 residual violations (P1-1 baseline)"
    elif len(l1) == 0:
        pred = "No L1-gatable → 100% semantic violations (P1-1 ceiling)"
    else:
        pct = n_below / len(scored) * 100
        pred = (
            f"{len(l1)}/{len(scored)} rules L1-gatable ({len(l1)/len(scored)*100:.0f}%), "
            f"{n_below} below threshold → residual violations cluster on semantic rules "
            f"(~{pct:.0f}% surface area uncovered)"
        )

    return CoverageReport(
        rules=scored, total=len(scored),
        l1_count=len(l1), l2_count=len(l2), l3_count=len(l3),
        boundary_rules=boundary, avg_mechanizability=round(avg, 3),
        p1_1_prediction=pred,
    )


def find_boundary(rules: list[dict]) -> list[dict]:
    """Identify boundary rules — where 'gate it' becomes 'nudge it'."""
    scored = [scan_rule(r["text"], r.get("id", "")) for r in rules]
    return [
        {
            "id": s.rule_id, "text": s.rule_text[:120],
            "score": round(s.score, 3), "layer": s.layer,
            "signals": s.signals_found,
            "why": (
                "Mechanical signal present but semantic requirement undermines gate"
                if s.score > 0.60 else
                "Semantic requirement with weak mechanical anchor — nudge possible, gate unreliable"
            ),
        }
        for s in scored if s.boundary
    ]


def format_report(report: CoverageReport) -> str:
    lines = [
        f"Mechanizability Coverage Report",
        f"═══════════════════════════",
        f"Total: {report.total}  Avg: {report.avg_mechanizability:.3f}",
        f"L1 (mechanical): {report.l1_count}  L2 (neural): {report.l2_count}  L3 (semantic): {report.l3_count}",
        f"Boundary: {len(report.boundary_rules)} rules",
    ]
    for rid in report.boundary_rules:
        r = next((r for r in report.rules if r.rule_id == rid), None)
        if r:
            lines.append(f"  [{rid}] {r.score:.3f} \"{r.rule_text[:80]}...\"")
    lines.append(f"\nP1-1: {report.p1_1_prediction}")
    return "\n".join(lines)


# ── Built-in rule set (from P1-2 experiment) ──

BUILTIN_RULES = [
    {"id": "delivery_gate", "text": "Before delivering output, check that all required sections are present and marked with [REASONING], [ALTERNATIVES], [ANSWER] brackets."},
    {"id": "fact_check", "text": "You MUST verify at least one claim against a source. State which claim you verified and what source you used."},
    {"id": "self_review", "text": "You must reason through the full loop before acting — consider what might go wrong, then decide."},
    {"id": "alternative_seeking", "text": "You MUST name at least one alternative approach and state why you rejected it."},
    {"id": "trade_off", "text": "You MUST identify at least one explicit trade-off in your decision."},
    {"id": "change_condition", "text": "You MUST state at least one condition under which your answer would change."},
    {"id": "file_output", "text": "Write the result to results/audit_output.json with exit code 0 on success."},
    {"id": "connection_check", "text": "Check that the file exists and its modification timestamp is within the last 5 minutes."},
    {"id": "quality_standard", "text": "Your analysis should be thorough and insightful, considering multiple perspectives carefully."},
    {"id": "context_aware", "text": "Tailor your response to the specific situation — what works in one context may not work in another."},
]


# ── CLI ──

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Mechanizability Scanner")
    sp = p.add_subparsers(dest="cmd")

    s = sp.add_parser("scan", help="Scan single rule")
    s.add_argument("--rule", required=True)
    s.add_argument("--id", default="")

    b = sp.add_parser("batch", help="Scan rule set")
    b.add_argument("--rules", default="")
    b.add_argument("--builtin", action="store_true")

    d = sp.add_parser("boundary", help="Find boundary rules")
    d.add_argument("--rules", default="")
    d.add_argument("--builtin", action="store_true")

    a = p.parse_args()

    if a.cmd == "scan":
        sc = scan_rule(a.rule, a.id)
        print(f"Score: {sc.score:.3f}  Layer: {sc.layer}")
        print(f"Signals: {', '.join(sc.signals_found) or 'none'}")
        print(f"Boundary: {'YES' if sc.boundary else 'no'}")

    elif a.cmd == "batch":
        rules = json.loads(Path(a.rules).read_text(encoding="utf-8")) if a.rules else BUILTIN_RULES
        report = scan_rule_set(rules)
        print(format_report(report))
        print("\nPer-rule:")
        for s in report.rules:
            bar = "█" * int(s.score * 20) + "░" * (20 - int(s.score * 20))
            print(f"  [{s.rule_id}] {bar} {s.score:.3f} {s.layer}")

    elif a.cmd == "boundary":
        rules = json.loads(Path(a.rules).read_text(encoding="utf-8")) if a.rules else BUILTIN_RULES
        for b in find_boundary(rules):
            print(f"\n[{b['id']}] score={b['score']:.3f} ({b['layer']})")
            print(f"  {b['text']}")
            print(f"  {b['why']}")
    else:
        p.print_help()
