"""P1-2: GateGuard-off Format Effect Re-test
================================================
2×2 factorial: Format (code vs prose) × GateGuard (on vs off)
Tests Mike Czerwinski's question: does format still matter for reasoning
depth when GateGuard is off?

Pre-registered hypothesis: Format effect on reasoning depth is LARGER
under GateGuard-OFF than under GateGuard-ON. (Gate compensates for format
differences when active; format becomes the only signal when gate is off.)

Design: 4 conditions × 30 trials = 120 API calls.
Scoring: deterministic regex-based (no LLM judge per Dipankar's warning).

Dual-pool supervision: Design reviewed by Methodology Expert (pre-registered
hypothesis + fixed classification) and Stats Reviewer (bootstrap CI + Cohen's d).
Digital twin: results registered against self-model's mechanizability gradient.
"""

import json, sys, time, os, re, random
sys.path.insert(0, os.path.dirname(__file__))

from engine.api_client import call_api, extract_text
from claims.metrics import cohens_d


# ── Pre-registered conditions (2×2, defined BEFORE data collection) ──
# "code" = syllogistic/imperative, explicit patterns, MUST statements
# "prose" = narrative/suggestive, implicit expectations, "should" phrasing
# GateGuard ON = mechanical enforcement language + specific pattern requirements
# GateGuard OFF = suggestion-only, no enforcement, no specific patterns

CONDITIONS = {
    "code_gate_on": {
        "format": "code",
        "gate": "on",
        "system": """RULES (ENFORCED — violations will be flagged):
1. Structure: [REASONING] → [ALTERNATIVES] → [ANSWER]
2. You MUST use at least one logical connector: 'because', 'since', 'therefore'
3. You MUST name at least one alternative approach and state why you rejected it
4. You MUST identify at least one explicit trade-off in your decision
5. You MUST state at least one condition under which your answer would change

Non-compliance with any rule will be recorded as a violation.""",
    },
    "code_gate_off": {
        "format": "code",
        "gate": "off",
        "system": """Some things you might find helpful for this question:
- You could structure your response in sections if that helps
- It's often useful to explain why you chose an approach
- You might consider what alternatives exist
- Trade-offs are sometimes interesting to discuss
- Your answer might change under different conditions

No particular format is required — use whatever approach works best.""",
    },
    "prose_gate_on": {
        "format": "prose",
        "gate": "on",
        "system": """When answering this question, you are required to carefully walk through your reasoning before reaching a conclusion. You must explain why you chose your approach, using logical connectors such as "because" or "since" to connect your reasoning to your answer. You must also discuss at least one alternative approach that you considered and explain what trade-off led you to reject it. You must identify at least one explicit trade-off inherent in your chosen approach. Finally, you must state at least one condition under which your recommendation would change — for example, if the team size were different or if certain requirements shifted.

Skipping any of these elements will be recorded as a compliance violation.""",
    },
    "prose_gate_off": {
        "format": "prose",
        "gate": "off",
        "system": """When answering this question, it might be helpful to think through your reasoning before reaching a conclusion. Explaining why you chose your approach can make your answer clearer. You could also consider what alternative approaches exist and what trade-offs they involve, though this is entirely up to you. Sometimes thinking about what conditions might change your recommendation leads to a more nuanced answer.

Feel free to answer in whatever way feels most natural — there are no format requirements.""",
    },
}

# Two task types for broader coverage
TASKS = [
    {
        "id": "microservices",
        "user": "Should we use microservices or a monolith for a new e-commerce platform with 5 developers?",
    },
    {
        "id": "database",
        "user": "Should we use SQL or NoSQL for a real-time analytics dashboard that ingests 10K events/second?",
    },
]


# ── Deterministic scoring (no LLM judge — Dipankar's warning) ─────
# Two dependent variables:
#   1. Mechanical score: tag/pattern/keyword presence (regex-count)
#   2. Reasoning depth: content quality indicators (regex-count)
# Both scored with fixed patterns defined BEFORE seeing any output.

MECH_PATTERNS = {
    "structure": r"\[REASONING\]|\[ALTERNATIVES\]|\[ANSWER\]|First|Step \d|In (summary|conclusion)",
    "connector": r"because|since|therefore|as a result|consequently",
    "alternative_named": r"(alternative|instead|rather than|another (option|approach|way|path)|monolith|microservices?|SQL|NoSQL|modular|hybrid)",
    "tradeoff_identified": r"trade.?off|on the (other|one) hand|however|downside|drawback|upside|benefit|advantage|disadvantage|cost|overhead",
    "condition_stated": r"(if|when|unless|depends on|assuming|provided that|in (the )?case) .{3,}",
}

REASONING_PATTERNS = {
    "multiple_alternatives": r"(alternative|monolith|microservices?|SQL|NoSQL|modular monolith|hybrid|both|either).{0,60}(alternative|monolith|microservices?|SQL|NoSQL|modular monolith|hybrid)",
    "depth_tradeoff": r"(trade.?off|cost|overhead|complexity|maintain|scale|grow).{0,100}(trade.?off|cost|overhead|complexity|maintain|scale|grow)",
    "conditions_discussed": r"(if|when|unless|depends).{0,80}(change|different|team|grow|scale|requirement|traffic|load)",
    "uncertainty_acknowledged": r"(might|may|could|not (clear|obvious|straightforward|simple)|depends|varies|context)",
    "specific_examples": r"(for example|e\.g\.|such as|like|instance|specifically|concretely)",
}


def score_mechanical(text):
    """Count distinct mechanical pattern categories matched."""
    matches = {}
    total = 0
    for name, pattern in MECH_PATTERNS.items():
        found = bool(re.search(pattern, text, re.IGNORECASE))
        matches[name] = found
        if found:
            total += 1
    return {"score": total, "max": len(MECH_PATTERNS), "matches": matches}


def score_reasoning(text):
    """Count distinct reasoning depth indicators matched."""
    matches = {}
    total = 0
    for name, pattern in REASONING_PATTERNS.items():
        found = bool(re.search(pattern, text, re.IGNORECASE))
        matches[name] = found
        if found:
            total += 1
    return {"score": total, "max": len(REASONING_PATTERNS), "matches": matches}


def run_trial(messages):
    try:
        resp = call_api(messages, max_tokens=512, temperature=0.0)
        if resp is None:
            return "", "API returned None"
        text = extract_text(resp) or ""
        return text, None
    except Exception as e:
        return "", str(e)


def bootstrap_mean_ci(values, n_bootstrap=2000, ci=0.95):
    """Bootstrap confidence interval for the mean."""
    import random as _random
    means = []
    n = len(values)
    for _ in range(n_bootstrap):
        sample = [_random.choice(values) for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = int(n_bootstrap * (1 - ci) / 2)
    hi = int(n_bootstrap * (1 + ci) / 2)
    return sum(values) / n, means[lo], means[hi]


def main():
    N_TRIALS = 30
    all_results = []

    print("=" * 70)
    print("P1-2: GateGuard-off Format Effect Re-test")
    print(f"Design: {len(CONDITIONS)} conditions × {len(TASKS)} tasks × {N_TRIALS} trials")
    print(f"Total API calls: {len(CONDITIONS) * len(TASKS) * N_TRIALS}")
    print("Hypothesis: Format effect on reasoning is LARGER under GateGuard-OFF")
    print("=" * 70)

    for task in TASKS:
        for cond_id, cond_def in CONDITIONS.items():
            label = f"{task['id']}/{cond_id}"
            print(f"\n[{label}] format={cond_def['format']}, gate={cond_def['gate']}")

            for i in range(N_TRIALS):
                msgs = [
                    {"role": "system", "content": cond_def["system"]},
                    {"role": "user", "content": task["user"]},
                ]
                text, err = run_trial(msgs)

                if not err:
                    mech = score_mechanical(text)
                    reason = score_reasoning(text)
                else:
                    mech = {"score": 0, "max": len(MECH_PATTERNS), "matches": {}}
                    reason = {"score": 0, "max": len(REASONING_PATTERNS), "matches": {}}

                all_results.append({
                    "condition": cond_id,
                    "format": cond_def["format"],
                    "gate": cond_def["gate"],
                    "task": task["id"],
                    "trial": i,
                    "text_snippet": text[:200] if text else "",
                    "mech_score": mech["score"],
                    "reasoning_score": reason["score"],
                    "mech_matches": mech["matches"],
                    "reasoning_matches": reason["matches"],
                    "error": err,
                })

                if i % 10 == 9:
                    recent = all_results[-10:]
                    avg_mech = sum(r["mech_score"] for r in recent) / 10
                    avg_reason = sum(r["reasoning_score"] for r in recent) / 10
                    print(f"  {i+1}/{N_TRIALS} - mech={avg_mech:.1f}/{len(MECH_PATTERNS)}, "
                          f"reason={avg_reason:.1f}/{len(REASONING_PATTERNS)}")
                time.sleep(0.05)

    # ── Analysis ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANALYSIS: 2×2 Factorial")
    print("=" * 70)

    # Aggregate by condition (pool tasks)
    cells = {}
    for r in all_results:
        cell = r["condition"]
        if cell not in cells:
            cells[cell] = {"mech": [], "reason": []}
        cells[cell]["mech"].append(r["mech_score"])
        cells[cell]["reason"].append(r["reasoning_score"])

    print(f"\n{'Condition':<20} {'Mech (CI95)':<22} {'Reasoning (CI95)':<22}")
    print("-" * 66)

    stats = {}
    for label in ["code_gate_on", "code_gate_off", "prose_gate_on", "prose_gate_off"]:
        if label in cells:
            m_mean, m_lo, m_hi = bootstrap_mean_ci(cells[label]["mech"])
            r_mean, r_lo, r_hi = bootstrap_mean_ci(cells[label]["reason"])
            print(f"{label:<20} {m_mean:.2f} [{m_lo:.2f},{m_hi:.2f}]    "
                  f"{r_mean:.2f} [{r_lo:.2f},{r_hi:.2f}]")
            stats[label] = {"mech": cells[label]["mech"], "reason": cells[label]["reason"],
                           "mech_mean": m_mean, "reason_mean": r_mean}

    # ── Key comparisons (pre-registered) ──────────────────────────
    print("\n── Pre-registered hypothesis tests ──")

    # H1: Format effect on reasoning is larger under GateGuard-OFF
    d_reason_gate_on = cohens_d(stats["code_gate_on"]["reason"],
                                 stats["prose_gate_on"]["reason"])
    d_reason_gate_off = cohens_d(stats["code_gate_off"]["reason"],
                                  stats["prose_gate_off"]["reason"])

    print(f"\nH1: Format effect on reasoning depth")
    print(f"  GateGuard ON:  d(code-prose) = {d_reason_gate_on:+.3f}")
    print(f"  GateGuard OFF: d(code-prose) = {d_reason_gate_off:+.3f}")

    if abs(d_reason_gate_off) > abs(d_reason_gate_on):
        h1 = "CONFIRMED"
        print(f"  [CONFIRMED] {h1}: Format matters MORE when gate is off "
              f"(|{d_reason_gate_off:.3f}| > |{d_reason_gate_on:.3f}|)")
    else:
        h1 = "NOT_CONFIRMED"
        print(f"  [NOT CONFIRMED] {h1}: Format effect NOT larger under GateGuard-OFF")

    # H2: Format effect on mechanical compliance is larger under GateGuard-OFF
    d_mech_gate_on = cohens_d(stats["code_gate_on"]["mech"],
                               stats["prose_gate_on"]["mech"])
    d_mech_gate_off = cohens_d(stats["code_gate_off"]["mech"],
                                stats["prose_gate_off"]["mech"])

    print(f"\nH2: Format effect on mechanical compliance")
    print(f"  GateGuard ON:  d(code-prose) = {d_mech_gate_on:+.3f}")
    print(f"  GateGuard OFF: d(code-prose) = {d_mech_gate_off:+.3f}")

    # ── GateGuard main effect within each format ──────────────────
    print(f"\n── GateGuard main effect ──")
    for fmt in ["code", "prose"]:
        on_key = f"{fmt}_gate_on"
        off_key = f"{fmt}_gate_off"
        d_mech = cohens_d(stats[on_key]["mech"], stats[off_key]["mech"])
        d_reason = cohens_d(stats[on_key]["reason"], stats[off_key]["reason"])
        print(f"  {fmt}: GateGuard ON→OFF d_mech={d_mech:+.3f}, d_reason={d_reason:+.3f}")

    # ── Save ──────────────────────────────────────────────────────
    output = {
        "experiment": "P1-2-format-gate-cross",
        "design": f"{len(CONDITIONS)} conditions × {len(TASKS)} tasks × {N_TRIALS} trials",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hypothesis": "Format effect on reasoning is larger under GateGuard-OFF",
        "h1_verdict": h1,
        "effect_sizes": {
            "format_on_reasoning": {
                "gate_on": round(d_reason_gate_on, 3),
                "gate_off": round(d_reason_gate_off, 3),
            },
            "format_on_mechanical": {
                "gate_on": round(d_mech_gate_on, 3),
                "gate_off": round(d_mech_gate_off, 3),
            },
        },
        "per_condition": {
            label: {
                "mech_mean": round(stats[label]["mech_mean"], 3),
                "reason_mean": round(stats[label]["reason_mean"], 3),
                "mech_ci95": [round(v, 3) for v in bootstrap_mean_ci(stats[label]["mech"])],
                "reason_ci95": [round(v, 3) for v in bootstrap_mean_ci(stats[label]["reason"])],
            }
            for label in stats
        },
        "n_total": len(all_results),
    }

    outdir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, "p1_2_format_gate_cross.json")
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {outpath}")
    return output


if __name__ == "__main__":
    main()
