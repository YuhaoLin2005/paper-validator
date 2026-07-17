"""Regex Gap Measurement — Does the Detection Gap Affect d=0.605?
==================================================================
Mike Czerwinski (DEV.to, 2026-07-16):
  "Does the 8% detection gap touch d=0.605 the same way it touched fact_check?"

Measures regex-vs-human scoring gap. Produces: detection rate, gap magnitude,
sensitivity analysis — how big must the gap be to flip the headline result?
"""

from __future__ import annotations

import json, random, math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class GapResult:
    total_reviewed: int
    agreement: int
    regex_only: int      # Regex OK, human says violation (regex MISS)
    human_only: int      # Human OK, regex says violation (regex FLAG)
    detection_rate: float
    false_positive_rate: float
    kappa: float


def generate_review_queue(
    data_path: str | Path,
    sample_size: int = 20,
    seed: int = 42,
    output: str = "",
) -> list[dict]:
    """Sample N trials for second-rater review. Outputs JSON with human_score=TBD."""
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))
    random.seed(seed)

    trials = []
    items = data.items() if isinstance(data, dict) else [("all", data)]
    for condition, results in items:
        for i, trial in enumerate(results if isinstance(results, list) else []):
            trials.append({
                "condition": condition, "trial_index": i,
                "rule_id": trial.get("rule_id", ""),
                "output_text": trial.get("output", trial.get("response", ""))[:500],
                "regex_score": trial.get("compliance", trial.get("score", None)),
            })

    sampled = random.sample(trials, min(sample_size, len(trials)))
    queue = [{**t, "human_score": "TBD", "notes": ""} for t in sampled]

    if output:
        Path(output).write_text(json.dumps(queue, indent=2, ensure_ascii=False), encoding="utf-8")
    return queue


def compare_regex_vs_human(regex_path: str | Path, human_path: str | Path) -> GapResult:
    """Compare regex-scored results against human-reviewed results."""
    regex_data = json.loads(Path(regex_path).read_text(encoding="utf-8"))
    human_data = json.loads(Path(human_path).read_text(encoding="utf-8"))

    regex_map = {
        (r.get("condition", ""), r.get("trial_index", -1), r.get("rule_id", "")):
        int(r.get("compliance", r.get("regex_score", 0)))
        for r in regex_data
    }

    agreement = regex_only = human_only = total = 0
    for h in human_data:
        key = (h.get("condition", ""), h.get("trial_index", -1), h.get("rule_id", ""))
        if key not in regex_map:
            continue
        r = regex_map[key]
        hu = int(h.get("human_score", h.get("compliance", 0)))
        if r == hu:
            agreement += 1
        elif r and not hu:
            regex_only += 1
        elif hu and not r:
            human_only += 1
        total += 1

    if total == 0:
        return GapResult(0, 0, 0, 0, 0.0, 0.0, 0.0)

    det = (total - regex_only) / total
    fp = human_only / total
    p_o = agreement / total
    r_comp = sum(int(_.get("compliance", 0)) for _ in regex_data) / max(1, len(regex_data))
    h_comp = sum(int(_.get("human_score", 0)) for _ in human_data) / max(1, len(human_data))
    p_e = r_comp * h_comp + (1 - r_comp) * (1 - h_comp)
    kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0.0

    return GapResult(
        total_reviewed=total, agreement=agreement,
        regex_only=regex_only, human_only=human_only,
        detection_rate=round(det, 4),
        false_positive_rate=round(fp, 4),
        kappa=round(kappa, 4),
    )


def sensitivity_analysis(
    d: float = 0.605,
    gap_range: tuple[float, float, float] = (0.0, 0.20, 0.02),
) -> list[dict]:
    """How large must the gap be to flip d below 0.2 (small effect)?"""
    results = []
    for gap in (round(gap_range[0] + i * gap_range[2], 3)
                for i in range(int((gap_range[1] - gap_range[0]) / gap_range[2]) + 1)):
        adjusted = round(d * (1.0 - gap), 4)
        results.append({
            "gap": round(gap, 3),
            "adjusted_d": adjusted,
            "d_loss": round(d - adjusted, 4),
            "flips": adjusted < 0.2,
        })
    return results


def estimate_gap_impact(data_path: str | Path, gap: float = 0.08) -> dict:
    """Estimate impact of detection gap on a specific experiment's results."""
    data = json.loads(Path(data_path).read_text(encoding="utf-8"))

    rates = {}
    for key, trials in (data.items() if isinstance(data, dict) else [("all", data)]):
        if isinstance(trials, list) and trials:
            comp = sum(int(t.get("compliance", t.get("score", 0))) for t in trials) / len(trials)
            rates[key] = round(comp, 4)

    adjusted = {
        cond: round(rate + gap * (1.0 - rate), 4)
        for cond, rate in rates.items()
    }

    pairs = []
    keys = list(rates.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            raw = rates[keys[i]] - rates[keys[j]]
            adj = adjusted[keys[i]] - adjusted[keys[j]]
            pairs.append({
                "pair": f"{keys[i]} vs {keys[j]}",
                "raw_diff": round(raw, 4),
                "adj_diff": round(adj, 4),
                "delta": round(adj - raw, 4),
            })

    return {
        "gap": gap, "rates": rates, "adjusted": adjusted,
        "pairs": pairs,
        "conclusion_stable": all(abs(p["delta"]) < 0.05 for p in pairs),
    }


# ── CLI ──

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Regex Gap Measurement")
    sp = p.add_subparsers(dest="cmd")

    r = sp.add_parser("review", help="Generate review queue")
    r.add_argument("--data", required=True)
    r.add_argument("--sample", type=int, default=20)
    r.add_argument("--output", default="review_queue.json")

    c = sp.add_parser("compare", help="Compare regex vs human")
    c.add_argument("--regex", required=True)
    c.add_argument("--human", required=True)

    s = sp.add_parser("sensitivity", help="Sensitivity analysis")
    s.add_argument("--d", type=float, default=0.605)

    e = sp.add_parser("estimate", help="Estimate gap impact")
    e.add_argument("--data", required=True)
    e.add_argument("--gap", type=float, default=0.08)

    a = p.parse_args()

    if a.cmd == "review":
        q = generate_review_queue(a.data, a.sample, output=a.output)
        print(f"Review queue: {len(q)} trials → {a.output}")

    elif a.cmd == "compare":
        r = compare_regex_vs_human(a.regex, a.human)
        print(f"Agreement: {r.agreement}/{r.total_reviewed} ({r.agreement/r.total_reviewed*100:.1f}%)")
        print(f"Regex misses: {r.regex_only}  Human-only: {r.human_only}")
        print(f"Detection: {r.detection_rate:.1%}  FP rate: {r.false_positive_rate:.1%}  κ={r.kappa:.3f}")

    elif a.cmd == "sensitivity":
        for r in sensitivity_analysis(d=a.d):
            flag = " ⚠ FLIPS" if r["flips"] else ""
            print(f"  gap={r['gap']:.0%} → d={r['adjusted_d']:.4f} (Δ={r['d_loss']:.4f}){flag}")

    elif a.cmd == "estimate":
        r = estimate_gap_impact(a.data, gap=a.gap)
        print(f"Gap: {r['gap']:.0%}  Stable: {r['conclusion_stable']}")
        for pair in r["pairs"]:
            print(f"  {pair['pair']}: {pair['raw_diff']:.4f} → {pair['adj_diff']:.4f} (Δ={pair['delta']:+.4f})")

    else:
        p.print_help()
