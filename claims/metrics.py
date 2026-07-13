"""Statistical metrics for claim experiments — stdlib only.

Imported by: claims/base.py, claims/runner.py, claim modules
Data: Pure computation (math, random), no API or file I/O.

Return dict schemas:
  logprob_differential → {mean_diff, ci_low, ci_high, cohens_d, significant}
  response_rate_diff → {rate_treatment, rate_baseline, diff}
  classify_verdict → "CONFIRMED" | "PARTIALLY_CONFIRMED" | "REJECTED"
"""

from __future__ import annotations

import math
import random
from typing import Optional


def cohens_d(group_a: list[float], group_b: list[float]) -> float:
    """Cohen's d: (mean_a - mean_b) / pooled_std."""
    if not group_a or not group_b:
        return 0.0
    mean_a = sum(group_a) / len(group_a)
    mean_b = sum(group_b) / len(group_b)
    var_a = sum((x - mean_a) ** 2 for x in group_a) / (len(group_a) - 1) if len(group_a) > 1 else 0.0
    var_b = sum((x - mean_b) ** 2 for x in group_b) / (len(group_b) - 1) if len(group_b) > 1 else 0.0
    n_a, n_b = len(group_a), len(group_b)
    pooled_var = ((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2) if (n_a + n_b) > 2 else 1.0
    pooled_std = math.sqrt(max(pooled_var, 1e-10))
    return (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0.0


def bootstrap_ci(data: list[float], n_bootstrap: int = 1000,
                 confidence: float = 0.95) -> tuple[float, float]:
    """Bootstrap CI for mean. Returns (lower, upper)."""
    if not data:
        return (0.0, 0.0)
    if len(data) < 5:
        m = sum(data) / len(data)
        return (m, m)
    n = len(data)
    rng = random.Random(42)
    means = [sum(data[rng.randint(0, n - 1)] for _ in range(n)) / n
             for _ in range(n_bootstrap)]
    means.sort()
    alpha = (1 - confidence) / 2
    return (means[int(alpha * n_bootstrap)], means[int((1 - alpha) * n_bootstrap)])


def logprob_differential(results_treatment: list[dict],
                         results_baseline: list[dict],
                         target_tokens: Optional[list[str]] = None) -> dict:
    """Mean logprob difference between treatment and baseline.

    Each result dict should have "logprobs" key with token-level data.
    """
    def extract_top1(r):
        if not r.get("logprobs"):
            return None
        top = r["logprobs"][0].get("top_logprobs", [])
        return top[0]["logprob"] if top else None

    def extract_mean(r, tokens):
        if not r.get("logprobs"):
            return None
        top = {}
        for lp_entry in r["logprobs"]:
            for t in lp_entry.get("top_logprobs", []):
                top[t["token"]] = t["logprob"]
        vals = [top.get(t, -10.0) for t in (tokens or [])]
        return sum(vals) / len(vals) if vals else 0.0

    extract = extract_mean if target_tokens else extract_top1

    t_vals = [v for r in results_treatment if (v := extract(r)) is not None]
    b_vals = [v for r in results_baseline if (v := extract(r)) is not None]

    if not t_vals or not b_vals:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0,
                "n_treatment": 0, "n_baseline": 0, "cohens_d": 0.0, "significant": False}

    mean_t = sum(t_vals) / len(t_vals)
    mean_b = sum(b_vals) / len(b_vals)

    all_vals = t_vals + b_vals
    n_t = len(t_vals)
    diffs = []
    rng = random.Random(42)
    for _ in range(1000):
        rng.shuffle(all_vals)
        st, sb = all_vals[:n_t], all_vals[n_t:]
        diffs.append((sum(st) / len(st)) - (sum(sb) / len(sb)) if sb else 0)
    diffs.sort()

    d = cohens_d(t_vals, b_vals)
    significant = diffs[25] > 0 or diffs[974] < 0

    return {
        "mean_diff": round(mean_t - mean_b, 4),
        "ci_low": round(diffs[25], 4),
        "ci_high": round(diffs[974], 4),
        "n_treatment": n_t, "n_baseline": len(b_vals),
        "cohens_d": round(d, 3), "significant": significant,
    }


def response_rate_diff(results_treatment: list[str],
                       results_baseline: list[str],
                       match_pattern: str) -> dict:
    """Compare response rates matching pattern between conditions."""
    def rate(responses):
        if not responses:
            return 0.0
        p = match_pattern.lower()
        return sum(1 for r in responses if p in r.lower()) / len(responses)
    rt, rb = rate(results_treatment), rate(results_baseline)
    return {
        "rate_treatment": round(rt, 3), "rate_baseline": round(rb, 3),
        "diff": round(rt - rb, 3),
        "n_treatment": len(results_treatment), "n_baseline": len(results_baseline),
    }


def classify_verdict(p_value: Optional[float], effect_size: Optional[float],
                     significant: bool = False,
                     threshold_p: float = 0.05,
                     threshold_d: float = 0.3) -> str:
    """CONFIRMED (p<0.05 & |d|>=0.3) | PARTIALLY_CONFIRMED | REJECTED."""
    has_p = p_value is not None and p_value < threshold_p
    has_d = effect_size is not None and abs(effect_size) >= threshold_d
    if has_p and has_d:
        return "CONFIRMED"
    elif has_p or has_d or significant:
        return "PARTIALLY_CONFIRMED"
    return "REJECTED"
