"""Claim experiment orchestrator — run one or all claims.

Imported by: main.py (cmd_claim)
Data: loads claim modules dynamically via importlib; collects results via StateStore.
"""

from __future__ import annotations

import importlib
import time
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from state.store import StateStore
from claims.base import ClaimReport, BaseClaim

CLAIM_REGISTRY: dict[str, str] = {
    "l0-safety-prompt": "claims.l0_safety_prompt",
    "causal-swap": "claims.causal_swap",
    "logprob-probe-v3": "claims.logprob_probe_v3",
    "dissociation": "claims.dissociation",
    "gateguard-off": "claims.gateguard_off",
    "l1-visibility": "claims.l1_visibility",
    "prose-barrier": "claims.prose_barrier",
    "cross-model": "claims.cross_model",
}


def list_claims() -> list[str]:
    return list(CLAIM_REGISTRY.keys())


def get_claim_module(claim_name: str):
    if claim_name not in CLAIM_REGISTRY:
        print(f"Unknown: {claim_name}. Available: {list_claims()}")
        return None
    try:
        return importlib.import_module(CLAIM_REGISTRY[claim_name])
    except ImportError as e:
        print(f"Import error {claim_name}: {e}")
        return None


def run_claim(claim_name: str, n_trials: int = 30,
              api_key: Optional[str] = None,
              state_store: Optional["StateStore"] = None,
              logprobs: bool = False) -> Optional[ClaimReport]:
    mod = get_claim_module(claim_name)
    if mod is None:
        return None

    cls_name = "".join(w.capitalize() for w in claim_name.replace("-", "_").split("_"))
    claim_class = getattr(mod, cls_name, None)

    if claim_class is None:
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type) and issubclass(obj, BaseClaim) and obj is not BaseClaim:
                claim_class = obj
                break

    if claim_class is None:
        print(f"No claim class in {claim_name}")
        return None

    instance = claim_class(api_key=api_key, state_store=state_store)
    return instance.run(n_trials=n_trials, logprobs=logprobs)


def run_all(n_trials: int = 30, api_key: Optional[str] = None,
            state_store: Optional["StateStore"] = None) -> dict[str, Optional[ClaimReport]]:
    results: dict[str, Optional[ClaimReport]] = {}
    for i, name in enumerate(CLAIM_REGISTRY):
        print(f"\n{'='*50}\n[{i+1}/{len(CLAIM_REGISTRY)}] {name}\n{'='*50}")
        start = time.time()
        report = run_claim(name, n_trials=n_trials, api_key=api_key,
                          state_store=state_store)
        if report:
            print(f"  Verdict: {report.verdict} ({time.time()-start:.0f}s, "
                  f"{len(report.errors)} errs)")
        else:
            print(f"  FAILED")
        results[name] = report
    return results


def summarize(results: dict[str, Optional[ClaimReport]]) -> str:
    lines = ["=" * 70]
    lines.append(f"{'Claim':<25} {'Trials':>8} {'Verdict':<22} {'d':>6} {'Errs':>5}")
    lines.append("-" * 70)
    confirmed = partial = rejected = 0
    for name, report in results.items():
        if report is None:
            lines.append(f"{name:<25} {'N/A':>8} {'IMPORT ERROR':<22} {'-':>6} {'-':>5}")
            continue
        d = report.effect_size or 0.0
        lines.append(f"{name:<25} {report.total_trials:>8} {report.verdict:<22} "
                     f"{d:>6.2f} {len(report.errors):>5}")
        if report.verdict == "CONFIRMED": confirmed += 1
        elif report.verdict == "PARTIALLY_CONFIRMED": partial += 1
        else: rejected += 1
    lines.append("-" * 70)
    lines.append(f"Summary: {confirmed} confirmed, {partial} partial, {rejected} rejected")
    lines.append("=" * 70)
    return "\n".join(lines)
