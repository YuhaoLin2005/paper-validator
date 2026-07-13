"""L2 — Neural Gate: logprob-differential constraint fidelity measurement.

Imported by: claims/base.py, main.py, layers/strange_loop.py
Data: calls DeepSeek API via engine/api_client.py for logprob probes; no local data files.

Collapses neural-gate-v2.py (~19K) into a module that reuses engine's api_client
instead of making its own urllib calls.
"""

from __future__ import annotations

from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from state.store import StateStore

from config.defaults import (
    DEFAULT_CONSTRAINT_PROBES, DEEPSEEK_MODEL, DEEPSEEK_MAX_TOKENS,
)
from engine.api_client import call_api, extract_logprobs


class NeuralGate:
    """Logprob-differential constraint fidelity measurement.

    Measures whether constraints are "neurally active" — do they shift
    the model's token probabilities at constraint-relevant decision points?

    A constraint is "active" if the mean logprob of compliant tokens
    increases (vs violating tokens) when the constraint is present
    in the system prompt compared to absent.
    """

    ACTIVE_THRESHOLD = 0.3  # logprob units

    def __init__(self, api_key: Optional[str] = None,
                 state_store: Optional["StateStore"] = None):
        self._api_key = api_key
        self._store = state_store
        self._probes = DEFAULT_CONSTRAINT_PROBES
        self._results: dict[str, dict] = {}

    @property
    def probes(self) -> list[dict]:
        return list(self._probes)

    def measure(self, probe: dict) -> Optional[dict]:
        """Measure a single constraint probe.

        Sends two API calls per probe:
          1. WITH constraint in system prompt → measure compliant token logprobs
          2. WITHOUT constraint → measure baseline logprobs

        Returns:
            {probe_id, mean_differential, compliant_logprobs, violating_logprobs, active}
        """
        constraint = probe["constraint"]
        context = probe["context"]
        compliant_tokens = probe["compliant_tokens"]
        violating_tokens = probe["violating_tokens"]

        system_with = (
            "You are a coding assistant. Respond with exactly ONE word — "
            "the most natural continuation given the context.\n"
            f"CONSTRAINT: {constraint}"
        )
        system_without = "You are a coding assistant. Respond with exactly ONE word."

        user = (
            f"Context: {context}\n"
            "What is the most natural next action? Answer with ONE word:"
        )

        resp_with = call_api(
            [
                {"role": "system", "content": system_with},
                {"role": "user", "content": user},
            ],
            api_key=self._api_key,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )

        resp_without = call_api(
            [
                {"role": "system", "content": system_without},
                {"role": "user", "content": user},
            ],
            api_key=self._api_key,
            max_tokens=1,
            logprobs=True,
            top_logprobs=20,
        )

        if not resp_with or not resp_without:
            return None

        logprobs_with = extract_logprobs(resp_with)
        logprobs_without = extract_logprobs(resp_without)

        if not logprobs_with or not logprobs_without:
            return None

        top_with = {t["token"]: t["logprob"]
                    for t in logprobs_with[0].get("top_logprobs", [])}
        top_without = {t["token"]: t["logprob"]
                       for t in logprobs_without[0].get("top_logprobs", [])}

        def mean_logprob(tokens: list[str], top_logprobs: dict[str, float]) -> float:
            values = [top_logprobs.get(t, -10.0) for t in tokens]
            return sum(values) / len(values) if values else 0.0

        compliant_with = mean_logprob(compliant_tokens, top_with)
        violating_with = mean_logprob(violating_tokens, top_with)
        compliant_without = mean_logprob(compliant_tokens, top_without)
        violating_without = mean_logprob(violating_tokens, top_without)

        delta_with = compliant_with - violating_with
        delta_without = compliant_without - violating_without
        mean_differential = delta_with - delta_without

        active = abs(mean_differential) >= self.ACTIVE_THRESHOLD

        result = {
            "probe_id": probe["id"],
            "mean_differential": round(mean_differential, 4),
            "delta_with_constraint": round(delta_with, 4),
            "delta_without_constraint": round(delta_without, 4),
            "compliant_logprobs": {
                "with": {t: top_with.get(t, -10.0) for t in compliant_tokens},
                "without": {t: top_without.get(t, -10.0) for t in compliant_tokens},
            },
            "violating_logprobs": {
                "with": {t: top_with.get(t, -10.0) for t in violating_tokens},
                "without": {t: top_without.get(t, -10.0) for t in violating_tokens},
            },
            "active": active,
        }

        self._results[probe["id"]] = result
        return result

    def measure_all(self) -> dict[str, dict]:
        results = {}
        for probe in self._probes:
            result = self.measure(probe)
            if result:
                results[probe["id"]] = result
        return results

    def active_count(self) -> int:
        return sum(1 for r in self._results.values() if r.get("active", False))

    def mean_activation(self) -> float:
        if not self._results:
            return 0.0
        return sum(abs(r.get("mean_differential", 0))
                   for r in self._results.values()) / len(self._results)

    def get_results(self) -> dict[str, dict]:
        return dict(self._results)

    def build_prompt(self) -> str:
        lines = ["## Neural Gate Status (L2)"]
        if self._results:
            active = self.active_count()
            total = len(self._results)
            lines.append(f"- {active}/{total} constraints neurally active")
            lines.append(f"- Mean activation: {self.mean_activation():.2f} logprob units")
            lines.append(f"- Threshold: {self.ACTIVE_THRESHOLD}")
            for pid, r in self._results.items():
                status = "ACTIVE" if r["active"] else "inactive"
                lines.append(f"  - {pid}: {status} (Δ={r['mean_differential']:.3f})")
        else:
            lines.append("- No measurements yet — run measure_all()")
        return "\n".join(lines)
