"""Claim ABC — base class for all paper claim experiments.

Imported by: claims/*.py (all 8 claim modules), claims/runner.py
Data: calls DeepSeek API via engine/api_client.py; collects trial results.

Schemas:
  TrialResult: {trial_id, condition, response, logprobs, usage, latency_ms, error}
  ClaimReport: {claim_id, claim_title, total_trials, conditions, metrics,
                verdict, effect_size, p_value, confidence_interval, raw_results}
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from state.store import StateStore

from engine.api_client import call_api, extract_text, extract_logprobs


@dataclass
class TrialResult:
    trial_id: int
    condition: str
    response: str
    logprobs: Optional[list] = None
    usage: Optional[dict] = None
    latency_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


@dataclass
class ClaimReport:
    claim_id: str
    claim_title: str
    total_trials: int
    conditions: dict[str, int]
    metrics: dict
    verdict: str
    effect_size: Optional[float] = None
    p_value: Optional[float] = None
    confidence_interval: Optional[tuple] = None
    raw_results: list[TrialResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BaseClaim(ABC):
    """Abstract base for paper claim experiments.

    Subclasses implement:
      - build_prompts(): return {condition_name: [messages]}
      - analyze(): compute statistics from trial results
    """

    claim_id: str = "claim-0"
    claim_title: str = "Untitled Claim"

    def __init__(self, api_key: Optional[str] = None,
                 state_store: Optional["StateStore"] = None,
                 model: str = "deepseek-chat"):
        self._api_key = api_key
        self._store = state_store
        self._model = model
        self._results: list[TrialResult] = []

    @abstractmethod
    def build_prompts(self) -> dict[str, list[dict]]:
        """Build prompt variants for each condition.

        Returns: {condition_name: [{"role": ..., "content": ...}]}
        """
        ...

    @abstractmethod
    def analyze(self, results: list[TrialResult]) -> dict:
        """Compute statistics from trial results.

        Returns: {metric_name: value, effect_size, p_value,
                  confidence_interval, verdict}
        """
        ...

    def run_trial(self, trial_id: int, condition: str,
                  messages: list[dict], logprobs: bool = False) -> TrialResult:
        start = time.time()
        try:
            resp = call_api(
                messages, api_key=self._api_key, model=self._model,
                max_tokens=256, logprobs=logprobs,
                top_logprobs=20 if logprobs else None,
            )
            latency = (time.time() - start) * 1000
            if resp is None:
                return TrialResult(trial_id=trial_id, condition=condition,
                                   response="", error="API returned None",
                                   latency_ms=latency)
            text = extract_text(resp) or ""
            lp = extract_logprobs(resp) if logprobs else None
            return TrialResult(trial_id=trial_id, condition=condition,
                               response=text, logprobs=lp, usage=resp.get("usage"),
                               latency_ms=latency)
        except Exception as e:
            return TrialResult(trial_id=trial_id, condition=condition,
                               response="", error=str(e),
                               latency_ms=(time.time() - start) * 1000)

    def run(self, n_trials: int = 30, logprobs: bool = False) -> ClaimReport:
        prompts = self.build_prompts()
        conditions = list(prompts.keys())
        errors: list[str] = []

        print(f"  [{self.claim_id}] {self.claim_title}")
        print(f"  Conditions: {conditions}, n={n_trials} per condition")

        for ci, condition in enumerate(conditions):
            messages = prompts[condition]
            print(f"  [{ci+1}/{len(conditions)}] {condition}... ", end="", flush=True)

            for i in range(n_trials):
                tid = ci * n_trials + i + 1
                result = self.run_trial(tid, condition, messages, logprobs)
                self._results.append(result)
                if result.error:
                    errors.append(f"T{tid}/{condition}: {result.error}")
                if self._store:
                    self._store.save_trial(self.claim_id, tid, {
                        "condition": condition,
                        "response": result.response[:500],
                        "error": result.error,
                        "latency_ms": result.latency_ms,
                    })
            success = n_trials - sum(
                1 for r in self._results[-n_trials:] if r.error)
            print(f"{success}/{n_trials} OK")
            time.sleep(0.1)

        metrics = self.analyze(self._results)
        return ClaimReport(
            claim_id=self.claim_id, claim_title=self.claim_title,
            total_trials=len(self._results),
            conditions={c: n_trials for c in conditions},
            metrics=metrics, verdict=metrics.get("verdict", "UNKNOWN"),
            effect_size=metrics.get("effect_size"),
            p_value=metrics.get("p_value"),
            confidence_interval=metrics.get("confidence_interval"),
            raw_results=list(self._results), errors=errors,
        )

    def get_results(self) -> list[TrialResult]:
        return list(self._results)

    def success_rate(self) -> float:
        if not self._results:
            return 0.0
        return sum(1 for r in self._results if not r.error) / len(self._results)
