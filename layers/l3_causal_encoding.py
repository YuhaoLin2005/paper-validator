"""L3 — Causal Encoding: EvalField (5-persona consensus) + CanonizationPipeline.

Imported by: layers/strange_loop.py, claims/base.py
Data: calls DeepSeek API via engine/api_client.py quick_chat.

Collapses eval-field.py, canonization.py, dual-pool review methodology.
Each persona evaluates candidates independently; 3/5 consensus + 24h cooling → canonize.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from state.store import StateStore

from config.defaults import EVAL_PERSONAS
from engine.api_client import call_api, extract_text


@dataclass
class EvalResult:
    persona: str
    score: float
    reasoning: str = ""
    concerns: list[str] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)


@dataclass
class ConsensusResult:
    rule_id: str
    rule_title: str
    eval_results: list[EvalResult]
    mean_score: float
    consensus: float
    passed: bool
    cooling_expiry: float


class EvalField:
    """Independent evaluation by 5 named personas (the dual-pool internalized).

    Each persona has a distinct philosophical lens. They evaluate rule candidates
    independently (separate API calls), then results are aggregated for consensus.
    """

    MIN_CONSENSUS = 0.6   # 3/5 personas must score >= 0.5
    MIN_MEAN = 0.5
    COOLING_HOURS = 24

    def __init__(self, api_key: Optional[str] = None,
                 state_store: Optional["StateStore"] = None):
        self._api_key = api_key
        self._store = state_store
        self._personas = EVAL_PERSONAS
        self._history: list[ConsensusResult] = []

    @property
    def personas(self) -> list[dict]:
        return list(self._personas)

    def _build_persona_prompt(self, persona: dict, candidate: dict) -> str:
        return (
            f"You are {persona['name']}, {persona['role']}. "
            f"Your evaluation lens: {persona['lens']}.\n\n"
            f"Evaluate this proposed rule amendment:\n"
            f"  ID: {candidate['rule_id']}\n"
            f"  Title: {candidate['title']}\n"
            f"  Text: {candidate['text']}\n"
            f"  Layer: {candidate.get('layer', 'L1')}\n"
            f"  Mechanizable: {candidate.get('mechanizable', False)}\n\n"
            f"Rate from 0.0 (reject) to 1.0 (strongly accept). Reply in JSON:\n"
            f'{{"score": 0.0-1.0, "reasoning": "...", '
            f'"concerns": ["..."], "strengths": ["..."]}}'
        )

    def evaluate_sync(self, candidate: dict) -> list[EvalResult]:
        """Synchronous evaluation — one persona at a time."""
        results = []
        system_prompt = (
            "You are an expert evaluator. Always respond with valid JSON "
            "containing score, reasoning, concerns, and strengths fields."
        )
        for persona in self._personas:
            user_prompt = self._build_persona_prompt(persona, candidate)
            response = call_api(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                api_key=self._api_key,
                max_tokens=512,
            )
            text = extract_text(response) if response else "{}"
            results.append(self._parse_response(persona["name"], text))
        return results

    def _parse_response(self, persona_name: str, response: str) -> EvalResult:
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
            else:
                data = {}
        except (json.JSONDecodeError, ValueError):
            data = {}
        return EvalResult(
            persona=persona_name,
            score=float(data.get("score", 0.0)),
            reasoning=str(data.get("reasoning", "")),
            concerns=list(data.get("concerns", [])),
            strengths=list(data.get("strengths", [])),
        )

    def reach_consensus(self, results: list[EvalResult],
                        rule_id: str, rule_title: str) -> ConsensusResult:
        if not results:
            return ConsensusResult(
                rule_id=rule_id, rule_title=rule_title,
                eval_results=[], mean_score=0.0, consensus=0.0,
                passed=False, cooling_expiry=time.time(),
            )
        scores = [r.score for r in results]
        mean_score = sum(scores) / len(scores)
        approving = sum(1 for s in scores if s >= 0.5)
        consensus = approving / len(scores)
        passed = consensus >= self.MIN_CONSENSUS and mean_score >= self.MIN_MEAN
        return ConsensusResult(
            rule_id=rule_id, rule_title=rule_title,
            eval_results=results,
            mean_score=round(mean_score, 3),
            consensus=round(consensus, 3),
            passed=passed,
            cooling_expiry=time.time() + self.COOLING_HOURS * 3600,
        )

    def stats(self) -> dict:
        evaluations = len(self._history)
        passed = sum(1 for c in self._history if c.passed)
        return {
            "evaluations": evaluations,
            "passed": passed,
            "pass_rate": passed / evaluations if evaluations else 0.0,
            "mean_consensus": (
                sum(c.consensus for c in self._history) / evaluations
                if evaluations else 0.0
            ),
        }


class CanonizationPipeline:
    """Full amendment lifecycle: submit → eval → consensus → cooling → canonize."""

    def __init__(self, eval_field: EvalField,
                 state_store: Optional["StateStore"] = None):
        self._eval = eval_field
        self._store = state_store
        self._pending: dict[str, ConsensusResult] = {}

    def submit(self, candidate: dict) -> ConsensusResult:
        results = self._eval.evaluate_sync(candidate)
        consensus = self._eval.reach_consensus(
            results, candidate["rule_id"], candidate["title"])
        self._pending[candidate["rule_id"]] = consensus
        if self._store:
            self._store.log_audit("l3_evaluation", {
                "rule_id": candidate["rule_id"],
                "mean_score": consensus.mean_score,
                "consensus": consensus.consensus,
                "passed": consensus.passed,
            })
        return consensus

    def cooling_complete(self, rule_id: str) -> bool:
        if rule_id not in self._pending:
            return False
        return time.time() >= self._pending[rule_id].cooling_expiry

    def ready_to_canonize(self) -> list[str]:
        return [
            rid for rid, cr in self._pending.items()
            if cr.passed and self.cooling_complete(rid)
        ]

    def get_pending(self) -> dict[str, ConsensusResult]:
        return dict(self._pending)

    def clear(self, rule_id: str):
        self._pending.pop(rule_id, None)

    def build_prompt(self) -> str:
        lines = ["## Causal Encoding (L3)"]
        lines.append(f"- Personas: {len(self._eval.personas)}")
        lines.append(f"- Consensus threshold: {self._eval.MIN_CONSENSUS} "
                     f"({int(self._eval.MIN_CONSENSUS * 5)}/5)")
        lines.append(f"- Cooling period: {self._eval.COOLING_HOURS}h")
        eval_stats = self._eval.stats()
        if eval_stats["evaluations"] > 0:
            lines.append(f"- Evaluations: {eval_stats['evaluations']} "
                         f"({eval_stats['pass_rate']:.0%} pass rate)")
        if self._pending:
            lines.append("- Pending amendments:")
            for rid, cr in self._pending.items():
                status = "cooling" if cr.passed else "rejected"
                remaining = max(0, cr.cooling_expiry - time.time())
                lines.append(f"  - {rid}: {status} "
                             f"(consensus={cr.consensus:.2f}, "
                             f"cooling={remaining/3600:.1f}h)")
        return "\n".join(lines)
