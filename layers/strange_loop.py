"""Strange Loop — self-referential regeneration cycle.

Imported by: main.py
Data: orchestrates L0-L4 layers + StateStore. Writes self-model.md on regeneration.

The "strange loop" (Hofstadter) closes the feedback loop:
the agent detects its own self-model staleness, regenerates it, validates
the result, and audits the process.

6-phase cycle (5 mechanized + 1 AI):
  detect → trigger → regenerate(AI) → validate → audit → clear
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from state.store import StateStore
    from state.flags import FlagManager

from state.flags import Flag


@dataclass
class LoopResult:
    phase: str
    success: bool
    details: str
    timestamp: str = field(default_factory=lambda: time.strftime(
        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()))


class StrangeLoop:
    """Self-referential regeneration cycle bridging all 5 layers.

    L0 → governance rules for the cycle
    L1 → detect stale flag, validate structure
    L2 → measure constraint fidelity after regeneration
    L3 → evaluate regeneration quality via persona consensus
    L4 → assess drift risk before/after
    """

    MAX_VERSIONS = 3
    SELF_MODEL_PATH = "~/.claude/projects/C--Users-86131/memory/self-model.md"

    def __init__(self, store: "StateStore", flags: "FlagManager",
                 constitution=None, gates=None,
                 neural_gate=None, eval_field=None, drift=None):
        self._store = store
        self._flags = flags
        self._constitution = constitution
        self._gates = gates
        self._neural_gate = neural_gate
        self._eval_field = eval_field
        self._drift = drift
        self._history: list[LoopResult] = []
        self._versions: list[str] = []

    # ── Phase 1: Detect ──

    def is_stale(self) -> bool:
        return self._flags.get(Flag.SELF_MODEL_STALE)

    def needs_regeneration(self) -> bool:
        if self.is_stale():
            return True
        if self._drift and self._drift.trending_up(n=3):
            return True
        return False

    # ── Phase 2: Trigger ──

    def trigger(self, reason: str = "auto") -> LoopResult:
        self._flags.set(Flag.SELF_MODEL_STALE, True)
        self._flags.set(Flag.AUDIT_OVERDUE, True)

        pre_drift = None
        if self._drift:
            pre_drift = self._drift.assess()

        self._store.log_regeneration(False, {
            "phase": "trigger", "reason": reason,
            "pre_drift": pre_drift.risk_score if pre_drift else None,
        })

        result = LoopResult(
            phase="trigger", success=True,
            details=f"Reason: {reason}, pre-drift: {pre_drift.risk_score if pre_drift else 'N/A'}",
        )
        self._history.append(result)
        return result

    # ── Phase 3: Regenerate (accepts AI-generated text) ──

    def regenerate(self, new_self_model: str) -> LoopResult:
        if not new_self_model or len(new_self_model.strip()) < 100:
            result = LoopResult(
                phase="regenerate", success=False,
                details="Self-model too short (<100 chars)")
            self._history.append(result)
            return result

        self._versions.append(new_self_model.strip())
        if len(self._versions) > self.MAX_VERSIONS:
            self._versions = self._versions[-self.MAX_VERSIONS:]

        try:
            path = os.path.expanduser(self.SELF_MODEL_PATH)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(new_self_model.strip() + "\n")
        except Exception as e:
            result = LoopResult(
                phase="regenerate", success=False,
                details=f"Write failed: {e}")
            self._history.append(result)
            return result

        result = LoopResult(
            phase="regenerate", success=True,
            details=f"Wrote {len(new_self_model)} chars, v{len(self._versions)}")
        self._history.append(result)
        return result

    # ── Phase 4: Validate (L1 structural + L3 consensus) ──

    def validate(self, self_model_text: Optional[str] = None) -> LoopResult:
        text = self_model_text or (self._versions[-1] if self._versions else "")
        if not text:
            result = LoopResult(phase="validate", success=False,
                               details="No text to validate")
            self._history.append(result)
            return result

        issues = []

        if self._gates and hasattr(self._gates, 'regeneration'):
            r = self._gates.regeneration.validate(text)
            if not r["valid"]:
                issues.append(f"L1 structural: {r['issues']}")
            else:
                issues.append(f"L1: score={r['score']}/100")

        if self._eval_field:
            try:
                candidate = {
                    "rule_id": "self-model-regeneration",
                    "title": "Self-Model Regeneration Quality",
                    "text": text[:2000],
                    "layer": "L3", "mechanizable": False,
                }
                evals = self._eval_field.evaluate_sync(candidate)
                consensus = self._eval_field.reach_consensus(
                    evals, "self-model-regeneration", "Self-Model Regeneration Quality")
                if consensus.passed:
                    issues.append(f"L3 consensus: {consensus.consensus:.2f} PASS")
                else:
                    issues.append(f"L3 consensus: {consensus.consensus:.2f} FAIL")
            except Exception as e:
                issues.append(f"L3 eval error: {e}")

        success = not any("FAIL" in i or "error" in i.lower() for i in issues)
        result = LoopResult(phase="validate", success=success,
                           details=" | ".join(issues))
        self._history.append(result)
        return result

    # ── Phase 5: Audit ──

    def audit(self) -> LoopResult:
        self._store.log_regeneration(True, {
            "phase": "audit",
            "versions": len(self._versions),
            "history": len(self._history),
        })

        post_drift = None
        if self._drift:
            post_drift = self._drift.assess()

        result = LoopResult(
            phase="audit", success=True,
            details=f"Post-drift: {post_drift.risk_score if post_drift else 'N/A'}")
        self._history.append(result)
        return result

    # ── Phase 6: Clear ──

    def clear(self) -> LoopResult:
        self._flags.clear(Flag.SELF_MODEL_STALE)
        self._flags.clear(Flag.AUDIT_OVERDUE)
        result = LoopResult(phase="clear", success=True,
                           details="Cleared SELF_MODEL_STALE, AUDIT_OVERDUE")
        self._history.append(result)
        return result

    # ── Full cycle ──

    def run_cycle(self, new_self_model: Optional[str] = None,
                  reason: str = "auto") -> list[LoopResult]:
        results: list[LoopResult] = []

        if not self.needs_regeneration():
            results.append(LoopResult(phase="detect", success=True,
                                      details="No regeneration needed"))
            return results
        results.append(LoopResult(phase="detect", success=True,
                                  details="Stale flag detected"))
        results.append(self.trigger(reason))

        if new_self_model:
            results.append(self.regenerate(new_self_model))
            results.append(self.validate(new_self_model))
            results.append(self.audit())
            results.append(self.clear())

        return results

    # ── Dry-run (for claim experiments) ──

    def dry_run(self) -> dict:
        return {
            "is_stale": self.is_stale(),
            "needs_regeneration": self.needs_regeneration(),
            "flag_burden": len(self._flags.active_flags()),
            "pre_drift": self._drift.assess().risk_score if self._drift else None,
            "constitution_coverage": (
                self._constitution.gate_coverage() if self._constitution else None),
            "neural_active": (
                self._neural_gate.active_count() if self._neural_gate else None),
            "versions": len(self._versions),
            "history": len(self._history),
        }

    def get_history(self) -> list[LoopResult]:
        return list(self._history)

    def get_versions(self) -> list[str]:
        return list(self._versions)

    def stats(self) -> dict:
        return {
            "cycles_completed": sum(
                1 for r in self._history if r.phase == "clear" and r.success),
            "cycles_failed": sum(1 for r in self._history if not r.success),
            "versions_kept": len(self._versions),
            "last_cycle": (self._history[-1].timestamp if self._history else None),
        }

    def build_prompt(self) -> str:
        lines = ["## Self-Referential Loop (Strange Loop)"]
        lines.append(f"- Cycles completed: {self.stats()['cycles_completed']}")
        lines.append(f"- Versions kept: {len(self._versions)}")
        lines.append(f"- Stale flag: {self.is_stale()}")
        if self._history:
            lines.append("- Recent phases:")
            for r in self._history[-5:]:
                status = "OK" if r.success else "FAIL"
                lines.append(f"  - [{status}] {r.phase}: {r.details[:80]}")
        return "\n".join(lines)
