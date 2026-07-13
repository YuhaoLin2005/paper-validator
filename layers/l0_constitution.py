"""L0 — Constitution: immutable governance rules with amendment lifecycle.

Imported by: layers/strange_loop.py, claims/base.py
Data: reads DEFAULT_RULES from config/defaults.py; writes via StateStore canonization_log
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
import time

if TYPE_CHECKING:
    from state.store import StateStore

from config.defaults import DEFAULT_RULES


@dataclass
class Rule:
    id: str
    title: str
    text: str
    layer: str = "L1"
    mechanizable: bool = False
    status: str = "active"
    eval_hash: str = ""
    canonical_date: str = ""
    amended_from: Optional[str] = None
    amended_date: Optional[str] = None


class Constitution:
    """Immutable governance rules with amendment lifecycle.

    Amendment requires: L3 eval field (5 personas) → 3/5 consensus → 24h cooling.
    """

    def __init__(self, state_store: Optional["StateStore"] = None):
        self._store = state_store
        self._rules: dict[str, Rule] = {}
        self._proposals: dict[str, Rule] = {}
        self._load_defaults()

    def _load_defaults(self):
        for r in DEFAULT_RULES:
            rule = Rule(
                id=r["id"], title=r["title"], text=r["text"],
                layer=r.get("layer", "L1"),
                mechanizable=r.get("mechanizable", False),
            )
            self._rules[rule.id] = rule

    @property
    def rules(self) -> dict[str, Rule]:
        return dict(self._rules)

    def get(self, rule_id: str) -> Optional[Rule]:
        return self._rules.get(rule_id)

    def active_rules(self) -> list[Rule]:
        return [r for r in self._rules.values() if r.status == "active"]

    def rules_by_layer(self, layer: str) -> list[Rule]:
        return [r for r in self._rules.values()
                if r.layer == layer and r.status == "active"]

    def mechanizable_count(self) -> int:
        return sum(1 for r in self.active_rules() if r.mechanizable)

    def gate_coverage(self) -> float:
        active = self.active_rules()
        return self.mechanizable_count() / len(active) if active else 0.0

    def propose(self, rule_id: str, title: str, text: str,
                layer: str = "L1", mechanizable: bool = False) -> Rule:
        rule = Rule(id=rule_id, title=title, text=text,
                    layer=layer, mechanizable=mechanizable, status="proposed")
        self._proposals[rule_id] = rule
        if self._store:
            self._store.log_audit("rule_proposed", {
                "rule_id": rule_id, "title": title, "layer": layer})
        return rule

    def debate(self, rule_id: str, consensus: float):
        if rule_id not in self._proposals:
            raise ValueError(f"No proposal: {rule_id}")
        rule = self._proposals[rule_id]
        rule.status = "debate"
        rule.eval_hash = f"consensus={consensus:.2f}"

    def canonize(self, rule_id: str) -> Rule:
        if rule_id not in self._proposals:
            raise ValueError(f"No proposal: {rule_id}")
        rule = self._proposals.pop(rule_id)
        consensus = float(rule.eval_hash.split("=")[1]) if rule.eval_hash else 0.0
        if consensus < 0.6:
            raise ValueError(f"Consensus {consensus:.2f} < 0.6 for {rule_id}")
        rule.status = "active"
        rule.canonical_date = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        if rule.id in self._rules:
            old = self._rules[rule.id]
            old.status = "superseded"
            rule.amended_from = old.id
            rule.amended_date = rule.canonical_date
        self._rules[rule.id] = rule
        if self._store:
            self._store.log_canonization(rule_id, consensus, "canonized")
        return rule

    def build_prompt(self) -> str:
        lines = ["## Governance Rules (L0 Constitution)"]
        for rule in self.active_rules():
            tag = f"[{rule.layer}{' M' if rule.mechanizable else ''}]"
            lines.append(f"- {tag} {rule.id}: {rule.title} — {rule.text}")
        return "\n".join(lines)

    def stats(self) -> dict:
        active = self.active_rules()
        return {
            "total_rules": len(self._rules),
            "active_rules": len(active),
            "proposed": len(self._proposals),
            "mechanizable": self.mechanizable_count(),
            "gate_coverage": self.gate_coverage(),
            "layers": {l: len(self.rules_by_layer(l))
                       for l in ["L0", "L1", "L2", "L3", "L4"]},
        }
