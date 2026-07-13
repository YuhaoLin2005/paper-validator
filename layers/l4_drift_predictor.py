"""L4 — Drift Predictor: feature collection + risk scoring for agent degradation.

Imported by: layers/strange_loop.py, main.py
Data: reads state counters/flags from StateStore; no API calls.

Collapses drift_predictor.py and behavioral baseline analysis.
Uses DEFAULT_DRIFT_FEATURE_WEIGHTS from config/defaults.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from state.store import StateStore
    from state.flags import FlagManager

from config.defaults import DRIFT_FEATURE_WEIGHTS
from state.flags import Flag


@dataclass
class DriftAssessment:
    """Result of a drift risk assessment."""
    risk_score: float            # 0-100
    risk_level: str              # "low", "medium", "high", "critical"
    features: dict[str, float]   # feature_name → normalized value (0-100)
    weighted_score: float        # Σ(weight_i * feature_i)
    timestamp: str
    recommendation: str


class DriftPredictor:
    """Collects behavioral features and computes drift risk.

    Features track agent health across dimensions:
      - session_age: how long since last fresh start (hours)
      - error_rate: tool failures per session
      - compaction_frequency: compactions per session
      - rule_idleness: inactive rules / total rules
      - flag_burden: active flags / total flags
      - self_model_staleness: days since last self-model update
      - response_length_trend: placeholder
      - tool_diversity: placeholder

    Risk = Σ(weight_i × feature_i), normalized to 0-100.
    """

    LOW_THRESHOLD = 25
    MEDIUM_THRESHOLD = 50
    HIGH_THRESHOLD = 75

    def __init__(self, store: "StateStore", flags: "FlagManager",
                 feature_weights: Optional[dict[str, float]] = None):
        self._store = store
        self._flags = flags
        self._weights = feature_weights or DRIFT_FEATURE_WEIGHTS
        self._history: list[DriftAssessment] = []

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def collect_features(self) -> dict[str, float]:
        """Collect current feature values (raw, unweighted)."""
        sessions = max(1, self._store.get_counter("sessions"))
        compactions = self._store.get_counter("compactions")

        features: dict[str, float] = {}

        # Session age
        audit_log = self._store.get_audit_log(1)
        if audit_log:
            try:
                first_ts = audit_log[0].get("at", "")
                if first_ts:
                    from datetime import datetime
                    dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                    features["session_age"] = (time.time() - dt.timestamp()) / 3600
                else:
                    features["session_age"] = 0.0
            except Exception:
                features["session_age"] = 0.0
        else:
            features["session_age"] = 0.0

        # Error rate: proxy via flag count
        active = len(self._flags.active_flags())
        total_flags = len(Flag)
        features["error_rate"] = active / max(1, total_flags) * 100

        # Compaction frequency
        features["compaction_frequency"] = compactions / sessions

        # Rule idleness (set by strange_loop later)
        features["rule_idleness"] = 0.0

        # Flag burden
        features["flag_burden"] = float(active)

        # Self-model staleness
        features["self_model_staleness"] = 7.0 if self._flags.get(Flag.SELF_MODEL_STALE) else 0.0

        # Placeholders
        features["response_length_trend"] = 0.0
        features["tool_diversity"] = 0.0

        return features

    def _normalize(self, features: dict[str, float]) -> dict[str, float]:
        """Normalize raw features to 0-100 scale."""
        norms: dict[str, float] = {}
        norms["session_age"] = min(100, features.get("session_age", 0) / 48 * 100)
        norms["error_rate"] = min(100, features.get("error_rate", 0))
        norms["compaction_frequency"] = min(100, features.get("compaction_frequency", 0) / 5 * 100)
        norms["rule_idleness"] = min(100, features.get("rule_idleness", 0) * 100)
        norms["flag_burden"] = min(100, features.get("flag_burden", 0) / 6 * 100)
        norms["self_model_staleness"] = min(100, features.get("self_model_staleness", 0) / 30 * 100)
        norms["response_length_trend"] = min(100, features.get("response_length_trend", 0))
        norms["tool_diversity"] = max(0, 100 - features.get("tool_diversity", 0))
        return norms

    def assess(self) -> DriftAssessment:
        """Collect features and compute drift risk score."""
        raw = self.collect_features()
        norms = self._normalize(raw)

        weighted_score = 0.0
        for feature, weight in self._weights.items():
            if feature in norms:
                weighted_score += weight * norms[feature]

        total_weight = sum(self._weights.values())
        risk_score = (weighted_score / total_weight) if total_weight > 0 else 0.0

        if risk_score >= self.HIGH_THRESHOLD:
            level = "critical"
            rec = "Immediate regeneration required. Consider session restart."
        elif risk_score >= self.MEDIUM_THRESHOLD:
            level = "high"
            rec = "Schedule regeneration soon. Review active flags."
        elif risk_score >= self.LOW_THRESHOLD:
            level = "medium"
            rec = "Monitor. No immediate action needed."
        else:
            level = "low"
            rec = "System healthy. No action needed."

        assessment = DriftAssessment(
            risk_score=round(risk_score, 1),
            risk_level=level,
            features=norms,
            weighted_score=round(weighted_score, 2),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            recommendation=rec,
        )

        self._history.append(assessment)
        if len(self._history) > 100:
            self._history = self._history[-50:]

        self._store.log_audit("l4_drift", {
            "risk_score": assessment.risk_score,
            "risk_level": assessment.risk_level,
        })

        return assessment

    def trend(self, n: int = 5) -> list[float]:
        return [a.risk_score for a in self._history[-n:]]

    def trending_up(self, n: int = 5) -> bool:
        scores = self.trend(n)
        if len(scores) < 3:
            return False
        last3 = scores[-3:]
        return last3[0] < last3[1] < last3[2]

    def build_prompt(self) -> str:
        lines = ["## Drift Prediction (L4)"]
        lines.append(f"- Features tracked: {len(self._weights)}")
        lines.append(f"- Risk thresholds: LOW<{self.LOW_THRESHOLD} "
                     f"MED<{self.MEDIUM_THRESHOLD} HIGH<{self.HIGH_THRESHOLD}")
        if self._history:
            latest = self._history[-1]
            lines.append(f"- Latest: {latest.risk_score:.1f}/100 ({latest.risk_level})")
            lines.append(f"- Trend (last 5): {self.trend(5)}")
            if self.trending_up():
                lines.append("- ⚠ WARNING: Risk trending upward")
        else:
            lines.append("- No assessments yet — run assess()")
        return "\n".join(lines)
