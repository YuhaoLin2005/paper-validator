"""JSON-backed key-value store with atomic writes.

Imported by: main.py, layers/l1_gates.py, layers/strange_loop.py, claims/runner.py
Data: .paper-validator-state/state.json (schema in state/__init__.py)
"""

from __future__ import annotations

import json, os, time, threading
from pathlib import Path
from typing import Optional


class StateStore:
    """Thread-safe JSON-backed key-value store with atomic writes."""

    def __init__(self, state_dir: str = ".paper-validator-state"):
        self._dir = Path(state_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "state.json"
        self._lock = threading.Lock()
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._file.exists():
            try:
                with open(self._file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return self._default()

    def _default(self) -> dict:
        return {
            "flags": {},
            "counters": {"sessions": 0, "compactions": 0, "trials": 0},
            "audit_log": [],
            "regeneration_log": [],
            "canonization_log": [],
            "meta": {"created": self._now(), "updated": self._now(), "version": "0.1.0"},
        }

    def _save(self):
        self._data["meta"]["updated"] = self._now()
        tmp = self._file.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, self._file)

    @staticmethod
    def _now() -> str:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def set_flag(self, name: str, value: bool = True):
        with self._lock:
            self._data["flags"][name] = value; self._save()

    def get_flag(self, name: str) -> bool:
        with self._lock:
            return bool(self._data["flags"].get(name, False))

    def clear_flag(self, name: str):
        with self._lock:
            self._data["flags"].pop(name, None); self._save()

    def list_flags(self) -> dict[str, bool]:
        with self._lock:
            return dict(self._data["flags"])

    def increment(self, name: str, amount: int = 1) -> int:
        with self._lock:
            v = self._data["counters"].get(name, 0) + amount
            self._data["counters"][name] = v; self._save()
            return v

    def get_counter(self, name: str) -> int:
        with self._lock:
            return self._data["counters"].get(name, 0)

    def log_audit(self, event: str, detail: Optional[dict] = None):
        with self._lock:
            self._data["audit_log"].append({
                "ts": self._now(), "event": event, "detail": detail or {},
            })
            if len(self._data["audit_log"]) > 1000:
                self._data["audit_log"] = self._data["audit_log"][-1000:]
            self._save()

    def get_audit_log(self, limit: int = 50) -> list[dict]:
        with self._lock:
            return list(self._data["audit_log"][-limit:])

    def log_regeneration(self, valid: bool, detail: Optional[dict] = None):
        with self._lock:
            self._data["regeneration_log"].append({
                "ts": self._now(), "valid": valid, "detail": detail or {},
            })
            if len(self._data["regeneration_log"]) > 100:
                self._data["regeneration_log"] = self._data["regeneration_log"][-100:]
            self._save()

    def log_canonization(self, rule_id: str, consensus: float, outcome: str):
        with self._lock:
            self._data["canonization_log"].append({
                "ts": self._now(), "rule_id": rule_id,
                "consensus": consensus, "outcome": outcome,
            })
            self._save()

    def save_trial(self, claim_name: str, trial_index: int, data: dict):
        with self._lock:
            key = f"trial_{claim_name}_{trial_index}"
            self._data.setdefault("trials", {})[key] = {
                "ts": self._now(), "claim": claim_name,
                "trial": trial_index, "data": data,
            }
            self._save()

    def get_trials(self, claim_name: str) -> list[dict]:
        with self._lock:
            prefix = f"trial_{claim_name}_"
            return [v for k, v in self._data.get("trials", {}).items()
                    if k.startswith(prefix)]

    def dump(self) -> dict:
        with self._lock:
            return dict(self._data)

    def snapshot_path(self) -> Path:
        return self._file
