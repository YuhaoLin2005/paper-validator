"""Flag types and manager for paper-validator state.

Imported by: layers/l1_gates.py, layers/strange_loop.py
No data files — pure enum + wrapper over StateStore.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from state.store import StateStore


class Flag(Enum):
    """System flags for lifecycle management."""
    SELF_MODEL_STALE = "self-model-stale"
    REGRESSION_DETECTED = "regression-detected"
    SESSION_DEGRADED = "session-degraded"
    TRIAL_INCOMPLETE = "trial-incomplete"
    AUDIT_OVERDUE = "audit-overdue"
    CONFIG_STALE = "config-stale"


class FlagManager:
    """Convenience wrapper around StateStore for flag operations."""

    def __init__(self, store: "StateStore"):
        self._store = store

    def set(self, flag: Flag, value: bool = True):
        self._store.set_flag(flag.value, value)

    def get(self, flag: Flag) -> bool:
        return self._store.get_flag(flag.value)

    def clear(self, flag: Flag):
        self._store.clear_flag(flag.value)

    def any_active(self, *flags: Flag) -> bool:
        return any(self.get(f) for f in flags)

    def all_clear(self, *flags: Flag) -> bool:
        return not self.any_active(*flags)

    def active_flags(self) -> list[Flag]:
        all_flags = self._store.list_flags()
        return [Flag(k) for k, v in all_flags.items()
                if v and k in Flag._value2member_map_]
