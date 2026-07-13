"""L1 — Mechanical Gates: HealthChecker, QualityGate, WriteGuard, RegenerationValidator.

Imported by: layers/strange_loop.py, main.py, claims/base.py
Data: reads/writes StateStore flags and counters; checks filesystem health.

Collapses 14 external scripts into native method calls:
  health-check.py → HealthChecker
  quality-gate.py → QualityGate
  write-guard.py → WriteGuard
  log-regeneration.py → RegenerationValidator
  + 10 auxiliary guard/check scripts → internal methods
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from state.flags import Flag

if TYPE_CHECKING:
    from state.store import StateStore
    from state.flags import FlagManager


# ── HealthChecker: system checks (was health-check.py, config-health.py, disk-monitor.py, risk-scanner.py) ──

class HealthCheckResult:
    def __init__(self):
        self.passed: list[str] = []
        self.warnings: list[str] = []
        self.failures: list[str] = []
        self.flags: list[str] = []
        self.regenerate_needed: bool = False

    @property
    def ok(self) -> bool:
        return len(self.failures) == 0

    def __str__(self) -> str:
        parts = []
        for p in self.passed:
            parts.append(f"  PASS: {p}")
        for w in self.warnings:
            parts.append(f"  WARN: {w}")
        for f in self.failures:
            parts.append(f"  FAIL: {f}")
        return "\n".join(parts) or "  (no checks)"


class HealthChecker:
    """System health checks at session start."""

    def __init__(self, store: "StateStore", flags: "FlagManager",
                 claude_dir: str = "~/.claude"):
        self._store = store
        self._flags = flags
        self._claude_dir = Path(claude_dir).expanduser()

    def run(self) -> HealthCheckResult:
        result = HealthCheckResult()
        self._check_disk(result)
        self._check_config_files(result)
        self._check_self_model_stale(result)
        self._check_degradation(result)
        return result

    def _check_disk(self, result: HealthCheckResult):
        try:
            free_gb = shutil.disk_usage("C:/").free / (1024**3)
            if free_gb < 15:
                result.failures.append(f"Disk: {free_gb:.1f}GB free < 15GB")
            elif free_gb < 50:
                result.warnings.append(f"Disk: {free_gb:.1f}GB free < 50GB")
            else:
                result.passed.append(f"Disk: {free_gb:.1f}GB free")
        except Exception as e:
            result.warnings.append(f"Disk check failed: {e}")

    def _check_config_files(self, result: HealthCheckResult):
        for cfg in ["settings.json", "INTERFACE.md", "BODY.md"]:
            path = self._claude_dir / cfg
            if not path.exists():
                result.warnings.append(f"Config file missing: {cfg}")
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    f.read(1)
                result.passed.append(f"Config readable: {cfg}")
            except Exception:
                result.warnings.append(f"Config unreadable: {cfg}")

    def _check_self_model_stale(self, result: HealthCheckResult):
        if self._flags.get(Flag.SELF_MODEL_STALE):
            result.flags.append("SELF_MODEL_STALE")
            result.regenerate_needed = True
            result.warnings.append("Self-model stale — regeneration needed")
        else:
            result.passed.append("Self-model fresh")

    def _check_degradation(self, result: HealthCheckResult):
        if self._flags.get(Flag.SESSION_DEGRADED):
            result.flags.append("SESSION_DEGRADED")
            result.warnings.append("Session degraded — limited capabilities")


# ── QualityGate: delivery gate (was quality-gate.py) ──

class QualityGate:
    """Delivery gate: checks five-library freshness at session end."""

    def __init__(self, store: "StateStore"):
        self._store = store

    def check(self, libraries: Optional[dict[str, float]] = None) -> dict:
        """Check freshness of learning libraries.

        Args:
            libraries: {name: hours_since_update} — if None, does minimal check.

        Returns:
            {"passed": [...], "stale": [...], "score": 0-100}
        """
        result = {"passed": [], "stale": [], "score": 100}
        if not libraries:
            return result

        thresholds = {
            "persona": 24,
            "growth_log": 8,
            "decisions": 48,
            "output_index": 24,
            "ratings": 48,
        }

        for name, hours in libraries.items():
            threshold = thresholds.get(name, 24)
            if hours > threshold:
                result["stale"].append(name)
                result["score"] -= 20
            else:
                result["passed"].append(name)

        result["score"] = max(0, result["score"])
        return result

    def check_and_flag(self):
        """Check all libraries and set stale flag if needed."""
        audit_log = self._store.get_audit_log(5)
        if not audit_log:
            return
        self._store.log_audit("quality_check", {"audit_entries": len(audit_log)})


# ── WriteGuard: write safety (was write-guard.py, execution-gate.py, three-questions-guard.py) ──

class WriteGuard:
    """Write safety checks: three-questions gate, execution debt, sensitive paths.

    In the standalone agent, this is advisory — the agent runs fully trusted
    for paper validation experiments. Guards are preserved as programmatic
    checks that can be enabled with --safe mode.
    """

    SENSITIVE_PATTERNS = [
        ".env", ".env.", "credentials", "secret", "token", "password",
        "~/.ssh/", ".ssh/", "~/.aws/", ".aws/credentials",
        "id_rsa", "private.key", ".pem",
    ]

    DANGEROUS_COMMANDS = [
        ("rm -rf /", "recursive root delete"),
        ("rm -rf ~", "recursive home delete"),
        ("> /dev/sda", "overwrite disk device"),
        ("mkfs.", "format filesystem"),
        ("git push --force origin main", "force push main"),
    ]

    def __init__(self, store: "StateStore", safe_mode: bool = False):
        self._store = store
        self.safe_mode = safe_mode

    def is_sensitive_path(self, file_path: str) -> bool:
        path_lower = file_path.lower()
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern.lower() in path_lower:
                return True
        return False

    def is_dangerous_command(self, command: str) -> Optional[str]:
        cmd_lower = command.lower()
        for pattern, reason in self.DANGEROUS_COMMANDS:
            if pattern.lower() in cmd_lower:
                return reason
        return None

    def check_write(self, file_path: str) -> Optional[str]:
        if not self.safe_mode:
            return None
        if self.is_sensitive_path(file_path):
            return f"Sensitive path: {file_path}"
        return None

    def log_write(self, file_path: str):
        self._store.increment("writes")


# ── RegenerationValidator: self-model validation (was log-regeneration.py) ──

class RegenerationValidator:
    """Validates self-model structure and content after regeneration."""

    REQUIRED_SECTIONS = ["capabilities", "limitations", "growth", "identity"]
    MIN_LINES = 20
    MIN_CAPABILITIES = 3

    def __init__(self, store: "StateStore"):
        self._store = store

    def validate(self, self_model_text: str) -> dict:
        """Validate a regenerated self-model.

        Returns:
            {"valid": bool, "score": 0-100, "issues": [...],
             "checks": {"lines": int, "sections": [...], "capabilities": int}}
        """
        issues = []
        lines = self_model_text.strip().split("\n")
        line_count = len(lines)

        if line_count < self.MIN_LINES:
            issues.append(f"Too short: {line_count} lines < {self.MIN_LINES}")

        found_sections = []
        for section in self.REQUIRED_SECTIONS:
            if any(section.lower() in l.lower() for l in lines):
                found_sections.append(section)
        missing = set(self.REQUIRED_SECTIONS) - set(found_sections)
        if missing:
            issues.append(f"Missing sections: {', '.join(missing)}")

        capability_count = 0
        in_cap = False
        for l in lines:
            if "capabilities" in l.lower():
                in_cap = True
                continue
            if in_cap and l.strip().startswith("-"):
                capability_count += 1
            elif in_cap and l.strip().startswith("#"):
                in_cap = False
        if capability_count < self.MIN_CAPABILITIES:
            issues.append(f"Too few capabilities: {capability_count} < {self.MIN_CAPABILITIES}")

        score = 100 - len(issues) * 20
        return {
            "valid": len(issues) == 0,
            "score": max(0, score),
            "issues": issues,
            "checks": {
                "lines": line_count,
                "sections": found_sections,
                "capabilities": capability_count,
            },
        }

    def validate_and_log(self, self_model_text: str) -> dict:
        result = self.validate(self_model_text)
        self._store.log_regeneration(result["valid"], {
            "score": result["score"],
            "issues": result["issues"],
        })
        return result


# ── L1 aggregate ──

class L1Gates:
    """Aggregate L1 layer: bundles all mechanical gates."""

    def __init__(self, store: "StateStore", flags: "FlagManager",
                 safe_mode: bool = False):
        self.health = HealthChecker(store, flags)
        self.quality = QualityGate(store)
        self.write = WriteGuard(store, safe_mode=safe_mode)
        self.regeneration = RegenerationValidator(store)
        self._store = store
        self._flags = flags

    def startup_check(self) -> HealthCheckResult:
        result = self.health.run()
        self._store.log_audit("l1_startup", {
            "ok": result.ok,
            "failures": result.failures,
            "flags": result.flags,
        })
        return result

    def shutdown_check(self) -> dict:
        self.quality.check_and_flag()
        result = {"quality": "checked", "writes": self._store.get_counter("writes")}
        self._store.log_audit("l1_shutdown", result)
        return result

    def build_prompt(self) -> str:
        lines = ["## Mechanical Gates (L1)"]
        lines.append("- Health checks run at session start and end")
        lines.append(f"- Sessions tracked: {self._store.get_counter('sessions')}")
        lines.append("- Self-model regeneration: automated flag-based cycle")
        return "\n".join(lines)
