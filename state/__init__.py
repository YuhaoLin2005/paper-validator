"""State persistence layer for paper-validator.

Imported by: main.py, layers/*.py, claims/runner.py
Data: state/store.py writes JSON to state_dir/.paper-validator-state/

State file schema:
{
  "flags": {"self-model-stale": true/false, ...},
  "counters": {"sessions": N, "compactions": N, "trials": N},
  "audit_log": [{"ts": "ISO8601", "event": "...", "detail": {...}}, ...],
  "regeneration_log": [{"ts": "ISO8601", "valid": true/false, ...}, ...]
}
"""
