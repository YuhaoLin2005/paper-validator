# paper-validator

Reproducible validation harness for the paper **"Agent Configuration Drift: A Five-Layer Governance Architecture"** (Lin Yuhao, 2026).

**Importers:** None (standalone CLI tool). **Callers:** User via `python -m paper_validator`. **Schema:** TrialResult, ClaimReport, ClaimReport (see claims/base.py).
**User instruction:** Push harness to GitHub as standalone repo, then update email to professor.

## What this is

A minimal, self-contained Python agent harness that internalizes the 5-layer governance architecture (L0 → L4) as native source code and validates all 8 paper claims with reproducible experiments against live LLM APIs.

```bash
pip install requests
python -m paper_validator claim --claim all --trials 30
```

## Architecture

```
paper_validator/
├── main.py                     # CLI: --claim, --trials, --interactive, --health
├── engine/                     # API client (DeepSeek/OpenAI-compatible)
├── layers/                     # 5-layer governance architecture
│   ├── l0_constitution.py      # Immutable rules + amendment
│   ├── l1_gates.py             # HealthChecker, QualityGate, WriteGuard
│   ├── l2_neural_gate.py       # Logprob-differential probes
│   ├── l3_causal_encoding.py   # EvalField (5-persona dual-pool consensus)
│   ├── l4_drift_predictor.py   # 8-feature drift risk scoring
│   └── strange_loop.py         # Self-referential regeneration cycle
├── claims/                     # 8 reproducible paper claim experiments
├── config/                     # Model IDs, thresholds, defaults
└── state/                      # JSON-backed KV store + flags
```

## Requirements

- Python 3.12+
- `requests`
- DeepSeek API key (`DEEPSEEK_API_KEY`) or any OpenAI-compatible endpoint

## Quick Start

```bash
python -m paper_validator health          # Health check
python -m paper_validator claim --list    # List all claims
python -m paper_validator claim --claim causal-swap --trials 30
python -m paper_validator claim --claim all --trials 30
```

## License

MIT
