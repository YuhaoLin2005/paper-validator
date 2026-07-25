# paper-validator

**Independent governance audit tool for AI agent rule-following behavior.**

`paper-validator` validates 10 claims from the [hermes-workspace](https://github.com/YuhaoLin2005/hermes-workspace) paper (*The Prose Barrier*) through reproducible A/B tests against live LLM APIs. It is the mechanical verification layer — a CLI tool that runs the same experiments described in the paper, producing the same results, with zero human scoring required.

---

## What This Validates

The core question: **Do AI agents follow their own rules across sessions?**

The paper identifies a structural constraint — the **Prose Barrier** — where an LLM's self-verification shares the same decoder as its generation, making independent self-audit architecturally impossible. The solution is a **5-layer governance architecture** organized by position relative to this barrier:

| Layer | Position | Mechanism | Evidence |
|-------|----------|-----------|----------|
| **L0** Psychological Safety | Pre-barrier | Reframe uncertainty admission | Accuracy preserved (+0.01), r=+0.949 |
| **L1** Mechanical Gate | Outside | Filesystem checks (regex, mtime, exit codes) | 55.9% → 0.7% violation rate, 19/19 tests |
| **L2** Neural Gate | Inside | Logprob differential probes | d_z=+0.578, BF₁₀=282k (n=40 probes) |
| **L3** Causal Encoding | Through | Syllogistic format → attention routing | Non-monotonic constraint gradient |
| **L4** Drift Prediction | Outside (time) | 8-feature trend detection | 12 features deployed |

---

## Quick Start

```bash
git clone https://github.com/YuhaoLin2005/paper-validator.git
cd paper-validator
export DEEPSEEK_API_KEY=sk-your-key-here
python -m paper_validator claim --claim all --trials 30
```

**Requirements:** Python 3.10+, no dependencies beyond stdlib.

---

## The 10 Claims

| # | Claim | Key Metric |
|---|-------|------------|
| 1 | **Prose Barrier** — Self-verification structurally unreliable | 55.9% → 0.7% |
| 2 | **L1 Mechanical Gate** — Near-perfect compliance | 19/19 tests |
| 3 | **L2 Logprob Probes** — Detect constraint penetration | d_z=+0.578, BF=282k |
| 4 | **L2/L3 Dissociation** — Format ≠ behavior | L2 d=+0.578 vs L3 Δ=-0.024 |
| 5 | **Constraint Gradient** — Non-monotonic 3-phase | C₁:0.596 > C₀:0.315 > C₃:0.297 > C₂:0.091 |
| 6 | **Format Context-Fragility** — Collapses under multi-task | d_z 0.58→0.19 |
| 7 | **Format-L1 Synergy** — Format amplifies, not compensates | d_z=0.71 (L1-visible) vs 0.40 |
| 8 | **P1-2 Format×Gate** — Pre-registered H1 NOT_CONFIRMED | Prose+Gate best reasoning |
| 9 | **Cross-Model Gateability** — Structure × Capacity | 2D space, not single dimension |
| 10 | **GateGuard Ceiling** — 99.3% zero-violation ceiling | 149/150 |

---

## Architecture

```
paper-validator/
├── layers/              # L0-L4 architecture
│   ├── l0_constitution.py       # Immutable rules + amendment lifecycle
│   ├── l1_gates.py              # HealthChecker + QualityGate + WriteGuard
│   ├── l2_neural_gate.py        # Logprob differential probes
│   ├── l3_causal_encoding.py    # Syllogistic format + 5-persona eval
│   ├── l4_drift_predictor.py    # 8-feature 0-100 risk scoring
│   ├── mechanizability_scanner.py # Deterministic rule classifier
│   └── strange_loop.py          # Self-referential regeneration
├── claims/              # 10 reproducible claim experiments
├── engine/              # Minimal agent loop (urllib, stdlib)
├── dpo_training/        # QLoRA behavioral internalization
├── results/             # JSON experiment output
└── experiments/         # Standalone experiment scripts
```

---

## Three Deployment Modes

**CLI:** `python -m paper_validator claim --claim all --trials 30`

**MCP Server:** JSON-RPC 2.0 over stdin/stdout — `validate_rule`, `list_governance_claims`, `health_check`

**Python Import:** `from paper_validator.claims.runner import ClaimRunner`

---

## Key Findings

1. **Mechanical gates work.** L1 reduces violations 55.9% → 0.7%. Code+Gate = perfect mechanical (5.0) but worst reasoning (4.2).
2. **Format changes representations, not behavior.** Syllogistic: d_z=+0.578 at logprob level but IMP≈SYL behaviorally. The "Prose Barrier of Measurement."
3. **Gateability is model-dependent.** DS Flash = 100% hollow compliance. Qwen = 0% on uncertainty tasks.
4. **Format effects are context-fragile.** Single-scene d=0.58 collapses to d=0.19 under multi-scene load (67% reduction).
5. **DPO amplifies existing preferences.** Qwen2.5-1.5B DPO reduced per-token NLL ~5-10% on causal responses.

---

## Honest Limitations

Single-rater behavioral scoring (κ = −0.14). Single-model logprob evidence. Small samples (most n<50). No pre-registration (except P1-2). Single device. No supervisor, no funding. Undergraduate independent work.

**Most critical next step:** independent blind scoring by 2+ raters achieving κ > 0.7.

---

## Repository Map

| Repo | Purpose |
|------|---------|
| [hermes-workspace](https://github.com/YuhaoLin2005/hermes-workspace) | Paper + experiments + content pipeline |
| [paper-validator](https://github.com/YuhaoLin2005/paper-validator) | Independent validation CLI (this repo) |
| [imprint](https://github.com/YuhaoLin2005/imprint) | Cross-session identity persistence |
| [gategrow](https://github.com/gategrow) | Industrial-grade quality toolkit |

*Author: Yuhao Lin, FAFU 2023. License: MIT.*
