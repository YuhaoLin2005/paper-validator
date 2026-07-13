# paper-validator

> **A governance audit sub-agent.** Not an agent that does tasks — an agent that *measures whether governance rules actually change agent behavior.* It accompanies any LLM-powered agent and answers one question: "Are your rules working?"

Independent governance audit framework accompanying **"Agent Configuration Drift: A Five-Layer Governance Architecture"** (Lin Yuhao, 2026). Internalizes the 5-layer architecture as importable Python modules and independently validates 8 governance claims (distinct from but complementary to the paper's 12 experiments) with reproducible A/B tests against live LLM APIs.

```bash
pip install requests
python -m paper_validator claim --claim all --trials 30
# 8 claims, 85 API calls, 0 setup — all PARTIALLY_CONFIRMED or better
```

## What problem does this solve?

Most AI agents have governance rules (system prompts, safety guidelines, format requirements). But **nobody measures whether those rules actually work.** Are your rules changing the model's output, or are they just decorative text in a system prompt?

paper-validator gives you a **quantitative answer**: effect size, compliance rate differential, and a verdict (CONFIRMED / PARTIALLY_CONFIRMED / REJECTED) — per rule, per condition, with reproducible experiments.

## What this is (and isn't)

| This IS | This is NOT |
|---------|-------------|
| A governance audit layer you embed into any agent | A standalone task-completing agent |
| A measurement tool for rule effectiveness | A rule enforcement system |
| Importable Python modules (`from paper_validator.layers import ...`) | A black-box service |
| Paper-validated (8 claims, all PARTIALLY_CONFIRMED) | Theoretical / untested |

## Three ways to use it

### Path 1: CLI (any agent with a Bash tool)

```bash
# Single claim
python -m paper_validator claim --claim causal-swap --trials 30

# Full audit
python -m paper_validator claim --claim all --trials 30

# Health check
python -m paper_validator health
```

Works with any agent that can shell out — Claude Code, Codex, Gemini CLI, or a cron job.

### Path 2: MCP Tool (any MCP-compatible agent)

Add to your MCP client config (`claude_desktop_config.json` / `settings.json`):

```json
{
  "mcpServers": {
    "paper-validator": {
      "command": "python",
      "args": ["paper_validator/mcp_server.py"]
    }
  }
}
```

Three tools exposed:
- `validate_rule(claim, trials)` — measure a rule's causal effect
- `list_governance_claims` — list all 8 claim experiments
- `health_check` — layer status + drift risk score

The agent can invoke governance audit as a first-class tool call, not a shell escape.

### Path 3: Native Python import (your own agent)

```python
from paper_validator.layers.l1_gates import L1Gates
from paper_validator.layers.l4_drift_predictor import DriftPredictor
from paper_validator.claims.runner import run_claim

class MyAgent:
    def __init__(self):
        self.gates = L1Gates()
        self.drift = DriftPredictor()

    def before_action(self, action):
        """Governance gate: block unsafe actions before execution."""
        if not self.gates.health.run().ok:
            raise GovernanceViolation("Health check failed")

    def after_session(self):
        """Post-session: measure rule effectiveness, detect drift."""
        risk = self.drift.assess()
        if risk.risk_score > 50:
            self.flag_regeneration()
        # Validate that our rules still bind
        result = run_claim("causal-swap", n_trials=10)
        if result.effect_size < 0.3:
            self.alert("Rules are not measurably constraining output")
```

This is the real goal: governance as a **compiled-in system property**, not a bolt-on patch. No config files, no subprocess calls, no filesystem coupling between layers.

## Architecture

```
paper_validator/
├── main.py                     # CLI entry
├── mcp_server.py               # MCP stdio server (3 tools)
├── validate.bat                # Windows quick-launcher
├── engine/
│   └── api_client.py           # OpenAI-compatible API (stdlib urllib)
├── layers/
│   ├── l0_constitution.py      # Immutable rules + amendment process
│   ├── l1_gates.py             # HealthChecker, QualityGate, WriteGuard
│   ├── l2_neural_gate.py       # Logprob-differential probes
│   ├── l3_causal_encoding.py   # EvalField: 5-persona dual-pool consensus
│   ├── l4_drift_predictor.py   # 8-feature drift risk 0-100
│   └── strange_loop.py         # Self-referential regeneration cycle
├── claims/                     # 8 reproducible experiments
│   ├── l0_safety_prompt.py     # Claim 1: Constitution rules constrain output
│   ├── causal_swap.py          # Claim 2: Rule removal causes behavioral reversal
│   ├── logprob_probe_v3.py     # Claim 3: Logprob differentials measure fidelity
│   ├── dissociation.py         # Claim 4: L2/L3 measure distinct signals
│   ├── gateguard_off.py        # Claim 5: 3-tier rule gradient
│   ├── l1_visibility.py        # Claim 6: Mechanical gates produce differences
│   ├── prose_barrier.py        # Claim 7: Code-form rules > prose rules
│   └── cross_model.py          # Claim 8: Consistent governance patterns
├── config/defaults.py          # Model IDs, thresholds, constants
└── state/store.py              # Thread-safe JSON KV store
```

## The 8 Claims

| # | Claim | What it tests | Verdict |
|---|-------|---------------|---------|
| 1 | l0-safety-prompt | Rules measurably constrain outputs | PARTIALLY_CONFIRMED |
| 2 | causal-swap | Rule removal/reinsertion reverses behavior | PARTIALLY_CONFIRMED |
| 3 | logprob-probe-v3 | Logprob differentials measure constraint fidelity | PARTIALLY_CONFIRMED |
| 4 | dissociation | L2 neural gate and L3 causal encoding are distinct | PARTIALLY_CONFIRMED |
| 5 | gateguard-off | Rule removal degrades compliance (gradient) | PARTIALLY_CONFIRMED |
| 6 | l1-visibility | Mechanical gates produce measurable differences | PARTIALLY_CONFIRMED |
| 7 | prose-barrier | Code-form rules beat prose-form rules | PARTIALLY_CONFIRMED |
| 8 | cross-model | Governance effect generalizes | PARTIALLY_CONFIRMED |

**Key finding**: Format-enforcement tags (`[THINK]:`, `[VERIFY]:`) achieve **100% compliance** with rules vs. **0%** in control — a clean, replicable causal effect.

## Requirements

- Python 3.12+
- `pip install requests`
- DeepSeek API key (`DEEPSEEK_API_KEY`) or any OpenAI-compatible endpoint

## Quick Start

```bash
python -m paper_validator health
python -m paper_validator claim --list
python -m paper_validator claim --claim causal-swap --trials 30
python -m paper_validator claim --claim all --trials 30
```

## Related

- [**hermes-workspace**](https://github.com/YuhaoLin2005/hermes-workspace) — Paper, experiments, and data for the 5-layer governance architecture
- [**DEV.to**](https://dev.to/yuhaolin2005) — 6 technical articles detailing each layer and experiment
- [**掘金 (Juejin)**](https://juejin.cn/user/4250072430682412) — Chinese-language deep dives

## License

MIT
