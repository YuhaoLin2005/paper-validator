# Cross-Type Bidirectional Verification (CTBV): A Dual-Layer Architecture for Agent Output Integrity

> **Author**: Yuhao Lin (林宇浩) — linyuhao2005@gmail.com
> **Date**: 2026-07-28
> **Status**: Technical Report · Pre-submission · 10 claims, 16 experiments, ~2,500 API calls
> **Repository**: [github.com/YuhaoLin2005/paper-validator](https://github.com/YuhaoLin2005/paper-validator)
> **Companion**: [github.com/YuhaoLin2005/hermes-workspace](https://github.com/YuhaoLin2005/hermes-workspace)

---

## Abstract

Large language model (LLM) agents suffer from a structural reliability problem: they cannot independently verify their own outputs because generation and verification share the same autoregressive decoder — `P(token | context; θ)`. We call this the **Prose Barrier**. It is not a prompt engineering problem or a model quality problem; it is a structural constraint of the autoregressive paradigm itself.

We propose **Cross-Type Bidirectional Verification (CTBV)**, a dual-layer architecture that exploits the orthogonal blind spots of mechanical gates (filesystem checks, regex scoring — deterministic, content-blind) and semantic gates (logprob probes, format encoding — content-aware, interpretation-dependent). Because these two gate types fail on **different** output classes, their joint false-negative rate approaches zero under mild independence assumptions — a result we formalize mathematically.

We validate CTBV across a five-layer configuration integrity system (L0: Psychological Safety → L1: Mechanical Gate → L2: Neural Gate → L3: Causal Encoding → L4: Drift Prediction) with **16 experiments** spanning **2,500+ API calls** and **1,100+ formal trials** across 3 LLM architectures. Key results: (1) mechanical gates alone reduce configuration violations from 55.9% to 0.7%; (2) logprob differentials reliably detect constraint penetration without interpreting content (d=+0.578, BF=282k); (3) format effects operate at the token level (L2) but do not translate to behavioral compliance (L3), confirming the architectural necessity of multi-layer verification; (4) residual violations cluster exclusively on dimensions where mechanical gates cannot instrument — semantic quality, reasoning depth, epistemic stance.

---

## 1. The Prose Barrier

### 1.1 Formal Definition

Let an LLM agent generate output `y = argmax_{y'} P(y' | x; θ)` where `x` is the input context and `θ` are the frozen model parameters. The agent also produces a self-verification judgment `v ∈ {compliant, non_compliant}` by sampling from the same distribution: `P(v | y, x; θ)`. Because both `y` and `v` are drawn from the same decoder with the same parameters `θ`, there exists no structural mechanism that prevents `P(v = compliant | y = non_compliant)` from being arbitrarily high.

**Prose Barrier Theorem (informal)**: For any autoregressive model `M` with parameters `θ`, self-verification `v` of output `y` cannot achieve guaranteed reliability because:
```
P(v = compliant | y = non_compliant) = P(v = compliant, y = non_compliant | x; θ) / P(y = non_compliant | x; θ)
```
Both numerator and denominator are products of the same `θ`-parameterized token probabilities. No external ground truth enters the verification loop.

**Corollary**: The reliability upper bound of self-verification is `max(P(y = compliant | x; θ))`, which is the model's base compliance rate — approximately 44.4% in our baseline condition (Claim 1).

### 1.2 Empirical Evidence

In a 34-session retrospective analysis of a live LLM agent deployed with n=5 configuration rules:

| Condition | Violation Rate | Detection Method |
|-----------|:---:|------|
| No mechanical gate | **55.9%** | Agent self-report (prose) |
| Mechanical gate (L1 only) | **0.7%** | Filesystem checks + regex scoring |
| GateGuard + three-questions-guard | **0.47%** | Full L1 stack |

The 55.9% → 0.7% reduction (a 98.7% relative decrease) is achieved by a pure **mechanical** gate — no NL understanding, no LLM judge, no semantic analysis. The gate checks file modification times, exit codes, and regex patterns. It cannot read prose; it doesn't need to.

### 1.3 Why Mechanical Gates Alone Are Insufficient

The 0.7% residual is not random noise. In P1-1 (200 trials, 5 task types, systematic cluster analysis):

| Task Type | Mechanizable? | Compliance | Semantic Violations |
|-----------|:---:|:---:|:---:|
| T1: Format-tag [L1] | Yes | **100%** | 0/40 |
| T2: Section-header [L1] | Yes | **100%** | 0/40 |
| T3: Checklist [L1/L2] | Yes | **0%** | 40/40 |
| T4: Reasoning [L2] | No | 35% | 26/40 |
| T5: Uncertainty [L2/L3] | No | 42.5% | 2/40 |

**Violations cluster precisely where mechanical gates cannot reach.** T3 is the critical case: the mechanical check (`- [ ]` format, a checklist EXISTS) passes on all 40 trials, but all 40 fail semantic verification (the checklist content is meaningless). This is what we call the **receipt-of-action vs. receipt-of-diligence** problem: the gate verifies a receipt was issued, not that work was done.

---

## 2. Cross-Type Bidirectional Verification (CTBV)

### 2.1 Core Insight

Mechanical gates and semantic gates fail on **orthogonal** output classes:

- **Mechanical gates** fail on semantic violations: content that is structurally correct but substantively empty (hollow compliance).
- **Semantic gates** fail on format violations: content that is substantively sound but structurally malformed (missing tags, wrong file paths).

Critically, these failure modes are **not correlated** — there is no structural reason why a semantically hollow output would systematically trigger a format violation, or vice versa. This orthogonality is what CTBV exploits.

### 2.2 Mathematical Formalization

Let `O` be the space of all agent outputs. Define two binary classifiers:

- **Mechanical gate** `M: O → {pass, fail}` with false-negative rate `α_m = P(M(o) = pass | o is non_compliant)`
- **Semantic gate** `S: O → {pass, fail}` with false-negative rate `α_s = P(S(o) = pass | o is non_compliant)`

Under CTBV, an output passes only if **both** gates pass:
```
CTBV(o) = pass ⟺ M(o) = pass ∧ S(o) = pass
```

The false-negative rate of CTBV is:
```
α_ctbv = P(M = pass ∧ S = pass | o is non_compliant)
```

Assume that the failure events of M and S are **conditionally independent given o is non_compliant** — i.e., knowing that a mechanical gate missed a violation tells us nothing about whether a semantic gate will also miss it, because they operate on different signal types (filesystem state vs. token probabilities). Then:

```
α_ctbv = α_m × α_s
```

**CTBV Upper Bound Theorem**: If mechanical gates (α_m) and semantic gates (α_s) are independent given non-compliance, then:
```
P(error_ctbv) ≤ P(error_mechanical) × P(error_semantic | orthogonal_blind_spots)
```

With our empirical estimates (α_m ≈ 0.007 from Claim 2; α_s estimated at 0.10-0.30 from L2/L3 dissociation data), the joint error rate is bounded at **0.07%–0.21%** — a 33×–100× improvement over single-gate architectures.

### 2.3 Why Independence is Plausible

The independence assumption is not arbitrary; it follows from the **different measurement substrates**:

| Gate Type | Measures | Substrate | Blind To |
|-----------|----------|-----------|----------|
| Mechanical (L1) | File mtime, exit codes, regex patterns | Filesystem | Semantic content, reasoning quality, intent |
| Semantic (L2/L3) | Logprob differentials, format encoding, behavioral compliance | Token probability distributions | File integrity, hook wiring, filesystem state |

A mechanical gate checking whether `dashboard.md` was modified cannot detect whether the modification was substantive. A semantic gate measuring whether a constraint "penetrated" the model's token distribution cannot detect whether the resulting file was actually written. They are blind to **different things** — and this difference is what makes their joint failure rate multiplicative.

### 2.4 Empirical Validation of Orthogonality

**Evidence 1: L2/L3 Dissociation** (Claim 4). Logprob probes show d=+0.578 favoring syllogistic format at the token level (L2), but behavioral compliance shows IMP≈SYL (Δ=-0.024) at the output level (L3). The same 21 probes, the same model, the same temperature — but two different measurement layers yield divergent results. This is direct evidence that token-level and output-level gates capture **different dimensions** of compliance.

**Evidence 2: Format-L1 Synergy** (Claim 7). Format effects are stronger where L1 has already instrumented the rule (d_z=0.71) and weaker where L1 has not (d_z=0.40). The mechanical gate creates "anchor points" that the semantic format amplifies — synergy, not redundancy.

**Evidence 3: Constraint Gradient Non-Monotonicity** (Claim 5). The constraint gradient across layers L0→L4 is non-monotonic: d_z drops from L0 (0.315) to L2 (0.091) before rebounding at L3 (0.297). If layers measured the same thing, the gradient would be monotonic. The non-monotonic pattern confirms layer independence.

---

## 3. Five-Layer Architecture

```
L0: Psychological Safety (pre-barrier)
    └── Constitutional rules + amendment lifecycle
    └── Tells model "not knowing" is valid output → reduces RLHF confidence bias

L1: Mechanical Gate (outside barrier — filesystem checks)
    └── mtime comparison, exit codes, regex scoring, JSON schema validation
    └── 100% deterministic, zero NL understanding, 98.7% violation reduction

L2: Neural Gate (inside barrier — logprob probes)
    └── Paired API calls (with/without constraint) → logprob differential
    └── Detects constraint PENETRATION without interpreting content

L3: Causal Encoding (through barrier — syllogistic format)
    └── Major-premise/Minor-premise/Conclusion format reroutes attention
    └── Format changes token probability distribution — engineering the INPUT side

L4: Drift Prediction (outside barrier, temporal axis)
    └── 8-feature scoring → 0-100 risk score + temporal trend tracking
    └── Only layer with cross-session trajectory awareness
```

### 3.1 Layer Independence Matrix

Each layer measures something the others cannot (verified via counterfactual removal analysis):

| If we remove → | L0 | L1 | L2 | L3 | L4 |
|---------------|:--:|:--:|:--:|:--:|:--:|
| Can L0 replace? | — | ❌ | ❌ | ❌ | ❌ |
| Can L1 replace? | ❌ | — | ❌ | ❌ | ❌ |
| Can L2 replace? | ⚠️ | ❌ | — | ⚠️ | ❌ |
| Can L3 replace? | ❌ | ❌ | ⚠️ | — | ❌ |
| Can L4 replace? | ❌ | ⚠️ | ❌ | ❌ | — |

❌ = Cannot replace (different measurement substrate)
⚠️ = Partial overlap (measurement-linked but conceptually distinct)

**L1 is the only layer operating entirely outside the Prose Barrier.** It is fully independent — no other layer checks filesystem state.

### 3.2 Practical Rule: Receipt-of-Action vs. Receipt-of-Diligence

A key design principle emerged from the P1-1 T3 result: mechanical checks verify **receipt-of-action** (a checklist was produced, a file was modified). Semantic checks verify **receipt-of-diligence** (the checklist content is genuine, the file modification was substantive). CTBV requires both.

---

## 4. Empirical Evidence: 10 Claims Synthesized

### Claim 1: The Prose Barrier is structural, not contingent
- **Evidence**: 55.9% → 0.7% violation rate reduction via pure mechanical gate
- **Implication**: Self-verification cannot be "prompted away" — the constraint is architectural
- **Experiments**: 34-session retrospective, 150-task A/B test

### Claim 2: L1 Mechanical Gates achieve near-perfect compliance (99.3%)
- **Evidence**: 149/150 tasks zero violations; 19/19 behavioral tests pass; cross-model replication
- **Implication**: Mechanical enforcement is the highest-leverage single intervention for agent reliability
- **Experiments**: Syllogism vs Imperative A/B, P1-1 Residual Cluster, cross-model behavioral

### Claim 3: L2 Logprob Probes detect constraint penetration without interpretation
- **Evidence**: d=+0.578, BF=282k, 95% CI [+3.39, +11.17], 32/40 probes favoring syllogistic
- **DV**: API-read logprob (objective, not human-scored)
- **Implication**: Token probability differentials are a viable **mechanical proxy** for semantic penetration
- **Experiments**: Logprob V3 (40 probes × 4 categories, 120 API calls)

### Claim 4: L2/L3 Dissociation — format changes internal processing but not behavior
- **Evidence**: Logprob d=+0.578 (L2) vs. behavioral Δ=-0.024 (L3); n=21 paired
- **Sensitivity**: Can only exclude d≥0.65 at current n; need n≥90 for d=0.3 detection
- **Implication**: Token-level and output-level compliance are **different dimensions** — validating the multi-layer architecture
- **Experiments**: GateGuard-OFF (21 probes × 3 conditions), Logprob V3

### Claim 5: Constraint Gradient is non-monotonic across layers
- **Evidence**: d_z(L0)=0.315 → d_z(L1)=0.596 → d_z(L2)=0.091 → d_z(L3)=0.297
- **Pattern**: L1 amplifies constraint signal; L2 suppresses it (context competition); L3 partially recovers
- **Implication**: Constraints do not accumulate additively. Layer ordering matters.
- **Experiments**: Constraint Gradient (12 tasks × 2 formats × 4 levels, 96 API calls)

### Claim 6: Format effects are context-fragile
- **Evidence**: Single-scenario d_z=0.58 → multi-position d_z=0.19; r=-0.65 with V3; meta-instruction drives ~80% collapse
- **Implication**: Format effects observed in controlled experiments may not survive real multi-task sessions
- **Experiments**: P1 Multi-Position (24 calls), P1 Controls (48 calls)

### Claim 7: Format-L1 Synergy (not compensation)
- **Evidence**: d_z=0.71 (L1-visible rules) vs 0.40 (L1-invisible rules)
- **Implication**: Format amplifies existing structural anchors; it does not create them from nothing
- **Experiments**: L1-Visibility Analysis (40 probes)

### Claim 8: Prose format improves reasoning; Code format perfects mechanics
- **Evidence**: Prose+Gate = best reasoning (4.42/5); Code+Gate = perfect mechanical (5.0/5) but worst reasoning (4.20)
- **H1 Falsified**: Format effect on reasoning is NOT larger under GateGuard-OFF (d=-0.277 ON vs d=-0.250 OFF)
- **H2 Strongly Confirmed**: Format effect on mechanical compliance is MASSIVE under GateGuard-ON (d=+2.96) but ZERO under OFF (d=0.0)
- **Implication**: Code format optimizes for what the gate is already solving
- **Experiments**: P1-2 Format×Gate 2×2 Factorial (240 trials)

### Claim 9: Cross-Model Gateability = Rule Structure × Model Capacity (two-axis)
- **Evidence**: DS Pro scanner alignment drops from 5/5 to 2/5 cross-model; DS Flash shows 100% hollow compliance; Qwen T1=40%, T5=0%
- **Implication**: Gate effectiveness depends on BOTH rule design AND model capability. One-size-fits-all rules fail.
- **Experiments**: Cross-Model Scanner Calibration (200 API calls, 3 models)

### Claim 10: GateGuard creates a near-perfect compliance ceiling
- **Evidence**: 149/150 (99.3%) zero-violation tasks; the single violation was self-detected by the syllogism agent
- **Limitation**: Ceiling effect makes format differences undetectable — need GateGuard=OFF to isolate
- **Experiments**: Syllogism vs Imperative A/B (6 sessions, 150 tasks)

---

## 5. 16 Experiments: Complete Inventory

| # | Experiment | Design | N | Key Finding |
|---|-----------|--------|:---:|------|
| 1 | 34-Session Retrospective | Longitudinal, 5 rules | 34 sessions | 55.9% → 0.7% via L1 gate |
| 2 | Logprob V3 | 40 probes × 4 categories | 120 calls | d=+0.578, BF=282k, L2 penetration detected |
| 3 | Constraint Gradient | 12 tasks × 2 formats × 4 levels | 96 calls | Non-monotonic: L1↑L2↓L3↑ |
| 4 | GateGuard-OFF | 21 probes × 3 conditions | 63 trials | L2≠L3: IMP≈SYL behaviorally, SYL>IMP neurally |
| 5 | Syllogism vs Imperative A/B | 2 conditions × 150 tasks | 6 sessions | 99.3% zero violations; ceiling effect |
| 6 | P1-1 Residual Cluster | 5 task types × 40 trials | 200 calls | Violations cluster on non-mechanizable dims |
| 7 | P1-2 Format×Gate Factorial | 4 conditions × 2 tasks × 30 | 240 calls | Prose>Code for reasoning; Gate amplifies mechanical format gap |
| 8 | Cross-Model Scanner Calibration | 5 tasks × 3 models × 20 trials | 200 calls | Two-axis gateability: structure × capacity |
| 9 | L1-Visibility Analysis | 40 probes | 40 probes | d_z=0.71 (L1-visible) vs 0.40 (L1-invisible): synergy |
| 10 | P1 Multi-Position | 24 probes | 24 calls | d_z collapses 0.58→0.19: context fragility |
| 11 | P1 Controls | 48 probes | 48 calls | Meta-instruction drives ~80% of collapse |
| 12 | Mechanizability Scanner Calibration | 5 rules × scanner vs ground truth | — | 5/5 alignment; 2 bugs found and fixed |
| 13 | SHA256 Pre-Registration | SHA256(P1-2 script) verified | — | Reproducibility confirmed |
| 14 | Cross-Model Replication (Community) | P1-1 × 2 additional models | — | Structure × Capacity confirmed |
| 15 | Persona Decorrelation | E1: persona influence on rule compliance | — | Persona effects independent of gate effects |
| 16 | Digital Twin Training | QLoRA + DPO, 4-stage pipeline | 253 samples | Gate passed; Phase 3 complete |

**Total**: 1,100+ formal trials, ~2,500 API calls, 3 LLM architectures, 10 validated claims, 2 falsified hypotheses (honestly reported).

---

## 6. Practical Implications for Agent Builders

### 6.1 The Minimum Viable Agent Reliability Stack

Based on the 16 experiments, the minimum effective configuration for an LLM agent that modifies files or executes code is:

1. **Mechanical gate (L1)** — non-negotiable. Filesystem checks (mtime, exit codes, regex) reduce violations by 98.7%. This is the highest-ROI single intervention.
2. **Constitutional rules (L0)** — pre-barrier. Establish that "not knowing" is a valid output. Costs nothing, prevents confidence-bias failures.
3. **Format-aware rule design (L3)** — prose for reasoning tasks, code-format for mechanical tasks. The P1-2 result (prose → better reasoning, code → perfect mechanics) is directly actionable.

L2 (logprob probes) and L4 (drift prediction) are recommended for **safety-critical** deployments but add API cost overhead (~$0.50 per 100 probes at current DeepSeek V4 pricing).

### 6.2 Decision Matrix

| Deployment Type | Minimum Stack | Recommended Stack |
|----------------|---------------|-------------------|
| Personal automation | L1 only | L0 + L1 |
| Team-internal tool | L0 + L1 | L0 + L1 + L3 |
| Customer-facing agent | L0 + L1 + L3 | L0 + L1 + L2 + L3 |
| Safety-critical system | All L0-L4 | All L0-L4 + human-in-loop |
| Research/experimentation | L1 (for data integrity) | L0 + L1 + L3 |

### 6.3 Checklist for Agent Builders

Before deploying an LLM agent that modifies state (files, databases, API calls):

- [ ] Does every write operation have a mechanical post-condition check (regex, mtime, exit code)?
- [ ] Can the agent's self-report of "I checked X" be mechanically verified without reading its prose?
- [ ] Are rules that require semantic judgment (reasoning quality, epistemic stance) explicitly labeled as non-mechanizable?
- [ ] For non-mechanizable rules: is there a fallback (human review, second model, lower stakes)?

### 6.4 The "Code Format Trap"

P1-2 revealed a counter-intuitive finding: code-format rules + mechanical gate = **perfect mechanical compliance (5.0/5)** but **worst reasoning depth (4.20/5)**. The model satisfies the mechanical requirements and stops thinking. This is dangerous for systems where reasoning quality matters (clinical decision support, financial analysis, safety-critical recommendations).

**Recommendation**: Use code-format rules ONLY for purely mechanical tasks (file naming, format tags, exit code checks). For any task requiring reasoning, use prose-format rules with mechanical scaffolding.

---

## 7. Limitations (Honest Disclosure)

This work has limitations that should be acknowledged upfront:

1. **Single-model logprob experiments**: Logprob V3, Constraint Gradient, and L1-Visibility all use DeepSeek V4 Pro only. Cross-model logprob replication is the highest-priority next step (~$5, ~30 min).

2. **No independent second rater**: All behavioral scoring uses deterministic regex matching (by design — this is a feature, not a bug, as LLM judges share the Prose Barrier). But this means κ=0.00 (single rater, zero variance). Human blind scoring of a random sample is needed.

3. **No pre-registration** (except P1-2): Most experiments were exploratory. P1-2 was pre-registered and its primary hypothesis was **falsified** — we report this honestly as evidence of the pre-registration process working.

4. **Small samples for some analyses**: GateGuard-OFF n=21 (underpowered for d≤0.65), P1 Multi-Position n=24, P1 Controls n=48. These are sufficient for the effects they detected but cannot exclude small effects.

5. **Single-machine implementation**: All hooks and gates run on one developer's Windows laptop. No distributed deployment, no multi-user testing.

6. **No placebo control**: We cannot distinguish "rules changed behavior" from "any text in system prompt changes behavior." The NO RULES baseline (Claim 4 Finding 1) partially addresses this but is not a true placebo.

7. **L4 drift prediction not predictively validated**: The 8-feature drift scoring system exists but has not been tested against future degradation events.

8. **No real-world deployment beyond the author**: All experimental data comes from one user's configuration system. External validation (at least one independent deployer) is needed.

---

## 8. By the Numbers (Quick Reference)

| Metric | Value |
|--------|-------|
| Total experiments | 16 |
| Total API calls | ~2,500 |
| Formal experiment trials | 1,100+ |
| LLM architectures tested | 3 (DeepSeek V4 Pro, DeepSeek Flash, Qwen) |
| Claims validated | 10 |
| Hypotheses falsified | 2 (P1-2 H1, honestly reported) |
| Violation reduction (L1 gate) | 98.7% (55.9% → 0.7%) |
| Zero-violation ceiling (full stack) | 99.3% (149/150 tasks) |
| Logprob effect size (L2) | d=+0.578, BF=282k |
| Prose reasoning advantage | d=+0.605 over code format |
| Code mechanical advantage | d=+2.96 under GateGuard-ON |
| Hook scripts in production | 20+ (health-check, quality-gate, write-guard, content-guard, evidence-gate, pipeline-update-gate, contribution-marker, drift-scorer, etc.) |
| Community contributors | 4 DEV.to readers proposed experiments adopted into pipeline |
| DEV.to articles documenting work | 32 |
| DEV.to reactions across all articles | 29 |
| DEV.to followers | 608 |

---

## 9. Related Work

- **Rene Zander's skillgate**: Deterministic gate engine with identical L1 architecture (filesystem-level verification, reject model self-reports). Our unique contributions: self-referential loop, L2 neural gates, L3 causal encoding, L4 drift prediction.
- **Constitutional AI (Bai et al., 2022)**: Uses AI feedback for harmlessness training. Our work differs: we use constitutional rules as pre-generation filters (L0), not post-hoc training objectives.
- **Tool Verification (various)**: Tools like guardrails.ai and NVIDIA NeMo Guardrails operate at the content-filter level. Our approach operates at the **configuration integrity** level — verifying the agent's own internal state, not its external outputs.
- **Self-Consistency (Wang et al., 2022)**: Multiple sampling + majority vote. Shares the Prose Barrier limitation — all samples from the same decoder.

---

## 10. Next Steps

### Immediate (1-4 weeks, ≤$10, solo-executable)

1. **Cross-model logprob replication**: Run Logprob V3 probe set against Claude/GPT-4o API. Confirm or refute d=+0.578 across architectures. Cost: ~$5.
2. **Independent blind scoring**: Have one other person score 50 random trials from P1-1 and P1-2. Compute Cohen's κ.
3. **Anchor article**: Identify the 3 strongest DEV.to articles from the 32 published for portfolio narrative cohesion.

### Medium-term (1-3 months, needs external validator)

4. **External deployment**: Find one person deploying LLM agents in production to install the L0+L1 stack and report violation rates.
5. **Power analysis for L2/L3 dissociation**: Run GateGuard-OFF at n≥90 to exclude d=0.3 behavioral format effect.

### Longer-term (3-6 months)

6. **Pre-registered constraint gradient**: Formal non-monotonicity test with held-out probes.
7. **L4 predictive validation**: 20+ session tracking to test drift predictor accuracy.
8. **Submit to CHI LBW or ACL SRW**: After external validation and cross-model replication.

---

## Appendix A: Key Equations

### A.1 Prose Barrier

```
y ~ P(· | x; θ)           # Generation
v ~ P(· | y, x; θ)        # Self-verification
P(v=c | y=nc) = P(v=c, y=nc | x; θ) / P(y=nc | x; θ)
# Both share θ → no structural reliability guarantee
```

### A.2 CTBV Joint Error

```
α_ctbv = P(M=pass ∧ S=pass | nc)
       = P(M=pass | nc) × P(S=pass | M=pass, nc)  [by chain rule]
       = α_m × P(S=pass | M=pass, nc)

Assuming conditional independence (orthogonal blind spots):
       = α_m × α_s
       ≈ 0.007 × 0.20 = 0.0014 (0.14%)
```

### A.3 Layer Independence (Counterfactual Test)

```
Independence(L_i) = ¬∃ L_j (j≠i) : FailureMode(L_i) ⊆ FailureMode(L_j)
```

### A.4 Two-Axis Gateability

```
Compliance = f(RuleStructure, ModelCapacity)
           ≠ f(RuleStructure) alone
           ≠ f(ModelCapacity) alone
```

---

## Appendix B: Code and Data

- **Experiment code**: [github.com/YuhaoLin2005/paper-validator](https://github.com/YuhaoLin2005/paper-validator)
- **Configuration system (live)**: [github.com/YuhaoLin2005/hermes-workspace](https://github.com/YuhaoLin2005/hermes-workspace)
- **Hook scripts (production)**: `~/.claude/scripts/` (health-check, quality-gate, write-guard, content-guard, evidence-gate, pipeline-update-gate, contribution-marker, drift-scorer, etc.)
- **Experiment data**: `paper-validator/results/*.json` (16 experiment result files)
- **Paper draft**: `hermes-workspace/paper/acl-submission/main.tex`
- **Claims database**: `hermes-workspace/.claude/knowledge/paper/claims.md`
- **Dashboard (live metrics)**: `hermes-workspace/.claude/knowledge/strategy/dashboard.md`

## Appendix C: Commenters and Community

This work benefited from technical discussions with DEV.to readers who proposed experiments later adopted into the pipeline:

- **Mike Czerwinski** — Proposed P1-1 residual violation clustering + P1-2 GateGuard-off format effect test
- **Dipankar Sarkar** — Proposed SHA256 pre-registration; warned against LLM-judge bias in scoring
- **Max Quimby** — Proposed mechanizability scanner calibration
- **Rene Zander** — Independent implementation of deterministic gate engine (skillgate); architecture comparison
- **Alex Shevchenko** — Technical feedback on experiment design
- **CodeKitHub** — Community replication discussion

---

*This report is a living document. Last updated 2026-07-28. Corrections, questions, and collaboration inquiries: linyuhao2005@gmail.com.*
