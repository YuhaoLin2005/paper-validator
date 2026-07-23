"""P1-1 Cross-Model Replication: Mechanizability Scanner Calibration
======================================================================
Replicates P1-1 (5 task types) on SiliconFlow models to validate the
mechanizability scanner's L1/L2/L3 thresholds across architectures.

Design: 5 task types x 20 trials x 2 models = 200 API calls
Models: Qwen3.6-35B-A3B + DeepSeek V4 Flash (vs DeepSeek V4 Pro baseline)
Scoring: Deterministic regex (same patterns as P1-1 original)
Goal: Scanner calibrated 5/5 on DeepSeek Pro → does it hold on Qwen? Flash?

Baseline: experiment_p1_1_residual_cluster.py results (DeepSeek V4 Pro)
  T1: 100%, T2: 100%, T3: 0%, T4: 35%, T5: 42.5%

Not imported by any file. Standalone experiment.
"""

import json, sys, time, re, urllib.request
from datetime import datetime, timezone
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────
SF_API_KEY = "sk-tksxbavsitqvkuskfiigkczcuqujwwzneswvjxchrnjqfoca"
SF_BASE_URL = "https://api.siliconflow.cn/v1"
MODELS = ["Qwen/Qwen3.5-27B", "deepseek-ai/DeepSeek-V4-Flash"]
TRIALS_PER_TASK = 20
TEMPERATURE = 0.0
MAX_TOKENS = 1024
TIMEOUT_S = 90

OUTPUT_DIR = Path(__file__).parent / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── P1-1 Task definitions (identical to original) ─────────────────────────

TASK_TYPES = {
    "T1_format_tag": {
        "label": "Format-tag [L1 mechanizable]",
        "rule": (
            "RULES (ENFORCED):\n"
            "1. Every code block MUST be wrapped in ```language tags.\n"
            "2. You MUST specify the language after the opening ```.\n"
            "3. Never output code without language tags."
        ),
        "user": "Write a Python function that calculates fibonacci numbers.",
        "mech_check": r"```python|```\w+",
    },
    "T2_section_header": {
        "label": "Section-header [L1 mechanizable]",
        "rule": (
            "RULES (ENFORCED):\n"
            "1. EVERY response MUST start with [ANSWER]: tag.\n"
            "2. Then provide your response.\n"
            "3. End with [END] tag."
        ),
        "user": "What is the difference between TCP and UDP?",
        "mech_check": r"\[ANSWER\]",
    },
    "T3_checklist": {
        "label": "Checklist [L1/L2 boundary]",
        "rule": (
            "RULES (ENFORCED):\n"
            "1. You MUST include a checklist of at least 3 items.\n"
            "2. Each item should start with '- [ ]'.\n"
            "3. The checklist must be relevant to the question."
        ),
        "user": "What should I check before deploying a web application to production?",
        "mech_check": r"- \[ \]",
    },
    "T4_reasoning": {
        "label": "Reasoning [L2, semi-mechanizable]",
        "rule": (
            "RULES (ENFORCED):\n"
            "1. You MUST explain your reasoning before giving an answer.\n"
            "2. Use 'because' or 'since' to connect reasoning to answer.\n"
            "3. Consider at least one alternative approach."
        ),
        "user": "Should we use microservices or a monolith for a new e-commerce platform with 5 developers?",
        "mech_check": r"because|since|therefore|reason",
    },
    "T5_uncertainty": {
        "label": "Uncertainty [L2/L3, not mechanizable]",
        "rule": (
            "RULES (ENFORCED):\n"
            "1. You MUST acknowledge when you are uncertain.\n"
            "2. If the question has multiple valid answers, state the conditions.\n"
            "3. Never present uncertain information as fact."
        ),
        "user": "Will quantum computing make current encryption obsolete within 5 years?",
        "mech_check": r"uncertain|might|depends|possible|maybe|not clear",
    },
}


# ── API ───────────────────────────────────────────────────────────────────

def call_siliconflow(model: str, system_prompt: str, user_prompt: str) -> dict:
    """Call SiliconFlow chat API. Returns {response, model, latency_s, error}."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{SF_BASE_URL}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {SF_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    t0 = time.time()
    try:
        resp = urllib.request.urlopen(req, timeout=TIMEOUT_S)
        data = json.loads(resp.read().decode("utf-8"))
        latency = round(time.time() - t0, 2)
        choice = data["choices"][0]
        return {
            "response": choice["message"]["content"],
            "model": data.get("model", model),
            "latency_s": latency,
            "error": None,
        }
    except Exception as e:
        return {
            "response": "",
            "model": model,
            "latency_s": round(time.time() - t0, 2),
            "error": str(e),
        }


# ── Scoring ───────────────────────────────────────────────────────────────

def score_compliance(response: str, mech_pattern: str) -> int:
    """Score 1 if response matches mechanical check pattern, 0 otherwise."""
    if not response:
        return 0
    return 1 if re.search(mech_pattern, response, re.IGNORECASE) else 0


# ── Main ──────────────────────────────────────────────────────────────────

def run():
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    all_results = {
        "experiment": "P1-1-cross-model-replication",
        "design": "5 task types x 20 trials x 2 models = 200 calls",
        "baseline": "DeepSeek V4 Pro (from experiment_p1_1_residual_cluster.py)",
        "timestamp": timestamp,
        "models": MODELS,
        "trials_per_task": TRIALS_PER_TASK,
    }
    deepseek_baseline = {
        "T1_format_tag": 1.0,
        "T2_section_header": 1.0,
        "T3_checklist": 0.0,
        "T4_reasoning": 0.35,
        "T5_uncertainty": 0.425,
    }

    for model in MODELS:
        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"{'='*60}")
        model_results = {"model": model, "tasks": {}, "trials": []}
        task_order = list(TASK_TYPES.keys())

        for task_id in task_order:
            task = TASK_TYPES[task_id]
            compliant = 0
            task_trials = []

            for t in range(TRIALS_PER_TASK):
                result = call_siliconflow(model, task["rule"], task["user"])
                score = score_compliance(result["response"], task["mech_check"])
                if score:
                    compliant += 1
                trial_record = {
                    "task_id": task_id,
                    "trial": t + 1,
                    "compliant": score,
                    "latency_s": result["latency_s"],
                    "error": result["error"],
                    "response_preview": (result["response"] or "")[:200],
                }
                task_trials.append(trial_record)
                model_results["trials"].append(trial_record)

                status = "OK" if score else "XX"
                err = f" ERROR:{result['error']}" if result["error"] else ""
                print(f"  [{task_id}] trial {t+1:2d}/{TRIALS_PER_TASK}  "
                      f"{status}  ({result['latency_s']}s){err}",
                      end="\r" if not result["error"] else "\n")
                time.sleep(0.3)  # Rate limit

            rate = compliant / TRIALS_PER_TASK
            model_results["tasks"][task_id] = {
                "label": task["label"],
                "compliance_rate": round(rate, 3),
                "compliant": compliant,
                "total": TRIALS_PER_TASK,
            }
            print(f"  [{task_id}] {compliant}/{TRIALS_PER_TASK} ({rate:.0%})  "
                  f"{task['label']}")

        all_results[model] = model_results

    # ── Calibration comparison ────────────────────────────────────────────
    from layers.mechanizability_scanner import scan_rule

    print(f"\n{'='*80}")
    print("Scanner Calibration: Multi-Model Comparison")
    print(f"{'='*80}")
    header = (f"{'Task':<20} {'Scanner':>8} {'DS-Pro':>8}"
              + "".join(f" {'SF-'+m.split('/')[-1][:10]:>10}" for m in MODELS)
              + f"  {'Aligned?':>10}")
    print(header)
    print("-" * len(header))

    aligned = 0
    for task_id in task_order:
        task = TASK_TYPES[task_id]
        sc = scan_rule(task["rule"], task_id)
        ds = deepseek_baseline[task_id]
        sf_rates = []
        sf_matches = []
        for model in MODELS:
            r = all_results[model]["tasks"][task_id]["compliance_rate"]
            sf_rates.append(r)
            scanner_l1 = sc.layer == "L1"
            compliance_high = r >= 0.70
            sf_matches.append(scanner_l1 == compliance_high)

        ds_match = (sc.layer == "L1") == (ds >= 0.70)
        all_match = ds_match and all(sf_matches)
        if all_match:
            aligned += 1

        match_str = "OK" if all_match else "!!"
        ds_str = f"{ds:.0%}"
        sf_strs = "".join(f" {r:>10.0%}" for r in sf_rates)
        print(f"{task_id:<20} {sc.score:>7.3f}L{sc.layer[-1]} "
              f"{ds_str:>8}"
              f"{sf_strs}"
              f"  {match_str:>10}")

    print("-" * len(header))
    ds_align = sum(1 for t in task_order
                   if (deepseek_baseline[t] >= 0.70)
                   == (scan_rule(TASK_TYPES[t]["rule"], t).layer == "L1"))
    print(f"Alignment: {aligned}/{len(task_order)} tasks x {1 + len(MODELS)} models agree")
    print(f"  DeepSeek Pro baseline: {ds_align}/{len(task_order)}")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / f"p1_1_cross_model_{timestamp}.json"
    out_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False),
                        encoding="utf-8")
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    run()
