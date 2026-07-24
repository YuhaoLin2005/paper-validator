"""P2: Causal Format Experiment — 三段论因果格式 vs 祈使命令格式

Hypothesis: 三段论因果格式（大前提→判断→结论）比祈使命令格式产生更高的约束遵守率。
不是因为"更强的规则"，而是因果格式激活推理链而非服从链。

4 conditions: SYLL | IMP | TOK-CTRL | COT-CTRL
25 tasks (5 rule types × 5 tasks), blind LLM-judge scoring ×2 raters
"""

from __future__ import annotations

import hashlib, json, os, sys, time
from pathlib import Path
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from engine.api_client import quick_chat, extract_text
from scoring.rubric import (
    deterministic_score, judge_system_prompt, judge_user_prompt,
    aggregate_scores, fleiss_kappa,
)

BASE = Path(__file__).resolve().parent.parent
RULES_DIR = BASE / "rules"
TASKS_PATH = BASE / "tasks" / "25-standard-tasks.json"
RESULTS_DIR = BASE / "results"

CONDITION_FILES = {"syll": "syll.md", "imp": "imp.md",
                    "tokctrl": "tokctrl.md", "cotctrl": "cotctrl.md"}

RULE_DESCRIPTIONS = {
    "R1_read_after_write": "After Write/Edit, must Read file back to verify before claiming completion.",
    "R2_dual_pool_review": "Multi-file/architecture decisions require ≥2 independent review perspectives.",
    "R3_pre_action_check": "Before Edit/Write, check Q1 correctness, Q2 consistency, Q3 verification plan.",
    "R4_auto_deposit": "When task produces new insights/patterns/changes, write to growth-log or decisions/log.",
    "R5_adversarial_self_audit": "Before declaring completion, sequentially audit C/C/G/H dimensions.",
}

TRIGGER_CONTEXT = {
    "R1_read_after_write": "Task involves write/edit. Rule should trigger.",
    "R2_dual_pool_review": "Task may involve multi-file/architecture. Judge if rule triggers.",
    "R3_pre_action_check": "Task may involve write/edit. Judge if pre-check needed.",
    "R4_auto_deposit": "Task may produce insights. Judge if deposit needed.",
    "R5_adversarial_self_audit": "Task reaches completion. Judge if self-audit appropriate.",
}


def load_condition(condition: str) -> str:
    filename = CONDITION_FILES.get(condition)
    if not filename:
        raise ValueError(f"Unknown: {condition}. Valid: {list(CONDITION_FILES)}")
    return (RULES_DIR / filename).read_text(encoding="utf-8")


def load_tasks() -> list[dict]:
    with open(TASKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["tasks"]


def pre_register(condition: str, tasks: list[dict]) -> str:
    payload = {"condition": condition, "n_tasks": len(tasks),
               "task_ids": [t["id"] for t in tasks],
               "scoring": "blind_llm_judge_x2",
               "primary_metric": "mean_compliance_score",
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def run_condition(condition: str, *, api_key=None, model=None,
                   max_tasks=None, dry_run=False) -> dict:
    system_prompt = load_condition(condition)
    tasks = load_tasks()
    pr_hash = pre_register(condition, tasks)
    if max_tasks:
        tasks = tasks[:max_tasks]

    trials, total_tokens = [], 0
    print(f"\n{'='*60}\nCondition: {condition.upper()}  |  Pre-reg: {pr_hash}  |  Tasks: {len(tasks)}\n{'='*60}")

    for i, task in enumerate(tasks):
        tid = task["id"]
        if dry_run:
            print(f"  [{i+1}/{len(tasks)}] {tid} — DRY RUN")
            trials.append({"task_id": tid, "response": "[dry_run]", "scoring": None})
            continue

        print(f"  [{i+1}/{len(tasks)}] {tid} ...", end=" ", flush=True)
        response = quick_chat(system_prompt, task["prompt"],
                              api_key=api_key, model=model,
                              max_tokens=2048, temperature=0.0)
        if response is None:
            print("FAILED")
            trials.append({"task_id": tid, "response": "[API_ERROR]", "scoring": None, "usage": {}})
            continue

        text = extract_text(response)
        usage = response.get("usage", {})
        total_tokens += usage.get("total_tokens", 0)
        det = deterministic_score(task["rule_type"], text)
        print(f"OK ({len(text)}c, det={det['score']})")
        trials.append({"task_id": tid, "rule_type": task["rule_type"],
                       "task": task["prompt"], "response": text,
                       "usage": usage, "deterministic_pre_score": det["score"]})
        time.sleep(0.5)

    return {"condition": condition, "pre_reg_hash": pr_hash,
            "n_tasks": len(tasks),
            "n_completed": sum(1 for t in trials if t.get("response") != "[API_ERROR]"),
            "total_tokens": total_tokens, "trials": trials,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def blind_score_trials(trials: list[dict], *, api_key=None, model=None) -> list[dict]:
    for i, trial in enumerate(trials):
        if trial.get("response") in (None, "[API_ERROR]", "[dry_run]"):
            trial["llm_score"] = None; trial["final_score"] = None; continue

        rt = trial["rule_type"]
        sys_p = judge_system_prompt(rt, RULE_DESCRIPTIONS.get(rt, ""))
        usr_p = judge_user_prompt(trial.get("task", ""), trial["response"],
                                   rt, TRIGGER_CONTEXT.get(rt, ""))
        print(f"  Scoring {trial['task_id']} ...", end=" ", flush=True)
        jr = quick_chat(sys_p, usr_p, api_key=api_key, model=model,
                         max_tokens=256, temperature=0.0)
        if jr is None:
            trial["llm_score"] = {"score": "ERROR"}; trial["final_score"] = None
            print("FAILED"); continue

        jt = extract_text(jr).strip()
        try:
            if "```" in jt:
                jt = jt.split("```")[1]
                if jt.startswith("json"):
                    jt = jt[4:]
            parsed = json.loads(jt)
            s = parsed.get("score", "ERROR")
            trial["final_score"] = "NA" if s == "NA" else (int(s) if s != "ERROR" else None)
            trial["llm_score"] = parsed
            print(f"-> {s}")
        except (json.JSONDecodeError, ValueError, KeyError):
            trial["llm_score"] = {"score": "PARSE_ERROR", "raw": jt[:200]}
            trial["final_score"] = None
            print("PARSE_ERROR")
        time.sleep(0.3)
    return trials


def run_full_experiment(conditions=None, *, api_key=None, model=None,
                         max_tasks=None, dry_run=False) -> dict:
    if conditions is None:
        conditions = list(CONDITION_FILES)
    all_results = {}

    for cond in conditions:
        all_results[cond] = run_condition(cond, api_key=api_key, model=model,
                                           max_tasks=max_tasks, dry_run=dry_run)

    if dry_run:
        return all_results

    print(f"\n{'='*60}\nBLIND SCORING\n{'='*60}")
    for cond in conditions:
        print(f"\n  {cond.upper()}:")
        blind_score_trials(all_results[cond]["trials"], api_key=api_key, model=model)
        agg = aggregate_scores(all_results[cond]["trials"])
        all_results[cond]["aggregate"] = agg
        print(f"    Mean={agg.get('mean')}, n={agg.get('n')}, disc_acc={agg.get('discrimination_accuracy')}")

    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    for cond in conditions:
        a = all_results[cond].get("aggregate", {})
        m = a.get('mean')
    mean_str = f"{m:.3f}" if m is not None else "N/A"
    print(f"  {cond:10s}: mean={mean_str:>6}  n={a.get('n', 0)}  disc={a.get('discrimination_accuracy', 'N/A')}")
    return all_results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="P2 Causal Format Experiment")
    ap.add_argument("--condition", "-c", choices=list(CONDITION_FILES) + ["all"], default="all")
    ap.add_argument("--max-tasks", "-n", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--output", "-o", type=str, default=None)
    args = ap.parse_args()

    conditions = list(CONDITION_FILES) if args.condition == "all" else [args.condition]
    results = run_full_experiment(conditions, max_tasks=args.max_tasks, dry_run=args.dry_run)

    out_path = args.output or str(RESULTS_DIR / f"p2_causal_format_{time.strftime('%Y%m%d-%H%M%S')}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")
