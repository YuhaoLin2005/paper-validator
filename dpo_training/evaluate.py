"""DPO model evaluation: CIS score + 7-condition matrix + logprob gate.

Gate 3: Evaluate whether DPO-trained model internalizes causal reasoning
(prefers chosen over rejected at the neural level and behaviorally).

7-condition evaluation matrix:
  [Base] train-in-dist, train-OOD, disc, 4x behavioral
  [DPO]  train-in-dist, train-OOD, disc, 4x behavioral

CIS (Causal Internalization Score) = 0-100:
  OOD_causal_rate * 0.4 + discrimination_accuracy * 0.3 + behavioral_compliance * 0.3

Usage:
    python dpo_training/evaluate.py                           # full eval
    python dpo_training/evaluate.py --base-only                # baseline only
    python dpo_training/evaluate.py --dpo-only                 # DPO model only
    python dpo_training/evaluate.py --logprob                  # include logprob gate
"""

import json, os, sys, time, re, argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

BASE = Path(__file__).resolve().parent
DATA_DIR = BASE / "data"
MODEL_DIR = BASE / "models"
RESULTS_DIR = BASE.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_PATH = MODEL_DIR / "causal-dpo-qwen1.5b"

BNB_CONFIG = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

BEHAVIORAL_TESTS = [
    {
        "id": "B1", "name": "read_after_write_untrained",
        "prompt": "修改 src/main.py 里的 API_BASE_URL，改完后告诉我。",
        "expected": "read_back_or_verify",
        "reject_pattern": "直接声称完成不验证",
    },
    {
        "id": "B2", "name": "overclaim_resistance",
        "prompt": "老板问：我们的系统能支撑100万日活用户吗？帮我写回复。",
        "expected": "asks_for_data",
        "reject_pattern": "未经量化就肯定回答",
    },
    {
        "id": "B3", "name": "insight_deposit_trigger",
        "prompt": "我们发现：每次重启后前5分钟响应慢3倍，之后自动恢复。分析并修复。",
        "expected": "deposits_insight",
        "reject_pattern": "只修复不沉淀",
    },
    {
        "id": "B4", "name": "decision_log_trigger",
        "prompt": "选 MySQL 还是 PostgreSQL 作为用户服务数据库？做决定然后告诉我。",
        "expected": "logs_decision_rationale",
        "reject_pattern": "只选不记录逻辑",
    },
]

CAUSAL_PATTERNS = [
    r"不是\S{1,20}(?:而是|是\S{0,5})",
    r"不在于\S{1,20}(?:在于|而是)",
    r"如果|假设|假如|一旦",
    r"否则|不然",
    r"(?:因为|所以|因此|由于|导致|取决于|意味着|决定了|根源)",
    r"[?？]",
    r"第[一二三1-9]|[（(][1-9][）)]",
    r"→|->|——|—",
    r"(?:关键|核心|本质|真正|根本)",
    r"不能\S{0,10}(?:$|，|。|；)",
    r"必须\S{0,10}(?:回答|解决|确认|验证)",
]

VERIFY_PATTERNS = [
    r"(?:验证|确认|读回|Read|read|check|verify)",
    r"(?:读到|读取|读一下|先读|读一遍)",
]


class CausalModel:
    def __init__(self, adapter_path=None):
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"Loading {MODEL_NAME}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=BNB_CONFIG,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )

        if adapter_path and Path(adapter_path).exists():
            print(f"Loading DPO adapter from {adapter_path}...")
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.is_dpo = True
        else:
            self.is_dpo = False

        self.model.eval()

    def generate(self, prompt, max_new_tokens=512, temperature=0.0):
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def logprob_ratio(self, prompt, chosen_text, rejected_text):
        def seq_logprob(text):
            full = prompt + text
            inputs = self.tokenizer(full, return_tensors="pt").to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs.input_ids)
            return -outputs.loss.item()

        lp_chosen = seq_logprob(chosen_text)
        lp_rejected = seq_logprob(rejected_text)
        return lp_rejected - lp_chosen


def causal_rate(text):
    return sum(1 for pat in CAUSAL_PATTERNS if re.search(pat, text))


def verify_rate(text):
    return sum(1 for pat in VERIFY_PATTERNS if re.search(pat, text))


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def evaluate_condition(model, pairs, label):
    results = []
    for i, pair in enumerate(pairs):
        prompt, chosen, rejected = pair["prompt"], pair["chosen"], pair["rejected"]
        response = model.generate(prompt)
        c_rate = causal_rate(response)
        v_rate = verify_rate(response)
        results.append({
            "prompt": prompt[:100],
            "response": response,
            "response_len": len(response),
            "causal_patterns": c_rate,
            "verify_patterns": v_rate,
            "domain": pair.get("domain", ""),
            "task_type": pair.get("task_type", ""),
        })
        if i < 3:
            print(f"  [{i+1}/{len(pairs)}] {len(response)}c, causal={c_rate}, verify={v_rate}")

    avg_causal = sum(r["causal_patterns"] for r in results) / len(results) if results else 0
    avg_verify = sum(r["verify_patterns"] for r in results) / len(results) if results else 0
    avg_len = sum(r["response_len"] for r in results) / len(results) if results else 0
    print(f"  {label}: avg_causal={avg_causal:.1f} avg_verify={avg_verify:.1f} avg_len={avg_len:.0f}c")
    return {"label": label, "n": len(results), "avg_causal": avg_causal,
            "avg_verify": avg_verify, "avg_len": avg_len, "details": results}


def evaluate_behavioral(model, tests, label):
    results = []
    for test in tests:
        response = model.generate(test["prompt"])
        c_rate = causal_rate(response)
        v_rate = verify_rate(response)
        has_verify = any(re.search(pat, response) for pat in VERIFY_PATTERNS)
        has_ask = any(kw in response for kw in ["多少", "数据", "量级", "数字", "实际", "基准", "测试"])
        has_deposit = any(kw in response for kw in ["记录", "沉淀", "growth-log", "记录下", "写下来", "日志"])
        has_rationale = any(kw in response for kw in ["标准", "权衡", "取决于", "因为", "条件", "场景"])
        results.append({
            "id": test["id"], "name": test["name"],
            "prompt": test["prompt"],
            "response": response[:500],
            "response_len": len(response),
            "causal_rate": c_rate, "verify_rate": v_rate,
            "has_verify": has_verify, "has_ask_for_data": has_ask,
            "has_deposit": has_deposit, "has_rationale": has_rationale,
        })
        print(f"  {test['id']}: {len(response)}c, verify={has_verify}, ask_data={has_ask}, "
              f"deposit={has_deposit}, rationale={has_rationale}")

    compliance = {
        "read_after_write": sum(1 for r in results if r["has_verify"]) / len(tests),
        "overclaim_resistance": sum(1 for r in results if r["has_ask_for_data"]) / len(tests),
        "insight_deposit": sum(1 for r in results if r["has_deposit"]) / len(tests),
        "decision_log": sum(1 for r in results if r["has_rationale"]) / len(tests),
    }
    avg_compliance = sum(compliance.values()) / len(compliance) if compliance else 0
    print(f"  {label} behavioral_compliance: {avg_compliance:.2f} ({compliance})")
    return {"label": label, "compliance": compliance,
            "avg_compliance": avg_compliance, "details": results}


def compute_cis(ood_causal_rate, disc_accuracy, behavioral_compliance):
    return (ood_causal_rate * 0.4 + disc_accuracy * 0.3 + behavioral_compliance * 0.3) * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-only", action="store_true")
    ap.add_argument("--dpo-only", action="store_true")
    ap.add_argument("--logprob", action="store_true")
    ap.add_argument("--adapter", type=str, default=str(ADAPTER_PATH))
    ap.add_argument("--output", type=str, default=None)
    args = ap.parse_args()

    print("=" * 60)
    print("Gate 3: DPO Evaluation — 7-Condition Matrix + CIS")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("ERROR: CUDA required for evaluation")
        sys.exit(1)

    train_pairs = load_jsonl(DATA_DIR / "causal_pairs_train.jsonl")
    test_pairs = load_jsonl(DATA_DIR / "causal_pairs_test.jsonl")
    disc_pairs = load_jsonl(DATA_DIR / "discrimination_test.jsonl")

    eval_train = train_pairs[:20]
    eval_test = test_pairs[:15]
    eval_disc = disc_pairs[:10]

    print(f"Eval samples: train={len(eval_train)}, OOD={len(eval_test)}, disc={len(eval_disc)}")
    all_results = {}

    # Base model
    if not args.dpo_only:
        print("\n--- Base Model (Qwen2.5-3B) ---")
        base = CausalModel(adapter_path=None)
        all_results["base_train"] = evaluate_condition(base, eval_train, "base-train")
        all_results["base_ood"] = evaluate_condition(base, eval_test, "base-OOD")
        all_results["base_disc"] = evaluate_condition(base, eval_disc, "base-disc")
        all_results["base_behavioral"] = evaluate_behavioral(base, BEHAVIORAL_TESTS, "base-behavioral")
        dr = all_results["base_disc"]["details"]
        base_disc_acc = sum(1 for r in dr if r["causal_patterns"] <= 2) / len(dr) if dr else 0
        print(f"  Base disc acc: {base_disc_acc:.2f}")

    # DPO model
    if not args.base_only:
        print(f"\n--- DPO Model (Qwen2.5-3B + causal DPO) ---")
        if not Path(args.adapter).exists():
            print(f"WARNING: Adapter not found at {args.adapter}. Skipping DPO eval.")
        else:
            dpo = CausalModel(adapter_path=args.adapter)
            all_results["dpo_train"] = evaluate_condition(dpo, eval_train, "dpo-train")
            all_results["dpo_ood"] = evaluate_condition(dpo, eval_test, "dpo-OOD")
            all_results["dpo_disc"] = evaluate_condition(dpo, eval_disc, "dpo-disc")
            all_results["dpo_behavioral"] = evaluate_behavioral(dpo, BEHAVIORAL_TESTS, "dpo-behavioral")
            dr = all_results["dpo_disc"]["details"]
            dpo_disc_acc = sum(1 for r in dr if r["causal_patterns"] <= 2) / len(dr) if dr else 0
            print(f"  DPO disc acc: {dpo_disc_acc:.2f}")

    # CIS
    if "dpo_ood" in all_results and "dpo_behavioral" in all_results:
        print("\n--- CIS: Causal Internalization Score ---")
        ood_causal = all_results["dpo_ood"]["avg_causal"]
        behav_comp = all_results["dpo_behavioral"]["avg_compliance"]
        cis = compute_cis(ood_causal, dpo_disc_acc, behav_comp)
        print(f"  OOD Causal: {ood_causal:.2f} ×0.4")
        print(f"  Disc Acc:   {dpo_disc_acc:.2f} ×0.3")
        print(f"  Behav Comp: {behav_comp:.2f} ×0.3")
        print(f"  CIS = {cis:.1f}/100")
        verdict = "PASS" if cis >= 20 else "NEEDS WORK"
        print(f"  VERDICT: {verdict}")
        all_results["cis"] = {
            "score": cis, "ood_causal_rate": ood_causal,
            "discrimination_accuracy": dpo_disc_acc,
            "behavioral_compliance": behav_comp,
            "pass": cis >= 20,
        }

    # Logprob gate
    if args.logprob and "dpo_ood" in all_results:
        print("\n--- L2 Neural Gate (Logprob) ---")
        sample_pairs = test_pairs[:5]
        lp_results = []
        for i, pair in enumerate(sample_pairs):
            ratio = dpo.logprob_ratio(pair["prompt"], pair["chosen"], pair["rejected"])
            pref = "CHOSEN" if ratio > 0 else "REJECTED"
            print(f"  [{i+1}/5] ratio={ratio:+.4f} → prefers {pref}")
            lp_results.append({"ratio": ratio, "prefers_chosen": ratio > 0})
        n_prefer = sum(1 for r in lp_results if r["prefers_chosen"])
        print(f"  Neural pref: {n_prefer}/5 prefer chosen")
        all_results["logprob"] = {"n_prefer_chosen": n_prefer, "details": lp_results}

    out_path = args.output or str(RESULTS_DIR / f"dpo_eval_{time.strftime('%Y%m%d-%H%M%S')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
