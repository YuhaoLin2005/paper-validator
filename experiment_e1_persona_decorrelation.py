#!/usr/bin/env python3
"""
E1: Cross-Model Persona Decorrelation Experiment
=================================================
Tests Mike Czerwinski's hypothesis: does reviewer agreement track
the persona (genuine independence) or the model (costume diversity)?

Importers: Called by `python experiment_e1_persona_decorrelation.py` directly.
Standalone — not imported by any other file.

API routing:
  - DeepSeek models → DeepSeek direct API (api.deepseek.com) — cheaper
  - Other models → SiliconFlow (api.siliconflow.cn) — multi-model coverage

Output: results/e1_persona_decorrelation.json
  {experiment, pre_reg_hash, n_trials, n_errors, total_tokens,
   agreement: {within_model_diff_persona, across_model_same_persona,
               within_both, across_both},
   interpretation, results: [{trial_id, persona, model, snippet_id,
                              domain, pre_reg_hash, parsed, usage}]}

Fix history:
  v1.0 — Original: response_format JSON, single API, Qwen3.6-35B-A3B
  v1.1 — Removed response_format, added extract_json() 3-strategy fallback
  v1.2 — Switched Qwen3.6→Qwen3.5-27B, added reasoning_content handling
  v1.3 — Switched Qwen3.5→Kimi-K2.7-Code (clean JSON, cross-architecture)
         Added dual-API routing (DS direct + SF)
         Updated SF key after old key leaked
  Pre-reg: eebe2a31fb290860

Results (v1.3): 30/30 clean, 0 errors, 21,853 tokens
  Cross-model same-persona: 1.000 (n=15)
  Within-model diff-persona: 0.867 (n=30)
  Expert panel: ceiling effect (4/5 snippets unanimous), need chance-corrected
  See: growth-log/2026-07-23-e1-persona-decorrelation.md

User verbatim: "密钥给你了，你直接选模型...要先做完验证完确定严谨没问题再跑"
"""
import json, hashlib, os, subprocess, sys, time
from datetime import datetime, timezone

# DeepSeek direct: cheaper. SiliconFlow: multi-model but pricier.
DS_API_KEY = os.environ["DS_TOKEN"]
DS_API_URL = "https://api.deepseek.com/v1/chat/completions"
SF_API_KEY = os.environ["SF_TOKEN"]
SF_API_URL = "https://api.siliconflow.cn/v1/chat/completions"

def get_api(model):
    """Route DeepSeek models to direct API (cheaper), others to SiliconFlow."""
    if "deepseek" in model.lower():
        return DS_API_KEY, DS_API_URL, resolve_model(model)
    return SF_API_KEY, SF_API_URL, model

def resolve_model(model):
    """Translate SiliconFlow model IDs to provider-native IDs."""
    if model == "deepseek-ai/DeepSeek-V4-Pro":
        return "deepseek-chat"  # DeepSeek direct API name for V4 Pro
    return model
OUTPUT_FILE = "results/e1_persona_decorrelation.json"

PERSONAS = {
    "carmack": {
        "name": "John Carmack",
        "principle": "The best code is the code that runs fastest with fewest surprises.",
        "focus": "Performance, algorithmic efficiency, simplicity. Reject over-engineering. Prefer direct solutions that are fast and debuggable.",
        "red_flags": "Unnecessary abstraction layers, O(n^2) where O(n) is possible, allocations in hot paths, clever code that's not measurably faster."
    },
    "torvalds": {
        "name": "Linus Torvalds",
        "principle": "Bad programmers worry about the code. Good programmers worry about data structures and their relationships.",
        "focus": "Correctness, data flow, error handling, API design. Code must be obviously correct, not cleverly obscure. Error paths matter as much as happy paths.",
        "red_flags": "NULL/None without checking, silent error swallowing, data structures that don't match the problem, comments that describe what code does instead of why."
    },
    "knuth": {
        "name": "Donald Knuth",
        "principle": "Premature optimization is the root of all evil.",
        "focus": "Algorithmic correctness, mathematical rigor, literate style. Code should communicate intent to human readers first, machines second.",
        "red_flags": "Unstated invariants, proofs replaced by 'it works on my machine', algorithms chosen without complexity analysis, variable names that hide semantics."
    }
}

SNIPPETS = [
    {"id": "perf_quadratic", "title": "Quadratic dedup in hot path", "domain": "performance", "code": "def find_duplicates(items):\n    result = []\n    for i in range(len(items)):\n        for j in range(i+1, len(items)):\n            if items[i] == items[j] and items[i] not in result:\n                result.append(items[i])\n    return result"},
    {"id": "error_silent", "title": "Silent None propagation", "domain": "error_handling", "code": "def get_user_age(db, user_id):\n    user = db.query('SELECT age FROM users WHERE id=?', user_id)\n    if user is None:\n        return 0\n    age = user.get('age')\n    if age is None:\n        return 0\n    return int(age)"},
    {"id": "naming_obfuscated", "title": "Obfuscated variable names", "domain": "readability", "code": "def process(d, m, y):\n    a = [31,28,31,30,31,30,31,31,30,31,30,31]\n    if y%4==0 and (y%100!=0 or y%400==0):\n        a[1]=29\n    if m<1 or m>12:\n        return -1\n    if d<1 or d>a[m-1]:\n        return -1\n    total=d\n    for i in range(m-1):\n        total+=a[i]\n    return total"},
    {"id": "offbyone_boundary", "title": "Binary search boundary bug", "domain": "correctness", "code": "def binary_search(arr, target):\n    left, right = 0, len(arr)\n    while left < right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid\n        else:\n            right = mid\n    return -1"},
    {"id": "god_function", "title": "Function doing too many things", "domain": "architecture", "code": "def process_order(order_id, user_id, items, discount_code, shipping_addr):\n    if not items:\n        raise ValueError('No items')\n    for item in items:\n        stock = db.get_stock(item['sku'])\n        if stock < item['qty']:\n            return {'error': f'Out of stock: {item[\"sku\"]}'}\n    total = sum(i['price']*i['qty'] for i in items)\n    if discount_code:\n        d = db.get_discount(discount_code)\n        if d and d['valid']:\n            total *= (1 - d['pct']/100)\n    if not payment_gateway.charge(user_id, total):\n        return {'error': 'Payment failed'}\n    for item in items:\n        db.deduct_stock(item['sku'], item['qty'])\n    email.send(user_id, 'Order confirmed', f'Total: ${total}')\n    analytics.track('order_completed', {'user': user_id, 'total': total})\n    return {'status': 'ok', 'total': total}"}
]

SCORING_PROMPT = """
Review the code above according to your principles.

You MUST output exactly a JSON object with these fields, nothing else:
{"verdict": "APPROVE"|"REJECT"|"APPROVE_WITH_NOTES", "critical_issues": [], "score_clarity": 1-5, "score_correctness": 1-5, "score_efficiency": 1-5, "score_maintainability": 1-5, "key_observation": "one sentence"}

Do NOT wrap in markdown code blocks. Do NOT add explanation. Output pure JSON only.
"""

import re

def extract_json(text):
    """Try multiple strategies to extract JSON from model output."""
    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except:
        pass
    # Strategy 2: extract from ```json ... ``` block
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass
    # Strategy 3: find first { ... } pair
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return None

def call_model(model, system_prompt, user_prompt, max_tokens=None):
    # Thinking models need extra tokens for reasoning + output
    if max_tokens is None:
        max_tokens = 2000 if "qwen" in model.lower() else 600
    api_key, api_url, api_model = get_api(model)
    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0
    }
    try:
        result = subprocess.run(
            ["curl", "-s", api_url,
             "-H", f"Authorization: Bearer {api_key}",
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace"
        )
        resp = json.loads(result.stdout)
        if "choices" in resp and len(resp["choices"]) > 0:
            msg = resp["choices"][0]["message"]
            content = msg.get("content", "") or ""
            # Thinking models: combine reasoning_content + content for JSON extraction
            reasoning = msg.get("reasoning_content", "") or ""
            combined = content + "\n" + reasoning if reasoning else content
            parsed = extract_json(combined)
            if parsed:
                return {"ok": True, "parsed": parsed, "usage": resp.get("usage", {})}
            else:
                return {"ok": True, "parsed": {"verdict": "PARSE_ERROR"},
                        "usage": resp.get("usage", {}),
                        "raw": combined[:300],
                        "finish_reason": resp["choices"][0].get("finish_reason", "?")}
        return {"ok": False, "error": str(resp)[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def main():
    os.makedirs("results", exist_ok=True)
    models = ["deepseek-ai/DeepSeek-V4-Pro", "moonshotai/Kimi-K2.7-Code"]

    # Pre-registration hash
    prereg = json.dumps({
        "experiment": "e1_persona_decorrelation",
        "design": "3 personas x 2 models x 5 snippets = 30 calls",
        "personas": list(PERSONAS.keys()),
        "models": models,
        "snippet_ids": [s["id"] for s in SNIPPETS],
        "primary_hypothesis": "cross-model same-persona agreement > same-model diff-persona agreement => persona drives judgment",
        "null_hypothesis": "same-model agreement > cross-model => model drives judgment (costume diversity)"
    }, sort_keys=True)
    prereg_hash = hashlib.sha256(prereg.encode()).hexdigest()[:16]
    print(f"PRE-REG: {prereg_hash}")

    results = []
    trial_id = 0
    total_tk = 0

    for s in SNIPPETS:
        for pkey, p in PERSONAS.items():
            for model in models:
                trial_id += 1
                label = f"[{trial_id}/30] {pkey[:6]} @ {model.split('/')[-1][:15]} → {s['id']}"
                print(f"{label}", end=" ", flush=True)

                sp = f"You are {p['name']} conducting a code review. Your principle: {p['principle']} Your focus: {p['focus']} Red flags: {p['red_flags']}"
                up = f"Review:\n{s['code']}\n\n{SCORING_PROMPT}"

                r = call_model(model, sp, up)
                entry = {
                    "trial_id": trial_id, "persona": pkey,
                    "model": model, "snippet_id": s["id"],
                    "domain": s["domain"], "pre_reg_hash": prereg_hash
                }
                if r["ok"]:
                    entry["parsed"] = r["parsed"]
                    entry["usage"] = r["usage"]
                    tk = r["usage"].get("total_tokens", 0)
                    total_tk += tk
                    v = r["parsed"].get("verdict", "?")
                    print(f"→ {v} ({tk}tk)")
                else:
                    entry["error"] = r.get("error", "?")
                    print(f"→ ERR")
                results.append(entry)
                time.sleep(0.3)

    # Agreement analysis
    def pairs_by_condition(filter_fn):
        vals = []
        for i in range(len(results)):
            for j in range(i+1, len(results)):
                if not filter_fn(results[i], results[j]):
                    continue
                ri, rj = results[i], results[j]
                if ri["snippet_id"] != rj["snippet_id"]:
                    continue
                va = ri.get("parsed",{}).get("verdict") == rj.get("parsed",{}).get("verdict")
                vals.append({"verdict_agree": va, "pi": i, "pj": j})
        if not vals:
            return {"n": 0, "rate": 0}
        return {"n": len(vals), "rate": sum(1 for v in vals if v["verdict_agree"])/len(vals)}

    within_model = pairs_by_condition(lambda a,b: a["model"]==b["model"] and a["persona"]!=b["persona"])
    across_model = pairs_by_condition(lambda a,b: a["model"]!=b["model"] and a["persona"]==b["persona"])
    within_both = pairs_by_condition(lambda a,b: a["model"]==b["model"] and a["persona"]==b["persona"])
    across_both = pairs_by_condition(lambda a,b: a["model"]!=b["model"] and a["persona"]!=b["persona"])

    if across_model["rate"] > within_model["rate"]:
        interpretation = f"Persona drives judgment: cross-model same-persona ({across_model['rate']:.2f}) > same-model diff-persona ({within_model['rate']:.2f}). Expert board IS genuinely independent across architectures."
    elif within_model["rate"] > across_model["rate"]:
        interpretation = f"Model drives judgment: same-model agreement ({within_model['rate']:.2f}) > cross-model same-persona ({across_model['rate']:.2f}). Mike was right — diversity is costume."
    else:
        interpretation = f"Inconclusive: {within_model['rate']:.2f} ≈ {across_model['rate']:.2f}. Need more models/trials."

    output = {
        "experiment": "e1_persona_decorrelation",
        "pre_reg_hash": prereg_hash,
        "n_trials": len(results),
        "n_errors": sum(1 for r in results if "error" in r),
        "total_tokens": total_tk,
        "agreement": {
            "within_model_diff_persona": within_model,
            "across_model_same_persona": across_model,
            "within_both": within_both,
            "across_both": across_both
        },
        "interpretation": interpretation,
        "results": results
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n=== RESULTS ===")
    print(f"Trials: {len(results)}, Errors: {output['n_errors']}, Tokens: {total_tk}")
    print(f"Within-model diff-persona: {within_model['rate']:.3f} (n={within_model['n']})")
    print(f"Across-model same-persona: {across_model['rate']:.3f} (n={across_model['n']})")
    print(f"Within both: {within_both['rate']:.3f}, Across both: {across_both['rate']:.3f}")
    print(f"\n{interpretation}")

if __name__ == "__main__":
    main()
