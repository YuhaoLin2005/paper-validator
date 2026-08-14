#!/usr/bin/env python3
"""
E1b-v2: Current-model drift re-test (SiliconFlow, both models, split schedule).

WHY: E1b (2026-07-23) used DeepSeek-V4-Pro snapshot + Kimi K2.7. DeepSeek V4 is
now the official release; the user's runtime model is V4-Flash. Re-test whether
the E1b structural conclusion (persona does NOT carry agreement across models;
agreement tracks the base model) survives on CURRENT official models.

DESIGN: identical to E1b (14 snippets x 3 personas x 2 models + no-persona
control) so every stat is apples-to-apples vs results/e1b_cross_model.json.
  - DS arm  = deepseek-v4-flash via OFFICIAL DeepSeek API (api.deepseek.com),
              key = own DeepSeek key (ANTHROPIC_API_KEY/DEEPSEEK_API_KEY/DS_TOKEN
              -> ~/.claude/.ds_token). The SF key is NOT allowed for DeepSeek.
              (overridable via E1B_V2_DS_MODEL, e.g. deepseek-v4-pro)
  - Kimi arm = moonshotai/Kimi-K2.7-Code via SiliconFlow (the SF key's scope)

KEY ROUTING (user rule 2026-08-14): SiliconFlow key = OTHER models ONLY;
DeepSeek models use the user's own DeepSeek key on api.deepseek.com.

SPLIT (money: DeepSeek after Beijing 18:00):
  --model-set kimi     -> results/e1b_v2_partial_kimi.json   (run NOW)
  --model-set ds       -> results/e1b_v2_partial_ds.json     (run after 18:00)
  --model-set both     -> run both, write both partials
  --model-set analyze  -> merge partials -> results/e1b_v2_current.json
                          + side-by-side table vs 7-23 baseline
  --smoke [ds|kimi]    -> 1 call (default kimi) (key/model/parse check)

PROTOCOL (v2, differs from 7-23 baseline): max_tokens 2000 + content-first parse.
Baseline 7-23 ran max_tokens 600 + combined(content+reasoning) parse and was clean
for DS because that snapshot had NO reasoning_content. Official V4-flash is a
reasoning model: 600-cap left content empty (29/56 PARSE_ERROR), and the combined
parse's greedy {.*} regex breaks when reasoning text contains braces (baseline's
9 Kimi parse errors = same bug). v2 fixes both. See call_model._best_parse.

Keys: SF_TOKEN -> ~/.claude/.sf_token (OTHER models) | ANTHROPIC_API_KEY/
DEEPSEEK_API_KEY/DS_TOKEN -> ~/.claude/.ds_token (DeepSeek, official API)
Reuses E1b snippets/personas/prompts/stats (imported) — provenance preserved.

User verbatim (2026-08-14): "先给你硅基流动的ak...你先跑其他模型，deepseek到北京
时间18:00时候再跑这样省钱。所以设计要好同时要兼顾省钱"
"""
import json, os, re, subprocess, sys, time, random
from pathlib import Path

# import guards: the E1b module reads these at import time (its own DS/SF keys).
os.environ.setdefault("DS_TOKEN", "dummy-not-used")
os.environ.setdefault("SF_TOKEN", "dummy-not-used")
from experiment_e1b_cross_model import (
    SNIPPETS, PERSONAS, SCORING_PROMPT, NO_PERSONA_PROMPT,
    fleiss_kappa, bootstrap_kappa_ci, extract_json,
)

DS_URL = "https://api.deepseek.com/v1/chat/completions"   # official DeepSeek API
SF_URL = "https://api.siliconflow.cn/v1/chat/completions" # other models only
DS_MODEL = os.environ.get("E1B_V2_DS_MODEL", "deepseek-v4-flash")  # official model id
KIMI_MODEL = "moonshotai/Kimi-K2.7-Code"
OUTPUTS = {"kimi": "results/e1b_v2_partial_kimi.json", "ds": "results/e1b_v2_partial_ds.json"}
MODEL_SET = {"kimi": KIMI_MODEL, "ds": DS_MODEL}


def sf_token():
    env = os.environ.get("SF_TOKEN") or os.environ.get("SF_API_KEY")
    if env and env != "dummy-not-used":
        return env
    p = Path.home() / ".claude" / ".sf_token"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    raise RuntimeError("No SF key. Set SF_TOKEN env or write ~/.claude/.sf_token")


def ds_token():
    for var in ("ANTHROPIC_API_KEY", "DEEPSEEK_API_KEY", "DS_TOKEN"):
        v = os.environ.get(var, "")
        if v and v != "dummy-not-used":
            return v
    p = Path.home() / ".claude" / ".ds_token"
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    raise RuntimeError("No DeepSeek key. Set ANTHROPIC_API_KEY/DEEPSEEK_API_KEY/DS_TOKEN or write ~/.claude/.ds_token")


# Per-arm max_tokens: official DS V4-flash is a heavy reasoning model (chain-of-thought
# can be 10k+ chars before the final JSON — some snippets never conclude; those stay
# residual PARSE_ERRORs, documented). Kimi's content is short; 1500 is headroom.
MAX_TOKENS = {"ds": int(os.environ.get("E1B_V2_DS_MAX_TOKENS", "4000")),
              "kimi": int(os.environ.get("E1B_V2_KIMI_MAX_TOKENS", "1500"))}


def _best_parse(content, reasoning):
    """Final-answer JSON lives in `content`; fall back to reasoning, then combined.
    Parsing content+reasoning together breaks on reasoning text containing braces
    (greedy {.*} spans the JSON into the reasoning tail) — extract content first.
    """
    combined = content + "\n" + reasoning if reasoning else content
    for cand in (content, reasoning, combined):
        if not cand:
            continue
        p = extract_json(cand)
        if p:
            return p, cand
    return None, None


def call_model(model, system_prompt, user_prompt, max_tokens=2000):
    is_ds = "deepseek" in model.lower()
    url = DS_URL if is_ds else SF_URL
    key = ds_token() if is_ds else sf_token()
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": max_tokens, "temperature": 0,
    }
    try:
        r = subprocess.run(
            ["curl", "-s", url,
             "-H", f"Authorization: Bearer {key}",
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=180,
            encoding="utf-8", errors="replace",
        )
        resp = json.loads(r.stdout)
        if "choices" in resp and len(resp["choices"]) > 0:
            msg = resp["choices"][0]["message"]
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            parsed, src = _best_parse(content, reasoning)
            usage = resp.get("usage", {})
            diag = {"content": content, "reasoning_len": len(reasoning),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "src": "content" if src == content else ("reasoning" if src == reasoning else "combined" if src else "none")}
            if parsed:
                return {"ok": True, "parsed": parsed, "usage": usage, "diag": diag}
            return {"ok": True, "parsed": {"verdict": "PARSE_ERROR"},
                    "usage": usage, "diag": diag}
        return {"ok": False, "error": str(resp)[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def build_persona_system(persona):
    p = PERSONAS[persona]
    return (f"You are {p['name']} conducting a code review. Your principle: {p['principle']} "
            f"Your focus: {p['focus']} Red flags: {p['red_flags']}")


def run_model_set(mset):
    model = MODEL_SET[mset]
    os.makedirs("results", exist_ok=True)
    results, total_tk, tid = [], 0, 0
    for s in SNIPPETS:
        for pkey in PERSONAS:
            tid += 1
            sp = build_persona_system(pkey)
            up = f"Title: {s['title']}\n\nCode:\n{s['code']}\n\n{SCORING_PROMPT}"
            r = call_model(model, sp, up, max_tokens=MAX_TOKENS[mset])
            entry = {"trial_id": tid, "condition": "persona", "persona": pkey,
                     "model": model, "snippet_id": s["id"], "domain": s["domain"]}
            if r["ok"]:
                entry["parsed"] = r["parsed"]
                entry["usage"] = r["usage"]
                if "diag" in r:
                    entry["diag"] = r["diag"]
                total_tk += r["usage"].get("total_tokens", 0)
            else:
                entry["error"] = r.get("error", "?")
            results.append(entry)
            time.sleep(0.2)
    no_sp = "You are an experienced software engineer conducting a code review. Be objective and balanced."
    for s in SNIPPETS:
        tid += 1
        up = f"Title: {s['title']}\n\nCode:\n{s['code']}\n\n{NO_PERSONA_PROMPT}"
        r = call_model(model, no_sp, up, max_tokens=MAX_TOKENS[mset])
        entry = {"trial_id": tid, "condition": "control", "persona": "none",
                 "model": model, "snippet_id": s["id"], "domain": s["domain"]}
        if r["ok"]:
            entry["parsed"] = r["parsed"]
            entry["usage"] = r["usage"]
            total_tk += r["usage"].get("total_tokens", 0)
        else:
            entry["error"] = r.get("error", "?")
        results.append(entry)
        time.sleep(0.2)
    out = {"experiment": "e1b_v2", "model_set": mset, "model": model,
           "n_trials": len(results),
           "n_errors": sum(1 for r in results if "error" in r),
           "total_tokens": total_tk, "results": results}
    with open(OUTPUTS[mset], "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[e1b_v2:{mset}] {len(results)} trials, {total_tk} tokens, {sum(1 for r in results if 'error' in r)} errors -> {OUTPUTS[mset]}")
    return out


# ── Analysis (mirrors E1b main(), on merged partials) ─────────────────────
def analyze():
    merged = []
    for mset in ("kimi", "ds"):
        p = OUTPUTS[mset]
        if not os.path.exists(p):
            print(f"!! missing {p} — run --model-set {mset} first"); return
        merged += json.load(open(p, encoding="utf-8"))["results"]

    verdicts_set = set()
    for r in merged:
        if "parsed" in r and "verdict" in r["parsed"]:
            verdicts_set.add(r["parsed"]["verdict"])
    verdicts_set.add("PARSE_ERROR")
    cat_list = sorted(verdicts_set)
    cat_idx = {c: i for i, c in enumerate(cat_list)}

    def build_kappa_matrix(condition_filter):
        matrix = []
        for s in SNIPPETS:
            row = [0] * len(cat_list)
            relevant = [r for r in merged if r["snippet_id"] == s["id"] and condition_filter(r) and "parsed" in r]
            for r in relevant:
                v = r["parsed"].get("verdict", "PARSE_ERROR")
                row[cat_idx[v]] += 1
            if sum(row) > 0:
                matrix.append(row)
        return matrix

    persona_matrix = build_kappa_matrix(lambda r: r["condition"] == "persona")
    ds_matrix = build_kappa_matrix(lambda r: r["condition"] == "persona" and "deepseek" in r["model"].lower())
    kimi_matrix = build_kappa_matrix(lambda r: r["condition"] == "persona" and "kimi" in r["model"].lower())
    control_matrix = build_kappa_matrix(lambda r: r["condition"] == "control")

    stats = {}
    for name, mat in [("persona_6raters", persona_matrix), ("ds_3personas", ds_matrix),
                      ("kimi_3personas", kimi_matrix), ("control_2models", control_matrix)]:
        if mat and len(mat) > 0 and sum(mat[0]) >= 2:
            k, lo, hi = bootstrap_kappa_ci(mat)
            stats[name] = {"kappa": round(k, 4), "ci95_lower": round(lo, 4),
                           "ci95_upper": round(hi, 4), "n_subjects": len(mat)}
        else:
            stats[name] = {"kappa": None, "error": "insufficient data"}

    def verdict_agreement(fa, fb):
        ag, tot = 0, 0
        for s in SNIPPETS:
            ra = [r for r in merged if r["snippet_id"] == s["id"] and fa(r)]
            rb = [r for r in merged if r["snippet_id"] == s["id"] and fb(r)]
            for a in ra:
                for b in rb:
                    if a.get("parsed", {}).get("verdict") == b.get("parsed", {}).get("verdict"):
                        ag += 1
                    tot += 1
        return ag / tot if tot else 0, tot

    cross = {}
    for pkey in PERSONAS:
        rate, n = verdict_agreement(
            lambda r, p=pkey: r["condition"] == "persona" and r["persona"] == p and "deepseek" in r["model"].lower(),
            lambda r, p=pkey: r["condition"] == "persona" and r["persona"] == p and "kimi" in r["model"].lower(),
        )
        cross[pkey] = {"agreement_rate": round(rate, 4), "n_pairs": n}

    within = {}
    for m in set(r["model"] for r in merged if "model" in r):
        m_short = "DS" if "deepseek" in m.lower() else "Kimi"
        pers = [r for r in merged if r["condition"] == "persona" and r["model"] == m]
        ag, tot = 0, 0
        for s in SNIPPETS:
            trials = [r for r in pers if r["snippet_id"] == s["id"]]
            for i in range(len(trials)):
                for j in range(i + 1, len(trials)):
                    if trials[i].get("parsed", {}).get("verdict") == trials[j].get("parsed", {}).get("verdict"):
                        ag += 1
                    tot += 1
        within[m_short] = {"agreement_rate": round(ag / tot, 4) if tot else 0, "n_pairs": tot}

    total_tk = sum(json.load(open(OUTPUTS[m], encoding="utf-8"))["total_tokens"] for m in ("kimi", "ds"))
    out = {
        "experiment": "e1b_v2_current",
        "models": [KIMI_MODEL, DS_MODEL],
        "n_snippets": len(SNIPPETS),
        "n_trials": len(merged),
        "n_errors": sum(1 for r in merged if "error" in r),
        "total_tokens": total_tk,
        "categories": cat_list,
        "fleiss_kappa": stats,
        "cross_model_same_persona": cross,
        "within_model_diff_persona": within,
        "results": merged,
    }
    with open("results/e1b_v2_current.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    base = {}
    bpath = "results/e1b_cross_model.json"
    if os.path.exists(bpath):
        b = json.load(open(bpath, encoding="utf-8"))
        base = {"kappa": b["fleiss_kappa"], "cross": b["cross_model_same_persona"],
                "within": b["within_model_diff_persona"]}

    def fmt(s):
        return f"{s['kappa']:.3f} [{s.get('ci95_lower','-'):.3f},{s.get('ci95_upper','-'):.3f}]" if s.get("kappa") is not None else "n/a"

    pe_by_model = {}
    for r in merged:
        m = r["model"]
        pe_by_model.setdefault(m, [0, 0])
        pe_by_model[m][1] += 1
        if r.get("parsed", {}).get("verdict") == "PARSE_ERROR":
            pe_by_model[m][0] += 1
    print("\n=== E1b-v2 (current) vs E1b (2026-07-23) ===")
    for m, (pe, tot) in pe_by_model.items():
        print(f"  PARSE_ERROR {m}: {pe}/{tot} ({100*pe/tot:.0f}%)")
    print(f"{'metric':<26}{'7-23':<26}{'current'}")
    for name in ["persona_6raters", "ds_3personas", "kimi_3personas", "control_2models"]:
        old = base.get("kappa", {}).get(name, {})
        cur = stats.get(name, {})
        print(f"  {name:<24}{fmt(old) if old else 'n/a':<26}{fmt(cur)}")
    print("cross-model same-persona:")
    for pkey in PERSONAS:
        o = base.get("cross", {}).get(pkey, {}); c = cross.get(pkey, {})
        print(f"  {pkey:<10}  old={o.get('agreement_rate','n/a')}  new={c.get('agreement_rate','n/a')}")
    print("within-model diff-persona:")
    for m_short in ["DS", "Kimi"]:
        o = base.get("within", {}).get(m_short, {}); c = within.get(m_short, {})
        print(f"  {m_short:<10}  old={o.get('agreement_rate','n/a')}  new={c.get('agreement_rate','n/a')}")
    print(f"\nSaved results/e1b_v2_current.json ({len(merged)} trials, {total_tk} tokens)")


def smoke(model=None):
    if model in ("ds", "deepseek"):
        model = DS_MODEL
    elif model in ("kimi", "sf"):
        model = KIMI_MODEL
    model = model or KIMI_MODEL
    s = SNIPPETS[0]
    sp = build_persona_system("carmack")
    up = f"Title: {s['title']}\n\nCode:\n{s['code']}\n\n{SCORING_PROMPT}"
    r = call_model(model, sp, up)
    print(json.dumps(r, ensure_ascii=False)[:900])


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--smoke" in args:
        mi = args.index("--smoke")
        smoke(args[mi + 1] if mi + 1 < len(args) and not args[mi + 1].startswith("--") else None)
    elif "--model-set" in args:
        mset = args[args.index("--model-set") + 1]
        if mset == "both":
            for m in ("kimi", "ds"):
                run_model_set(m)
        elif mset == "analyze":
            analyze()
        elif mset in MODEL_SET:
            run_model_set(mset)
        else:
            print("model-set: kimi|ds|both|analyze")
    else:
        print("Usage: python experiment_e1b_v2_current.py [--smoke | --model-set kimi|ds|both|analyze]")
