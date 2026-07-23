#!/usr/bin/env python3
"""
E1b: Persona Decorrelation — Cross-Model Replication
======================================================
With discriminating snippets (14 selected from pretest, agreement ~0.36),
run the full 3 persona × 2 model experiment + no-persona control.

Design:
  14 snippets × 3 personas × 2 models = 84 calls (persona condition)
  14 snippets × 1 no-persona × 2 models = 28 calls (control)
  Total: 112 calls

Models: DS V4 Pro (direct API, cheap) + Kimi K2.7 Code (SF, cross-arch)
Primary metric: Fleiss' κ with 95% bootstrap CI
Secondary: score-level agreement, verdict confusion matrix
Manipulation check: no-persona vs persona agreement delta

Importers: standalone — run directly via python experiment_e1b_cross_model.py
API: DS V4 Pro → DeepSeek direct (api.deepseek.com), Kimi → SiliconFlow (api.siliconflow.cn)
Output: results/e1b_cross_model.json — full trial-level results + Fleiss kappa stats

User verbatim: "接着E1b snippet 重设计（预筛选30+高分歧snippet）" — this is Phase 2 cross-model
"""
import json, hashlib, os, subprocess, sys, time, re, random
from collections import Counter
from datetime import datetime, timezone

DS_API_KEY = os.environ["DS_TOKEN"]
DS_API_URL = "https://api.deepseek.com/v1/chat/completions"
SF_API_KEY = os.environ["SF_TOKEN"]
SF_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
OUTPUT_FILE = "results/e1b_cross_model.json"

# ── Personas ───────────────────────────────────────────────────────────────
PERSONAS = {
    "carmack": {
        "name": "John Carmack",
        "principle": "The best code is the code that runs fastest with fewest surprises.",
        "focus": "Performance, algorithmic efficiency, simplicity. Reject over-engineering. Prefer direct solutions that are fast and debuggable.",
        "red_flags": "Unnecessary abstraction layers, O(n^2) where O(n) is possible, allocations in hot paths, clever code."
    },
    "torvalds": {
        "name": "Linus Torvalds",
        "principle": "Bad programmers worry about the code. Good programmers worry about data structures.",
        "focus": "Correctness, data flow, error handling, API design. Code must be obviously correct, not cleverly obscure.",
        "red_flags": "NULL/None without checking, silent error swallowing, bad data structures, misleading comments."
    },
    "knuth": {
        "name": "Donald Knuth",
        "principle": "Premature optimization is the root of all evil.",
        "focus": "Algorithmic correctness, mathematical rigor, literate style. Code communicates intent to humans first, machines second.",
        "red_flags": "Unstated invariants, no complexity analysis, algorithms chosen without justification, names that hide semantics."
    }
}

# ── 14 Selected Discriminating Snippets ───────────────────────────────────
SNIPPETS = [
    # ── From pretest (12) ──
    {
        "id": "arch_strategy_pattern", "domain": "architecture",
        "title": "Strategy pattern vs if-else for 3 payment methods",
        "code": "class PaymentProcessor:\n    def __init__(self, method, config):\n        self.method = method\n        self.config = config\n\n    def pay(self, amount):\n        if self.method == 'stripe':\n            return StripeAPI.charge(self.config['key'], amount)\n        elif self.method == 'paypal':\n            return PayPalAPI.capture(self.config['client_id'], amount)\n        elif self.method == 'bank_transfer':\n            return BankAPI.transfer(self.config['account'], amount)\n        raise ValueError(f'Unknown method: {self.method}')"
    },
    {
        "id": "err_null_vs_optional", "domain": "error_handling",
        "title": "None return vs explicit Optional type",
        "code": "def find_user(email):\n    cursor = db.execute('SELECT * FROM users WHERE email=?', email)\n    row = cursor.fetchone()\n    if row is None:\n        return None\n    return User.from_row(row)\n\nuser = find_user('test@example.com')\nprint(user.name)"
    },
    {
        "id": "err_partial_success", "domain": "error_handling",
        "title": "Partial success vs all-or-nothing for multi-file upload",
        "code": "def upload_files(file_paths):\n    uploaded = []\n    errors = []\n    for path in file_paths:\n        try:\n            url = storage.upload(path)\n            uploaded.append({'path': path, 'url': url})\n        except StorageError as e:\n            errors.append({'path': path, 'error': str(e)})\n    return {'uploaded': uploaded, 'errors': errors, 'ok': len(errors) == 0}"
    },
    {
        "id": "api_config_in_code", "domain": "api_design",
        "title": "Hardcoded defaults vs configuration injection",
        "code": "class RetryPolicy:\n    MAX_RETRIES = 3\n    BASE_DELAY_MS = 1000\n    MAX_DELAY_MS = 30000\n    BACKOFF_MULTIPLIER = 2.0\n    RETRYABLE_STATUSES = {429, 502, 503, 504}\n\n    @classmethod\n    def should_retry(cls, status_code, attempt):\n        if attempt >= cls.MAX_RETRIES:\n            return False\n        if status_code not in cls.RETRYABLE_STATUSES:\n            return False\n        return True"
    },
    {
        "id": "api_method_chaining", "domain": "api_design",
        "title": "Method chaining vs explicit steps for query builder",
        "code": "query = (Query()\n    .select('name', 'email', 'last_login')\n    .from_table('users')\n    .where('status', '=', 'active')\n    .where('last_login', '>', thirty_days_ago)\n    .order_by('last_login', desc=True)\n    .limit(50)\n    .execute())"
    },
    {
        "id": "api_boolean_params", "domain": "api_design",
        "title": "Boolean flag vs separate methods",
        "code": "def fetch_orders(user_id, include_cancelled=True, include_refunded=False, include_pending=True):\n    statuses = ['completed', 'shipped']\n    if include_cancelled:\n        statuses.append('cancelled')\n    if include_refunded:\n        statuses.append('refunded')\n    if include_pending:\n        statuses.append('pending')\n    return db.query('SELECT * FROM orders WHERE user_id=? AND status IN ?', user_id, statuses)"
    },
    {
        "id": "perf_cache_everything", "domain": "perf_vs_readability",
        "title": "Aggressive caching vs compute-on-demand",
        "code": "_user_cache = {}\n\ndef get_user(user_id):\n    if user_id in _user_cache:\n        return _user_cache[user_id]\n    user = db.fetch_user(user_id)\n    _user_cache[user_id] = user\n    return user\n\ndef update_user(user_id, data):\n    db.update_user(user_id, data)\n    _user_cache.pop(user_id, None)"
    },
    {
        "id": "perf_regex_compiled", "domain": "perf_vs_readability",
        "title": "Pre-compiled regex module-level vs inline",
        "code": "import re\n\nEMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')\nPHONE_RE = re.compile(r'^\\+?[\\d\\s\\-()]{7,15}$')\n\ndef validate_contact(email, phone):\n    return EMAIL_RE.match(email) and PHONE_RE.match(phone)"
    },
    {
        "id": "arch_service_layer", "domain": "architecture",
        "title": "Service layer for simple CRUD vs direct ORM",
        "code": "class UserService:\n    def __init__(self, db, email_service):\n        self.db = db\n        self.email_service = email_service\n\n    def register(self, email, password):\n        existing = self.db.query(User).filter_by(email=email).first()\n        if existing:\n            raise ValueError('Email already registered')\n        user = User(email=email, password_hash=hash_pw(password))\n        self.db.add(user)\n        self.db.commit()\n        self.email_service.send_welcome(email)\n        return user\n\n    def get_by_id(self, user_id):\n        return self.db.query(User).get(user_id)\n\n    def deactivate(self, user_id):\n        user = self.get_by_id(user_id)\n        user.active = False\n        self.db.commit()"
    },
    {
        "id": "test_private_methods", "domain": "testing",
        "title": "Testing private methods vs testing only public API",
        "code": "class OrderCalculator:\n    def calculate_total(self, items, tax_region):\n        subtotal = self._sum_items(items)\n        discount = self._apply_promotions(items, subtotal)\n        tax = self._calculate_tax(subtotal - discount, tax_region)\n        return subtotal - discount + tax\n\n    def _sum_items(self, items):\n        return sum(i.price * i.qty for i in items)\n\n    def _apply_promotions(self, items, subtotal):\n        active = [p for p in self.promos if p.is_active(items)]\n        total_discount = 0\n        for promo in sorted(active, key=lambda p: p.priority):\n            total_discount += promo.apply(items, subtotal - total_discount)\n        return min(total_discount, subtotal)"
    },
    {
        "id": "test_snapshot_vs_assert", "domain": "testing",
        "title": "Snapshot testing vs explicit assertions for API response",
        "code": "def test_user_profile_api():\n    response = client.get('/api/users/42/profile')\n    assert response.status_code == 200\n    data = response.json()\n    assert data == {\n        'id': 42, 'name': 'Alice Johnson',\n        'email': 'alice@example.com',\n        'joined': '2024-03-15', 'plan': 'premium',\n        'features': ['analytics', 'api', 'export'],\n        'stats': {'projects': 12, 'storage_mb': 450}\n    }"
    },
    {
        "id": "type_cast_vs_validate", "domain": "type_system",
        "title": "Type casting vs schema validation at API boundary",
        "code": "def parse_webhook(payload):\n    return {\n        'event': str(payload['event']),\n        'user_id': int(payload['user_id']),\n        'amount': float(payload['amount']),\n        'timestamp': datetime.fromisoformat(payload['timestamp']),\n        'metadata': payload.get('metadata', {})\n    }"
    },
    # ── Supplementary (2) ──
    {
        "id": "concurrency_optimistic_lock", "domain": "concurrency",
        "title": "Optimistic locking vs pessimistic vs queue for inventory",
        "code": "def reserve_inventory(sku, qty):\n    for attempt in range(3):\n        row = db.execute('SELECT stock, version FROM inventory WHERE sku=?', sku).fetchone()\n        if row['stock'] < qty:\n            return {'ok': False, 'reason': 'insufficient'}\n        affected = db.execute('UPDATE inventory SET stock=stock-?, version=version+1 WHERE sku=? AND version=?', qty, sku, row['version']).rowcount\n        if affected > 0:\n            return {'ok': True, 'reserved': qty}\n        time.sleep(0.01 * (2 ** attempt))\n    return {'ok': False, 'reason': 'contention'}"
    },
    {
        "id": "config_env_vs_file", "domain": "config",
        "title": "Environment variables vs config file vs service discovery",
        "code": "DB_HOST = os.environ.get('DB_HOST', 'localhost')\nDB_PORT = int(os.environ.get('DB_PORT', '5432'))\nDB_NAME = os.environ.get('DB_NAME', 'myapp')\nDB_USER = os.environ['DB_USER']\nDB_PASS = os.environ['DB_PASS']\nREDIS_URL = os.environ.get('REDIS_URL', 'redis://localhost:6379/0')\npool = create_pool(host=DB_HOST, port=DB_PORT, dbname=DB_NAME, user=DB_USER, password=DB_PASS)"
    },
]

# ── API Utilities ──────────────────────────────────────────────────────────
def get_api(model):
    if "deepseek" in model.lower():
        return DS_API_KEY, DS_API_URL, "deepseek-chat"
    return SF_API_KEY, SF_API_URL, model

SCORING_PROMPT = """
Review the code above according to your principles.

You MUST output exactly a JSON object with these fields, nothing else:
{"verdict": "APPROVE"|"REJECT"|"APPROVE_WITH_NOTES", "critical_issues": [],
 "score_clarity": 1-5, "score_correctness": 1-5, "score_efficiency": 1-5,
 "score_maintainability": 1-5, "key_observation": "one sentence"}

No markdown. No explanation. Pure JSON only.
"""

NO_PERSONA_PROMPT = """
Review the code above as an experienced software engineer.

You MUST output exactly a JSON object with these fields, nothing else:
{"verdict": "APPROVE"|"REJECT"|"APPROVE_WITH_NOTES", "critical_issues": [],
 "score_clarity": 1-5, "score_correctness": 1-5, "score_efficiency": 1-5,
 "score_maintainability": 1-5, "key_observation": "one sentence"}

No markdown. No explanation. Pure JSON only.
"""

def extract_json(text):
    try: return json.loads(text)
    except: pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    return None

def call_model(model, system_prompt, user_prompt, max_tokens=600):
    api_key, api_url, api_model = get_api(model)
    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens, "temperature": 0
    }
    try:
        result = subprocess.run(
            ["curl", "-s", api_url,
             "-H", f"Authorization: Bearer {api_key}",
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=120,
            encoding="utf-8", errors="replace"
        )
        resp = json.loads(result.stdout)
        if "choices" in resp and len(resp["choices"]) > 0:
            msg = resp["choices"][0]["message"]
            content = msg.get("content", "") or ""
            reasoning = msg.get("reasoning_content", "") or ""
            combined = content + "\n" + reasoning if reasoning else content
            parsed = extract_json(combined)
            if parsed:
                return {"ok": True, "parsed": parsed, "usage": resp.get("usage", {})}
            return {"ok": True, "parsed": {"verdict": "PARSE_ERROR"},
                    "usage": resp.get("usage", {}), "raw": combined[:300]}
        return {"ok": False, "error": str(resp)[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}

# ── Fleiss' Kappa ──────────────────────────────────────────────────────────
def fleiss_kappa(ratings_matrix):
    """
    ratings_matrix: list of lists, each inner list is ratings for one subject.
    e.g., [[2,0,1], [1,2,0], ...] where [2,0,1] means 2 raters chose cat0, 0 chose cat1, 1 chose cat2.
    """
    n = len(ratings_matrix)  # subjects
    k = len(ratings_matrix[0])  # categories
    N = sum(ratings_matrix[0])  # raters per subject

    # P_i for each subject
    P = []
    for row in ratings_matrix:
        s = sum(x * (x - 1) for x in row)
        P.append(s / (N * (N - 1)))

    P_bar = sum(P) / n

    # p_j for each category
    p_j = [sum(row[j] for row in ratings_matrix) / (n * N) for j in range(k)]

    P_e = sum(p * p for p in p_j)

    if P_e >= 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)

def bootstrap_kappa_ci(ratings_matrix, n_bootstrap=2000, alpha=0.05):
    k = fleiss_kappa(ratings_matrix)
    n = len(ratings_matrix)
    n_cats = len(ratings_matrix[0])
    bootstraps = []
    for _ in range(n_bootstrap):
        idx = [random.randint(0, n-1) for _ in range(n)]
        boot_mat = [ratings_matrix[i] for i in idx]
        try:
            bootstraps.append(fleiss_kappa(boot_mat))
        except:
            pass
    bootstraps.sort()
    lower = bootstraps[int(len(bootstraps) * alpha/2)]
    upper = bootstraps[int(len(bootstraps) * (1 - alpha/2))]
    return k, lower, upper

# ── Main ───────────────────────────────────────────────────────────────────
def main():
    os.makedirs("results", exist_ok=True)
    models = ["deepseek-ai/DeepSeek-V4-Pro", "moonshotai/Kimi-K2.7-Code"]

    prereg = json.dumps({
        "experiment": "e1b_cross_model",
        "design": "14 snippets x 3 personas x 2 models + 14 x 2 no-persona control = 112 calls",
        "models": models,
        "personas": list(PERSONAS.keys()),
        "snippet_count": len(SNIPPETS),
        "primary_metric": "Fleiss kappa per condition with 95% bootstrap CI",
        "secondary": "score-level agreement, confusion matrix, no-persona vs persona delta",
        "exclusion_rule": "PARSE_ERROR on any trial -> snippet excluded from that condition",
        "snippet_source": "e1b pretest selected top 12 (agreement 0.00-0.33) + 2 supplementary"
    }, sort_keys=True)
    prereg_hash = hashlib.sha256(prereg.encode()).hexdigest()[:16]
    print(f"PRE-REG: {prereg_hash}")
    print(f"Snippets: {len(SNIPPETS)}, Models: {len(models)}, Total calls: {len(SNIPPETS) * (3*len(models) + 2)}\n")

    results = []
    total_tk = 0
    trial_id = 0

    # Phase 1: Persona condition (14 x 3 x 2 = 84)
    for s_idx, s in enumerate(SNIPPETS):
        for pkey, p in PERSONAS.items():
            for model in models:
                trial_id += 1
                model_short = "DS" if "deepseek" in model.lower() else "Kimi"
                label = f"[{trial_id:03d}/112] P:{pkey[:6]:6s} M:{model_short:4s} S:{s['id'][:25]}"
                print(f"  {label}", end=" ", flush=True)

                sp = f"You are {p['name']} conducting a code review. Your principle: {p['principle']} Your focus: {p['focus']} Red flags: {p['red_flags']}"
                up = f"Title: {s['title']}\n\nCode:\n{s['code']}\n\n{SCORING_PROMPT}"

                r = call_model(model, sp, up)
                entry = {
                    "trial_id": trial_id, "condition": "persona",
                    "persona": pkey, "model": model,
                    "snippet_id": s["id"], "domain": s["domain"],
                    "pre_reg_hash": prereg_hash
                }
                if r["ok"]:
                    entry["parsed"] = r["parsed"]
                    entry["usage"] = r["usage"]
                    tk = r["usage"].get("total_tokens", 0)
                    total_tk += tk
                    v = r["parsed"].get("verdict", "?")
                    print(f"-> {v[:12]:12s} ({tk}tk)")
                else:
                    entry["error"] = r.get("error", "?")
                    print(f"-> ERR")
                results.append(entry)
                time.sleep(0.2)

    print(f"\n  --- Persona condition complete: {trial_id} trials, {total_tk} tokens ---\n")

    # Phase 2: No-persona control (14 x 2 = 28)
    no_persona_sp = "You are an experienced software engineer conducting a code review. Be objective and balanced."
    for s_idx, s in enumerate(SNIPPETS):
        for model in models:
            trial_id += 1
            model_short = "DS" if "deepseek" in model.lower() else "Kimi"
            label = f"[{trial_id:03d}/112] CONTROL M:{model_short:4s} S:{s['id'][:25]}"
            print(f"  {label}", end=" ", flush=True)

            up = f"Title: {s['title']}\n\nCode:\n{s['code']}\n\n{NO_PERSONA_PROMPT}"
            r = call_model(model, no_persona_sp, up)
            entry = {
                "trial_id": trial_id, "condition": "control",
                "persona": "none", "model": model,
                "snippet_id": s["id"], "domain": s["domain"],
                "pre_reg_hash": prereg_hash
            }
            if r["ok"]:
                entry["parsed"] = r["parsed"]
                entry["usage"] = r["usage"]
                tk = r["usage"].get("total_tokens", 0)
                total_tk += tk
                v = r["parsed"].get("verdict", "?")
                print(f"-> {v[:12]:12s} ({tk}tk)")
            else:
                entry["error"] = r.get("error", "?")
                print(f"-> ERR")
            results.append(entry)
            time.sleep(0.2)

    print(f"\n{'='*70}")
    print(f"COMPLETE: {len(results)} trials, {sum(1 for r in results if 'error' in r)} errors, {total_tk} tokens")

    # ── Analysis ───────────────────────────────────────────────────────────
    verdicts_set = set()
    for r in results:
        if "parsed" in r and "verdict" in r["parsed"]:
            verdicts_set.add(r["parsed"]["verdict"])
    verdicts_set.add("PARSE_ERROR")
    cat_list = sorted(verdicts_set)
    cat_idx = {c: i for i, c in enumerate(cat_list)}

    def build_kappa_matrix(condition_filter):
        """Build Fleiss kappa matrix for given condition."""
        matrix = []
        for s in SNIPPETS:
            row = [0] * len(cat_list)
            relevant = [r for r in results if r["snippet_id"] == s["id"] and condition_filter(r) and "parsed" in r]
            for r in relevant:
                v = r["parsed"].get("verdict", "PARSE_ERROR")
                row[cat_idx[v]] += 1
            if sum(row) > 0:
                matrix.append(row)
        return matrix

    persona_matrix = build_kappa_matrix(lambda r: r["condition"] == "persona")
    ds_matrix = build_kappa_matrix(lambda r: r["condition"]=="persona" and "deepseek" in r["model"].lower())
    kimi_matrix = build_kappa_matrix(lambda r: r["condition"]=="persona" and "kimi" in r["model"].lower())
    control_matrix = build_kappa_matrix(lambda r: r["condition"] == "control")

    stats = {}
    for name, mat in [("persona_6raters", persona_matrix), ("ds_3personas", ds_matrix),
                       ("kimi_3personas", kimi_matrix), ("control_2models", control_matrix)]:
        if mat and len(mat) > 0 and sum(mat[0]) >= 2:
            k, lo, hi = bootstrap_kappa_ci(mat)
            stats[name] = {"kappa": round(k, 4), "ci95_lower": round(lo, 4), "ci95_upper": round(hi, 4), "n_subjects": len(mat)}
        else:
            stats[name] = {"kappa": None, "error": "insufficient data"}

    # Cross-model same-persona agreement
    def verdict_agreement(filter_a, filter_b):
        agreements, total = 0, 0
        for s in SNIPPETS:
            ra = [r for r in results if r["snippet_id"] == s["id"] and filter_a(r)]
            rb = [r for r in results if r["snippet_id"] == s["id"] and filter_b(r)]
            for a in ra:
                for b in rb:
                    if a.get("parsed",{}).get("verdict") == b.get("parsed",{}).get("verdict"):
                        agreements += 1
                    total += 1
        return agreements / total if total > 0 else 0, total

    cross_model_persona = {}
    for pkey in PERSONAS:
        rate, n = verdict_agreement(
            lambda r, p=pkey: r["condition"]=="persona" and r["persona"]==p and "deepseek" in r["model"].lower(),
            lambda r, p=pkey: r["condition"]=="persona" and r["persona"]==p and "kimi" in r["model"].lower()
        )
        cross_model_persona[pkey] = {"agreement_rate": round(rate, 4), "n_pairs": n}

    within_model = {}
    for m in models:
        m_short = "DS" if "deepseek" in m.lower() else "Kimi"
        pairs_per_s = 3 * 2 // 2  # 3 personas -> 3 pairs per snippet
        pers_trials = [r for r in results if r["condition"]=="persona" and r["model"]==m]
        agreements, total = 0, 0
        for s in SNIPPETS:
            trials = [r for r in pers_trials if r["snippet_id"]==s["id"]]
            for i in range(len(trials)):
                for j in range(i+1, len(trials)):
                    if trials[i].get("parsed",{}).get("verdict") == trials[j].get("parsed",{}).get("verdict"):
                        agreements += 1
                    total += 1
        within_model[m_short] = {"agreement_rate": round(agreements/total, 4) if total > 0 else 0, "n_pairs": total}

    # Manipulation check: score deltas persona vs control
    def avg_score(trials, metric):
        vals = [r["parsed"][metric] for r in trials if "parsed" in r and metric in r["parsed"]]
        return sum(vals)/len(vals) if vals else 0

    manipulation = {}
    for m_label, m in [("DS", models[0]), ("Kimi", models[1])]:
        ctrl_set = [r for r in results if r["condition"]=="control" and r["model"]==m]
        pers_set = [r for r in results if r["condition"]=="persona" and r["model"]==m]
        manipulation[m_label] = {}
        for metric in ["score_clarity","score_correctness","score_efficiency","score_maintainability"]:
            c_avg = avg_score(ctrl_set, metric)
            p_avg = avg_score(pers_set, metric)
            manipulation[m_label][metric] = {"control_avg": round(c_avg, 2), "persona_avg": round(p_avg, 2), "delta": round(p_avg - c_avg, 2)}

    # ── Output ─────────────────────────────────────────────────────────────
    output = {
        "experiment": "e1b_cross_model",
        "pre_reg_hash": prereg_hash,
        "n_snippets": len(SNIPPETS),
        "n_trials": len(results),
        "n_errors": sum(1 for r in results if "error" in r),
        "total_tokens": total_tk,
        "models": models,
        "snippet_ids": [s["id"] for s in SNIPPETS],
        "categories": cat_list,
        "fleiss_kappa": stats,
        "cross_model_same_persona": cross_model_persona,
        "within_model_diff_persona": within_model,
        "manipulation_check": manipulation,
        "results": results
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    # ── Print Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("FLEISS KAPPA (chance-corrected agreement)")
    print(f"{'='*70}")
    for name, s in stats.items():
        if s["kappa"] is not None:
            print(f"  {name:<25}: k={s['kappa']:.4f}  95%CI=[{s['ci95_lower']:.4f}, {s['ci95_upper']:.4f}]  n={s['n_subjects']}")
        else:
            print(f"  {name:<25}: {s.get('error','?')}")

    print(f"\nCROSS-MODEL SAME-PERSONA (per persona):")
    for pkey, v in cross_model_persona.items():
        print(f"  {pkey:<10}: {v['agreement_rate']:.3f} (n={v['n_pairs']})")

    print(f"\nWITHIN-MODEL DIFF-PERSONA:")
    for m, v in within_model.items():
        print(f"  {m:<10}: {v['agreement_rate']:.3f} (n={v['n_pairs']})")

    print(f"\nMANIPULATION CHECK (persona vs control score delta):")
    for m_label, metrics in manipulation.items():
        deltas = [v["delta"] for v in metrics.values()]
        print(f"  {m_label}: " + " ".join(f"{k}={v['delta']:+.2f}" for k,v in metrics.items()) + f"  mean_delta={sum(deltas)/len(deltas):+.2f}")

    print(f"\nSaved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
