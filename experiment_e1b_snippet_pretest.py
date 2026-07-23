#!/usr/bin/env python3
"""
E1b: Persona Decorrelation — Snippet Pre-test Phase
=====================================================
Pre-tests 30 snippets to find ~15 that produce high between-persona
disagreement within a single model. These become the instrument for
the full E1b cross-model replication.

Design:
  Phase 1 (this script): 30 snippets × 3 personas × 1 model = 90 calls
    → Rank snippets by within-model disagreement rate
    → Select top ~15 with highest disagreement (target: agreement ~40-60%)

  Phase 2 (e1b_cross_model.py, after pre-test):
    → 15 selected snippets × 3 personas × 2 models = 90 calls
    → No-persona control: 15 snippets × 2 models = 30 calls
    → Total: 120 calls
    → Primary: Fleiss' κ per condition
    → Manipulation check: no-persona vs persona agreement delta

Importers: standalone — run directly via python
API: DS V4 Pro direct API (cheapest, for pre-test)
Output: results/e1b_snippet_pretest.json

User verbatim: "接着E1b snippet 重设计（预筛选30+高分歧snippet）"
"""
import json, hashlib, os, subprocess, sys, time, re
from datetime import datetime, timezone

DS_API_KEY = os.environ["DS_TOKEN"]
DS_API_URL = "https://api.deepseek.com/v1/chat/completions"
OUTPUT_FILE = "results/e1b_snippet_pretest.json"

# ── Personas (same as E1) ──────────────────────────────────────────────────
PERSONAS = {
    "carmack": {
        "name": "John Carmack",
        "principle": "The best code is the code that runs fastest with fewest surprises.",
        "focus": "Performance, algorithmic efficiency, simplicity. Reject over-engineering. Prefer direct solutions that are fast and debuggable.",
        "red_flags": "Unnecessary abstraction layers, O(n²) where O(n) is possible, allocations in hot paths, clever code that's not measurably faster."
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

# ── 30 Ambiguous Snippets ──────────────────────────────────────────────────
# Designed to produce genuine disagreement. Each has a plausible defense
# AND a plausible criticism. No obvious right answer.

SNIPPETS = [
    # ── Error Handling Philosophy (6) ──
    {
        "id": "err_failfast_vs_graceful",
        "domain": "error_handling",
        "title": "Fail-fast vs graceful degradation in data pipeline",
        "code": "def process_batch(records):\n    results = []\n    for rec in records:\n        try:\n            validated = validate_schema(rec)\n            enriched = enrich_from_cache(validated)\n            results.append(enriched)\n        except SchemaError:\n            raise  # fail the entire batch\n        except CacheMissError:\n            results.append(validated)  # proceed without enrichment\n    return results"
    },
    {
        "id": "err_return_code_vs_exception",
        "domain": "error_handling",
        "title": "Return codes vs exceptions for business logic",
        "code": "def transfer_funds(src, dst, amount):\n    if amount <= 0:\n        return {'ok': False, 'code': 'INVALID_AMOUNT'}\n    src_bal = db.get_balance(src)\n    if src_bal < amount:\n        return {'ok': False, 'code': 'INSUFFICIENT_FUNDS'}\n    db.debit(src, amount)\n    db.credit(dst, amount)\n    return {'ok': True, 'txn_id': db.last_id()}"
    },
    {
        "id": "err_null_vs_optional",
        "domain": "error_handling",
        "title": "None return vs explicit Optional type",
        "code": "def find_user(email):\n    cursor = db.execute('SELECT * FROM users WHERE email=?', email)\n    row = cursor.fetchone()\n    if row is None:\n        return None\n    return User.from_row(row)\n\n# Usage:\nuser = find_user('test@example.com')\nprint(user.name)  # AttributeError if not found"
    },
    {
        "id": "err_log_and_swallow",
        "domain": "error_handling",
        "title": "Log-and-swallow vs propagate in background job",
        "code": "def send_notifications(users, message):\n    for user in users:\n        try:\n            email_service.send(user.email, message)\n        except EmailTimeoutError:\n            logger.warning(f'Failed to send to {user.id}, will retry next batch')\n            continue\n        except EmailBounceError:\n            logger.error(f'Bounced: {user.id}')\n            db.mark_bounced(user.id)\n            continue"
    },
    {
        "id": "err_assert_vs_validate",
        "domain": "error_handling",
        "title": "assert vs explicit validation in public API",
        "code": "def calculate_discount(cart_items, promo_code=None):\n    assert len(cart_items) > 0, 'cart must not be empty'\n    assert all(i.price > 0 for i in cart_items), 'all items must have price'\n    subtotal = sum(i.price * i.qty for i in cart_items)\n    if promo_code:\n        assert promo_code in ACTIVE_PROMOS, f'invalid promo: {promo_code}'\n        discount = ACTIVE_PROMOS[promo_code].apply(subtotal)\n        return subtotal - discount\n    return subtotal"
    },
    {
        "id": "err_partial_success",
        "domain": "error_handling",
        "title": "Partial success vs all-or-nothing for multi-file upload",
        "code": "def upload_files(file_paths):\n    uploaded = []\n    errors = []\n    for path in file_paths:\n        try:\n            url = storage.upload(path)\n            uploaded.append({'path': path, 'url': url})\n        except StorageError as e:\n            errors.append({'path': path, 'error': str(e)})\n    return {'uploaded': uploaded, 'errors': errors, 'ok': len(errors) == 0}"
    },

    # ── API Design Tradeoffs (5) ──
    {
        "id": "api_dict_vs_dataclass",
        "domain": "api_design",
        "title": "Dict vs dataclass as function return type",
        "code": "def get_metrics(time_range):\n    raw = prometheus.query_range('cpu_usage', time_range)\n    return {\n        'avg': sum(raw) / len(raw),\n        'p95': sorted(raw)[int(len(raw)*0.95)],\n        'p99': sorted(raw)[int(len(raw)*0.99)],\n        'max': max(raw),\n        'count': len(raw)\n    }"
    },
    {
        "id": "api_config_in_code",
        "domain": "api_design",
        "title": "Hardcoded defaults vs configuration injection",
        "code": "class RetryPolicy:\n    MAX_RETRIES = 3\n    BASE_DELAY_MS = 1000\n    MAX_DELAY_MS = 30000\n    BACKOFF_MULTIPLIER = 2.0\n    RETRYABLE_STATUSES = {429, 502, 503, 504}\n\n    @classmethod\n    def should_retry(cls, status_code, attempt):\n        if attempt >= cls.MAX_RETRIES:\n            return False\n        if status_code not in cls.RETRYABLE_STATUSES:\n            return False\n        return True"
    },
    {
        "id": "api_positional_vs_kwargs",
        "domain": "api_design",
        "title": "Positional vs keyword-only for multi-param function",
        "code": "def create_alert(name, severity, target, threshold,\n                  window_minutes=5, cooldown_minutes=15,\n                  notification_channel='email', auto_resolve=True,\n                  aggregation='avg', labels=None):\n    return db.insert('alerts', locals())"
    },
    {
        "id": "api_method_chaining",
        "domain": "api_design",
        "title": "Method chaining vs explicit steps for query builder",
        "code": "query = (Query()\n    .select('name', 'email', 'last_login')\n    .from_table('users')\n    .where('status', '=', 'active')\n    .where('last_login', '>', thirty_days_ago)\n    .order_by('last_login', desc=True)\n    .limit(50)\n    .execute())"
    },
    {
        "id": "api_boolean_params",
        "domain": "api_design",
        "title": "Boolean flag vs separate methods",
        "code": "def fetch_orders(user_id, include_cancelled=True,\n                  include_refunded=False, include_pending=True):\n    statuses = ['completed', 'shipped']\n    if include_cancelled:\n        statuses.append('cancelled')\n    if include_refunded:\n        statuses.append('refunded')\n    if include_pending:\n        statuses.append('pending')\n    return db.query(\n        'SELECT * FROM orders WHERE user_id=? AND status IN ?',\n        user_id, statuses)"
    },

    # ── Performance vs Readability (5) ──
    {
        "id": "perf_listcomp_vs_loop",
        "domain": "perf_vs_readability",
        "title": "Nested list comprehension vs explicit loop",
        "code": "def extract_active_emails(users_by_dept):\n    return [\n        u.email\n        for dept in users_by_dept.values()\n        for u in dept\n        if u.is_active and u.email_verified and u.role != 'bot'\n    ]"
    },
    {
        "id": "perf_cache_everything",
        "domain": "perf_vs_readability",
        "title": "Aggressive caching vs compute-on-demand",
        "code": "_user_cache = {}\n\ndef get_user(user_id):\n    if user_id in _user_cache:\n        return _user_cache[user_id]\n    user = db.fetch_user(user_id)\n    _user_cache[user_id] = user\n    return user\n\ndef update_user(user_id, data):\n    db.update_user(user_id, data)\n    _user_cache.pop(user_id, None)"
    },
    {
        "id": "perf_preallocate_vs_append",
        "domain": "perf_vs_readability",
        "title": "Pre-allocated array vs dynamic append",
        "code": "def transform_values(items):\n    n = len(items)\n    result = [None] * n\n    for i in range(n):\n        result[i] = complex_transform(items[i])\n    return result"
    },
    {
        "id": "perf_lazy_vs_eager",
        "domain": "perf_vs_readability",
        "title": "Generator pipeline vs immediate list",
        "code": "def find_matches(documents, query):\n    tokenized = (tokenize(d) for d in documents)\n    filtered = (t for t in tokenized if len(t) > 2)\n    scored = ((t, cosine_sim(t, query)) for t in filtered)\n    ranked = sorted(scored, key=lambda x: x[1], reverse=True)\n    return ranked[:10]"
    },
    {
        "id": "perf_regex_compiled",
        "domain": "perf_vs_readability",
        "title": "Pre-compiled regex module-level vs inline",
        "code": "import re\n\nEMAIL_RE = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$')\nPHONE_RE = re.compile(r'^\\+?[\\d\\s\\-()]{7,15}$')\n\ndef validate_contact(email, phone):\n    return EMAIL_RE.match(email) and PHONE_RE.match(phone)"
    },

    # ── Abstraction & Architecture (5) ──
    {
        "id": "arch_strategy_pattern",
        "domain": "architecture",
        "title": "Strategy pattern vs if-else for 3 payment methods",
        "code": "class PaymentProcessor:\n    def __init__(self, method, config):\n        self.method = method\n        self.config = config\n\n    def pay(self, amount):\n        if self.method == 'stripe':\n            return StripeAPI.charge(self.config['key'], amount)\n        elif self.method == 'paypal':\n            return PayPalAPI.capture(self.config['client_id'], amount)\n        elif self.method == 'bank_transfer':\n            return BankAPI.transfer(self.config['account'], amount)\n        raise ValueError(f'Unknown method: {self.method}')"
    },
    {
        "id": "arch_inheritance_depth",
        "domain": "architecture",
        "title": "Deep inheritance vs composition for model layer",
        "code": "class BaseModel:\n    def save(self): db.insert(self.table, self.to_dict())\n    def delete(self): db.delete(self.table, self.id)\n\nclass TimestampedModel(BaseModel):\n    def save(self):\n        self.updated_at = now()\n        super().save()\n\nclass SoftDeleteModel(TimestampedModel):\n    def delete(self):\n        self.deleted_at = now()\n        self.save()\n\nclass User(SoftDeleteModel):\n    table = 'users'"
    },
    {
        "id": "arch_service_layer",
        "domain": "architecture",
        "title": "Service layer for simple CRUD vs direct ORM",
        "code": "class UserService:\n    def __init__(self, db, email_service):\n        self.db = db\n        self.email_service = email_service\n\n    def register(self, email, password):\n        existing = self.db.query(User).filter_by(email=email).first()\n        if existing:\n            raise ValueError('Email already registered')\n        user = User(email=email, password_hash=hash_pw(password))\n        self.db.add(user)\n        self.db.commit()\n        self.email_service.send_welcome(email)\n        return user\n\n    def get_by_id(self, user_id):\n        return self.db.query(User).get(user_id)\n\n    def deactivate(self, user_id):\n        user = self.get_by_id(user_id)\n        user.active = False\n        self.db.commit()"
    },
    {
        "id": "arch_singleton",
        "domain": "architecture",
        "title": "Singleton vs dependency injection for DB connection",
        "code": "class Database:\n    _instance = None\n\n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n            cls._instance.pool = create_connection_pool(\n                host=os.environ['DB_HOST'],\n                port=int(os.environ['DB_PORT']),\n                min_conn=5, max_conn=20\n            )\n        return cls._instance\n\n    def query(self, sql, *params):\n        conn = self.pool.get()\n        try:\n            return conn.execute(sql, params)\n        finally:\n            self.pool.put(conn)"
    },
    {
        "id": "arch_microservice_boundary",
        "domain": "architecture",
        "title": "In-process vs HTTP call for order validation",
        "code": "def validate_order(order):\n    # Inline: fast, no network, but duplicates inventory logic\n    for item in order.items:\n        stock = inventory_db.get_stock(item.sku)\n        if stock < item.qty:\n            return {'ok': False, 'item': item.sku, 'available': stock}\n\n    # HTTP call for authoritative pricing\n    for item in order.items:\n        price = requests.get(\n            f'http://pricing-svc/price/{item.sku}'\n        ).json()\n        if price['current'] != item.unit_price:\n            item.unit_price = price['current']\n\n    return {'ok': True, 'order': order}"
    },

    # ── Testing Philosophy (4) ──
    {
        "id": "test_private_methods",
        "domain": "testing",
        "title": "Testing private methods vs testing only public API",
        "code": "class OrderCalculator:\n    def calculate_total(self, items, tax_region):\n        subtotal = self._sum_items(items)\n        discount = self._apply_promotions(items, subtotal)\n        tax = self._calculate_tax(subtotal - discount, tax_region)\n        return subtotal - discount + tax\n\n    def _sum_items(self, items):\n        return sum(i.price * i.qty for i in items)\n\n    def _apply_promotions(self, items, subtotal):\n        active = [p for p in self.promos if p.is_active(items)]\n        total_discount = 0\n        for promo in sorted(active, key=lambda p: p.priority):\n            total_discount += promo.apply(items, subtotal - total_discount)\n        return min(total_discount, subtotal)"
    },
    {
        "id": "test_mock_vs_real",
        "domain": "testing",
        "title": "Mock everything vs integration test for email sending",
        "code": "def test_welcome_email():\n    mock_sender = Mock()\n    mock_template = Mock(return_value='<html>Welcome, Alice!</html>')\n\n    service = EmailService(\n        sender=mock_sender,\n        templates={'welcome': mock_template}\n    )\n    service.send_welcome('alice@example.com', {'name': 'Alice'})\n\n    mock_sender.send.assert_called_once_with(\n        to='alice@example.com',\n        subject='Welcome to Our Platform!',\n        body='<html>Welcome, Alice!</html>'\n    )"
    },
    {
        "id": "test_coverage_target",
        "domain": "testing",
        "title": "100% coverage target vs risk-based testing",
        "code": "def format_currency(amount, locale='en_US'):\n    if amount is None:\n        return chr(0x2014)  # em dash\n    if not isinstance(amount, (int, float)):\n        raise TypeError(f'Expected number, got {type(amount)}')\n    if amount < 0:\n        sign = '-'\n        amount = abs(amount)\n    else:\n        sign = ''\n    formatted = locale.format_string('%.2f', amount, grouping=True)\n    return f'{sign}${formatted}'"
    },
    {
        "id": "test_snapshot_vs_assert",
        "domain": "testing",
        "title": "Snapshot testing vs explicit assertions for API response",
        "code": "def test_user_profile_api():\n    response = client.get('/api/users/42/profile')\n    assert response.status_code == 200\n    data = response.json()\n\n    # Snapshot: capture everything, detect any change\n    assert data == {\n        'id': 42,\n        'name': 'Alice Johnson',\n        'email': 'alice@example.com',\n        'joined': '2024-03-15',\n        'plan': 'premium',\n        'features': ['analytics', 'api', 'export'],\n        'stats': {'projects': 12, 'storage_mb': 450}\n    }"
    },

    # ── Type System & Safety (5) ──
    {
        "id": "type_any_vs_generic",
        "domain": "type_system",
        "title": "Any vs Generic in collection utility",
        "code": "def pluck(records, *keys):\n    \"\"\"Extract multiple fields from list of dicts.\"\"\"\n    result = []\n    for r in records:\n        if len(keys) == 1:\n            result.append(r.get(keys[0]))\n        else:\n            result.append(tuple(r.get(k) for k in keys))\n    return result"
    },
    {
        "id": "type_cast_vs_validate",
        "domain": "type_system",
        "title": "Type casting vs schema validation at API boundary",
        "code": "def parse_webhook(payload):\n    return {\n        'event': str(payload['event']),\n        'user_id': int(payload['user_id']),\n        'amount': float(payload['amount']),\n        'timestamp': datetime.fromisoformat(payload['timestamp']),\n        'metadata': payload.get('metadata', {})\n    }"
    },
    {
        "id": "type_enums_vs_strings",
        "domain": "type_system",
        "title": "String constants vs Enum for status values",
        "code": "ORDER_STATUS_PENDING = 'pending'\nORDER_STATUS_CONFIRMED = 'confirmed'\nORDER_STATUS_SHIPPED = 'shipped'\nORDER_STATUS_DELIVERED = 'delivered'\nORDER_STATUS_CANCELLED = 'cancelled'\n\nVALID_TRANSITIONS = {\n    ORDER_STATUS_PENDING: [ORDER_STATUS_CONFIRMED, ORDER_STATUS_CANCELLED],\n    ORDER_STATUS_CONFIRMED: [ORDER_STATUS_SHIPPED, ORDER_STATUS_CANCELLED],\n    ORDER_STATUS_SHIPPED: [ORDER_STATUS_DELIVERED],\n}\n\ndef can_transition(current, target):\n    return target in VALID_TRANSITIONS.get(current, [])"
    },
    {
        "id": "type_default_args",
        "domain": "type_system",
        "title": "Mutable default arguments vs None sentinel",
        "code": "def register_hooks(event_types=['push', 'pull_request'],\n                      callbacks=[], timeout=30, retry=True):\n    for event in event_types:\n        for cb in callbacks:\n            dispatcher.subscribe(event, cb, timeout=timeout)\n    if retry:\n        for event in event_types:\n            dispatcher.enable_retry(event, max_attempts=3)"
    },
    {
        "id": "type_duck_vs_isinstance",
        "domain": "type_system",
        "title": "Duck typing vs isinstance check for file-like objects",
        "code": "def export_data(data, destination):\n    \"\"\"Export to file path or file-like object.\"\"\"\n    if isinstance(destination, str):\n        with open(destination, 'w') as f:\n            json.dump(data, f)\n    elif hasattr(destination, 'write'):\n        json.dump(data, destination)\n    else:\n        raise TypeError(f'Cannot write to {type(destination)}')"
    },
]

# ── JSON and API utilities (reused from E1) ───────────────────────────────

SCORING_PROMPT = """
Review the code above according to your principles.

You MUST output exactly a JSON object with these fields, nothing else:
{"verdict": "APPROVE"|"REJECT"|"APPROVE_WITH_NOTES", "critical_issues": [], "score_clarity": 1-5, "score_correctness": 1-5, "score_efficiency": 1-5, "score_maintainability": 1-5, "key_observation": "one sentence"}

Do NOT wrap in markdown code blocks. Do NOT add explanation. Output pure JSON only.
"""

def extract_json(text):
    """Try multiple strategies to extract JSON from model output."""
    try:
        return json.loads(text)
    except:
        pass
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except:
            pass
    m = re.search(r'\{.*\}', text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except:
            pass
    return None

def call_model(model, system_prompt, user_prompt, max_tokens=600):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0
    }
    try:
        result = subprocess.run(
            ["curl", "-s", DS_API_URL,
             "-H", f"Authorization: Bearer {DS_API_KEY}",
             "-H", "Content-Type: application/json",
             "-d", json.dumps(payload)],
            capture_output=True, text=True, timeout=60,
            encoding="utf-8", errors="replace"
        )
        resp = json.loads(result.stdout)
        if "choices" in resp and len(resp["choices"]) > 0:
            content = resp["choices"][0]["message"]["content"]
            parsed = extract_json(content)
            if parsed:
                return {"ok": True, "parsed": parsed, "usage": resp.get("usage", {})}
            else:
                return {"ok": True, "parsed": {"verdict": "PARSE_ERROR"},
                        "usage": resp.get("usage", {}), "raw": content[:300]}
        return {"ok": False, "error": str(resp)[:200]}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}

# ── Disagreement scoring ──────────────────────────────────────────────────

def persona_disagreement_score(trials_for_snippet):
    """
    For a given snippet, calculate how much the 3 personas disagree.
    Returns (agreement_rate, disagree_count, verdicts_dict).
    Perfect agreement = 3/3 same → score 1.0
    All 3 different = 0/3 same → score 0.0
    2 agree, 1 differs = 1/3 same → score 0.33
    """
    verdicts = {}
    for t in trials_for_snippet:
        p = t["persona"]
        v = t.get("parsed", {}).get("verdict", "PARSE_ERROR")
        verdicts[p] = v

    personas = list(verdicts.keys())
    if len(personas) < 3:
        return 0.0, 3, verdicts

    pairs = [(personas[0], personas[1]), (personas[0], personas[2]), (personas[1], personas[2])]
    agreements = sum(1 for a, b in pairs if verdicts[a] == verdicts[b])
    return agreements / 3, 3 - agreements, verdicts

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs("results", exist_ok=True)

    # Pre-registration
    prereg = json.dumps({
        "experiment": "e1b_snippet_pretest",
        "phase": 1,
        "design": "30 snippets × 3 personas × 1 model (DS V4 Pro) = 90 calls",
        "goal": "Select ~15 snippets with high between-persona disagreement",
        "selection_criterion": "within-model agreement rate < 1.0 AND 0 errors",
        "personas": list(PERSONAS.keys()),
        "model": "deepseek-ai/DeepSeek-V4-Pro",
        "snippet_count": len(SNIPPETS),
        "exclusion_rule": "PARSE_ERROR on any persona → snippet excluded"
    }, sort_keys=True)
    prereg_hash = hashlib.sha256(prereg.encode()).hexdigest()[:16]
    print(f"PRE-REG: {prereg_hash}")
    print(f"Snippets: {len(SNIPPETS)}, Trials: {len(SNIPPETS) * 3}\n")

    results = []
    total_tk = 0

    for i, s in enumerate(SNIPPETS):
        snippet_trials = []
        for pkey, p in PERSONAS.items():
            label = f"[{i+1:02d}/30] {s['id'][:30]:30s} {pkey[:8]}"
            print(f"  {label}", end=" ", flush=True)

            sp = f"You are {p['name']} conducting a code review. Your principle: {p['principle']} Your focus: {p['focus']} Red flags: {p['red_flags']}"
            up = f"Title: {s['title']}\n\nCode:\n{s['code']}\n\n{SCORING_PROMPT}"

            r = call_model("deepseek-chat", sp, up)
            entry = {
                "snippet_id": s["id"], "domain": s["domain"],
                "persona": pkey, "pre_reg_hash": prereg_hash
            }
            if r["ok"]:
                entry["parsed"] = r["parsed"]
                entry["usage"] = r["usage"]
                tk = r["usage"].get("total_tokens", 0)
                total_tk += tk
                v = r["parsed"].get("verdict", "?")
                print(f"→ {v[:12]:12s} ({tk}tk)")
            else:
                entry["error"] = r.get("error", "?")
                print(f"→ ERR: {entry['error'][:40]}")
            snippet_trials.append(entry)
            results.append(entry)
            time.sleep(0.2)

        # Calculate disagreement for this snippet
        ag_rate, dis_count, verdicts = persona_disagreement_score(snippet_trials)
        marker = "★★★ HIGH DISAGREEMENT" if ag_rate <= 0.67 else "  unanimous" if ag_rate == 1.0 else "  moderate"
        print(f"       Agreement: {ag_rate:.2f} (disagree={dis_count}) {marker}")
        if (i + 1) % 10 == 0:
            print(f"  --- {i+1}/{len(SNIPPETS)} done, {total_tk} tokens ---\n")

    # ── Ranking & Selection ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"RANKING: Snippets by disagreement (low agreement = high disagreement)")
    print(f"{'='*70}")

    snippet_scores = []
    for s in SNIPPETS:
        trials = [r for r in results if r["snippet_id"] == s["id"]]
        ag_rate, dis_count, verdicts = persona_disagreement_score(trials)
        errors = sum(1 for t in trials if "error" in t)
        snippet_scores.append({
            "id": s["id"], "domain": s["domain"], "title": s["title"],
            "agreement_rate": ag_rate, "disagree_count": dis_count,
            "verdicts": verdicts, "errors": errors,
            "selectable": errors == 0 and ag_rate < 1.0
        })

    snippet_scores.sort(key=lambda x: x["agreement_rate"])

    selected = []
    print(f"{'#':<3} {'Agr':<6} {'Dis':<5} {'Domain':<22} {'Snippet':<40} {'Verdicts'}")
    print("-" * 110)
    for i, ss in enumerate(snippet_scores):
        status = ""
        if ss["errors"] > 0:
            status = " [EXCLUDED: errors]"
        elif ss["agreement_rate"] == 1.0:
            status = " [EXCLUDED: ceiling]"
        elif len(selected) < 15:
            selected.append(ss)
            status = " ← SELECTED"

        vstr = " | ".join(f"{p[:4]}={v[:6]}" for p, v in ss["verdicts"].items())
        print(f"{i+1:<3} {ss['agreement_rate']:.2f}   {ss['disagree_count']:<5} {ss['domain']:<22} {ss['id']:<40} {vstr}{status}")

    print(f"\nSelected: {len(selected)}/{len(SNIPPETS)} snippets")
    print(f"Total tokens: {total_tk}")
    print(f"Errors: {sum(1 for r in results if 'error' in r)}")

    # Save output
    output = {
        "experiment": "e1b_snippet_pretest",
        "phase": 1,
        "pre_reg_hash": prereg_hash,
        "n_snippets_tested": len(SNIPPETS),
        "n_trials_completed": len(results),
        "n_errors": sum(1 for r in results if "error" in r),
        "total_tokens": total_tk,
        "model": "deepseek-ai/DeepSeek-V4-Pro (via direct API as deepseek-chat)",
        "selection_criterion": "agreement_rate < 1.0 AND 0 errors",
        "ranking": snippet_scores,
        "selected": [s["id"] for s in selected],
        "selected_details": selected,
        "all_results": results
    }
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\nSaved to {OUTPUT_FILE}")
    print(f"Selected: {[s['id'] for s in selected]}")

    # Print for copy-paste
    print(f"\n{'='*70}")
    print("SELECTED SNIPPETS FOR E1B CROSS-MODEL:")
    print(f"{'='*70}")
    for ss in selected:
        print(f"  {ss['id']}  # agr={ss['agreement_rate']:.2f}, {ss['domain']}")

if __name__ == "__main__":
    main()
