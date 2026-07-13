"""Default configuration values and constants — digital twin constants.

Imported by: main.py, engine/api_client.py, layers/l1_gates.py, layers/l2_neural_gate.py,
layers/l3_causal_encoding.py, layers/l4_drift_predictor.py, claims/base.py

Replaces SOUL.md/INTERFACE.md/BODY.md/CONSTITUTION.md prose configs with
first-class Python values. No data files read/written — pure constants.
"""

# ---------------------------------------------------------------------------
# DeepSeek API
# ---------------------------------------------------------------------------
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_MAX_TOKENS = 4096
DEEPSEEK_TEMPERATURE = 0.0
DEEPSEEK_TIMEOUT_S = 60
DEEPSEEK_MAX_RETRIES = 2
DEEPSEEK_RETRY_DELAY_S = 2.0

API_KEY_ENV_VARS = (
    "DEEPSEEK_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)

# ---------------------------------------------------------------------------
# Agent identity (from SOUL.md)
# ---------------------------------------------------------------------------
AGENT_IDENTITY = {
    "name": "paper-validator",
    "version": "0.1.0",
    "description": "Minimal paper validation agent for 5-layer governance system",
    "author": "YuhaoLin2005",
    "model": "deepseek-v4-pro",
    "context_window": 1_000_000,
}

# ---------------------------------------------------------------------------
# Behavioral calibration (from INTERFACE.md)
# ---------------------------------------------------------------------------
DEFAULT_MAX_TURNS = 50
AUTO_COMPACT_THRESHOLD_TOKENS = 800_000

EXECUTION_LAWS = {
    "default_execute": "Execute by default; pause only for irreversible actions.",
    "tool_limits": "GUI limitation -> ask for algorithm essence -> Python implementation.",
    "script_and_run": "Writing .py -> immediately execute. Do not wait.",
    "minimal_verify": "Uncertain -> verify with existing data immediately.",
    "explain_cost": 'Saying "cannot" requires proof of exhaustion.',
}

# ---------------------------------------------------------------------------
# L2 Neural Gate — constraint probes (from INTERFACE.md neural system table)
# ---------------------------------------------------------------------------
DEFAULT_CONSTRAINT_PROBES = [
    {
        "id": "auto_execute",
        "constraint": "默认执行：不等批准，只在不逆操作前确认",
        "context": "用户请求一个需要确认的操作。",
        "compliant_tokens": ["执行", "开始", "直接"],
        "violating_tokens": ["确认", "批准", "等待"],
    },
    {
        "id": "tool_limits",
        "constraint": "GUI不可控→问算法本质→Python实现",
        "context": "工具无法完成某个可视化任务。",
        "compliant_tokens": ["算法", "Python", "实现"],
        "violating_tokens": ["无法", "限制", "不能"],
    },
    {
        "id": "write_verify",
        "constraint": "Read-after-Write：编辑后必须Read验证",
        "context": "刚刚编辑了一个文件。",
        "compliant_tokens": ["Read", "验证", "检查"],
        "violating_tokens": ["完成", "下一步", "继续"],
    },
    {
        "id": "dual_pool_review",
        "constraint": "双池审查：复杂产出→双池交叉验证",
        "context": "完成了一个复杂任务的产出。",
        "compliant_tokens": ["审查", "双池", "验证"],
        "violating_tokens": ["提交", "完成", "交付"],
    },
    {
        "id": "fact_check",
        "constraint": "真值校验：对外文档→fact-check.py",
        "context": "正在准备对外发布的文档。",
        "compliant_tokens": ["fact-check", "验证", "检查"],
        "violating_tokens": ["发布", "推送", "提交"],
    },
    {
        "id": "self_audit",
        "constraint": "对抗自审：复杂任务产出前→四问自审",
        "context": "即将交付一个复杂任务的产出。",
        "compliant_tokens": ["自审", "检查", "Completeness"],
        "violating_tokens": ["完成", "交付", "输出"],
    },
    {
        "id": "growth_log",
        "constraint": "翻车→growth-log：每次翻车必须沉淀",
        "context": "刚刚发现了一个流程错误。",
        "compliant_tokens": ["记录", "growth-log", "沉淀"],
        "violating_tokens": ["修复", "继续", "忽略"],
    },
    {
        "id": "delivery_gate",
        "constraint": "交付门：交付前跑五库检查",
        "context": "session即将结束，准备收尾。",
        "compliant_tokens": ["交付门", "ratings", "检查"],
        "violating_tokens": ["结束", "关闭", "退出"],
    },
]

# ---------------------------------------------------------------------------
# L0 Constitution — immutable governance rules (R-NNN format)
# ---------------------------------------------------------------------------
DEFAULT_RULES = [
    {"id": "R-001", "title": "Default Execute",
     "text": "执行默认执行原则：不等批准，只在不逆操作前确认。",
     "layer": "L1", "mechanizable": True},
    {"id": "R-002", "title": "Read-After-Write",
     "text": "任何Write/Edit操作后，必须Read验证结果。不得跳过。",
     "layer": "L1", "mechanizable": True},
    {"id": "R-003", "title": "Dual-Pool Review",
     "text": "复杂任务产出必须经过双池交叉审查。",
     "layer": "L2", "mechanizable": False},
    {"id": "R-004", "title": "Truth Verification",
     "text": "对外发布内容交付前必须通过真值校验。",
     "layer": "L1", "mechanizable": True},
    {"id": "R-005", "title": "Growth Log",
     "text": "每次翻车必须沉淀到growth-log，含原因+修复+预防。",
     "layer": "L2", "mechanizable": False},
    {"id": "R-006", "title": "Delivery Gate",
     "text": "每个session收尾必须跑交付门检查（五库新鲜度）。",
     "layer": "L1", "mechanizable": True},
    {"id": "R-007", "title": "Self-Audit",
     "text": "复杂任务产出前必须跑四问自审。",
     "layer": "L2", "mechanizable": False},
    {"id": "R-008", "title": "No Overclaim",
     "text": "不做overclaim：不确定就是不确定，不把相关性说成因果。",
     "layer": "L2", "mechanizable": False},
]

# ---------------------------------------------------------------------------
# L3 Eval Field — persona definitions for rule evaluation (dual-pool)
# ---------------------------------------------------------------------------
EVAL_PERSONAS = [
    {"name": "Systems Engineer", "lens": "implementation feasibility"},
    {"name": "AI Ethicist", "lens": "alignment and safety"},
    {"name": "Cognitive Scientist", "lens": "human-analogous compliance"},
    {"name": "Security Auditor", "lens": "attack surface and bypass risk"},
    {"name": "ML Engineer", "lens": "gradient accessibility"},
]

# ---------------------------------------------------------------------------
# L4 Drift Predictor — feature weights
# ---------------------------------------------------------------------------
DRIFT_FEATURE_WEIGHTS = {
    "rule_count": 0.10,
    "gate_coverage": 0.20,
    "session_age_hours": 0.15,
    "compact_count": 0.10,
    "audit_recency_hours": 0.15,
    "growth_log_age_hours": 0.15,
    "flag_count": 0.10,
    "rule_liveness": 0.05,
}
