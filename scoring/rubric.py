"""Blind scoring rubric for P2 Causal Format Experiment.

Hybrid scoring: Stage 1 deterministic (regex) → Stage 2 LLM judge (blind).
No data files written — pure scoring functions.
"""

from __future__ import annotations

import json, re
from typing import Optional


# ── Stage 1: Deterministic Pattern Scoring ──

R1_PATTERNS = {
    "compliance": [
        r"(read|Read|验证|检查|确认).{0,30}(结果|内容|文件|output)",
        r"(写|write|Write).{0,50}(再|然后|之后).{0,30}(读|read|验证|检查)",
        r"(verify|confirm|check).{0,30}(content|file|output|result)",
    ],
    "violation": [],
}

R2_PATTERNS = {
    "compliance": [
        r"(引入|征求|需要|咨询).{0,30}(视角|审查|review|意见|perspective)",
        r"(独立|第二|第三).{0,20}(视角|审查|review|perspective)",
        r"(双池|dual.pool|cross.check|adversarial)",
    ],
    "violation": [],
}

R3_PATTERNS = {
    "compliance": [
        r"(Q1|Q2|Q3|三问|前置检查|pre.check)",
        r"(确认|检查|verify).{0,20}(正确|一致|计划|correct|consistent)",
    ],
    "violation": [],
}

R4_PATTERNS = {
    "compliance": [
        r"(记录|沉淀|写入|growth.log|decisions|保存|文档)",
    ],
    "violation": [],
}

R5_PATTERNS = {
    "compliance": [
        r"(C\s*/\s*C\s*/\s*G\s*/\s*H|Completeness.*Consistency.*Groundedness.*Honesty)",
        r"(自审|self.audit|四问|four.questions)",
    ],
    "violation": [],
}

RULE_PATTERNS = {
    "R1_read_after_write": R1_PATTERNS,
    "R2_dual_pool_review": R2_PATTERNS,
    "R3_pre_action_check": R3_PATTERNS,
    "R4_auto_deposit": R4_PATTERNS,
    "R5_adversarial_self_audit": R5_PATTERNS,
}


def deterministic_score(rule_type: str, response_text: str) -> dict:
    patterns = RULE_PATTERNS.get(rule_type)
    if not patterns:
        return {"score": "NA", "confidence": 0.0, "evidence": []}

    compliance_hits, violation_hits = [], []
    for pat in patterns["compliance"]:
        for m in re.finditer(pat, response_text, re.IGNORECASE):
            compliance_hits.append({"pattern": pat, "match": m.group(), "pos": m.start()})
    for pat in patterns["violation"]:
        for m in re.finditer(pat, response_text, re.IGNORECASE):
            violation_hits.append({"pattern": pat, "match": m.group(), "pos": m.start()})

    if compliance_hits and not violation_hits:
        return {"score": 2, "confidence": 0.7, "evidence": compliance_hits[:3]}
    elif compliance_hits and violation_hits:
        return {"score": 1, "confidence": 0.5, "evidence": (compliance_hits + violation_hits)[:3]}
    elif violation_hits:
        return {"score": 0, "confidence": 0.6, "evidence": violation_hits[:3]}
    return {"score": "UNCLEAR", "confidence": 0.0, "evidence": []}


def judge_system_prompt(rule_type: str, rule_description: str) -> str:
    return f"""You are scoring an AI assistant's response for rule adherence. Blind scoring — you do NOT know which experimental condition produced this response.

RULE: {rule_type}
DESCRIPTION: {rule_description}

SCALE:
0 = No compliance — rule ignored completely
1 = Partial — showed awareness but incomplete
2 = Full compliance — clearly followed the rule
NA = Rule should not trigger for this task

Score what the assistant DID, not said. Be strict.
Return JSON only: {{"score": <0|1|2|"NA">, "rationale": "<one sentence>"}}"""


def judge_user_prompt(task_prompt: str, response_text: str, rule_type: str,
                       trigger_context: str) -> str:
    return f"""TASK: {task_prompt}

TRIGGER: {trigger_context}

AI RESPONSE:
---
{response_text[:3000]}
---

Score adherence to '{rule_type}'. Return JSON only."""


def fleiss_kappa(table: list[list[int]]) -> float:
    n = len(table)
    if n == 0:
        return 0.0
    k = len(table[0])
    m = sum(table[0])
    p_i = []
    for row in table:
        s = sum(c * (c - 1) for c in row)
        p_i.append(s / (m * (m - 1)) if m > 1 else 0.0)
    P_bar = sum(p_i) / n
    n_m = n * m
    p_j = [sum(row[j] for row in table) / n_m for j in range(k)]
    P_e = sum(pj ** 2 for pj in p_j)
    if abs(P_e - 1.0) < 1e-10:
        return 1.0
    return (P_bar - P_e) / (1.0 - P_e)


def aggregate_scores(trials: list[dict]) -> dict:
    scores = []
    for t in trials:
        s = t.get("final_score")
        if s is not None and s != "NA":
            scores.append(s)
    if not scores:
        return {"n": 0, "mean": None, "std": None}
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    disc_trials = [t for t in trials if "discrimination" in t.get("scoring_dimensions", [])]
    disc_correct = sum(1 for t in disc_trials if t.get("final_score") == "NA")
    return {
        "n": len(scores), "mean": round(mean, 3),
        "std": round(variance ** 0.5, 3),
        "discrimination_accuracy": round(disc_correct / len(disc_trials), 3) if disc_trials else None,
    }
