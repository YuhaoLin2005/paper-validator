#!/usr/bin/env python3
# paper-validator MCP server. Importers: MCP clients (Claude Code, Desktop, Codex).
# Callers: claims/runner.py, layers/l1_gates.py, layers/l4_drift_predictor.py.
# Schema: ClaimReport {claim_id, verdict, effect_size}. User: wrap as MCP tool.
"""MCP server wrapping paper-validator — any MCP agent can invoke governance audit."""
from __future__ import annotations

import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from claims.runner import run_claim, run_all, list_claims


def handle_request(request: dict) -> dict:
    method = request.get("method", "")
    req_id = request.get("id", 0)

    if method == "tools/list":
        return {
            "jsonrpc": "2.0", "id": req_id,
            "result": {"tools": [
                {"name": "validate_rule",
                 "description": "Measure governance rule causal effect on LLM output. Returns compliance rate diff, effect size, verdict.",
                 "inputSchema": {"type": "object", "properties": {
                     "claim": {"type": "string", "enum": list(list_claims()) + ["all"]},
                     "trials": {"type": "integer", "default": 30}},
                     "required": ["claim"]}},
                {"name": "list_governance_claims",
                 "description": "List all available governance rule validation claims",
                 "inputSchema": {"type": "object", "properties": {}}},
                {"name": "health_check",
                 "description": "Run governance health check — layer status, rule liveness, drift risk",
                 "inputSchema": {"type": "object", "properties": {}}},
            ]}
        }

    elif method == "tools/call":
        name = request["params"]["name"]
        args = request["params"].get("arguments", {})

        if name == "list_governance_claims":
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": json.dumps(list(list_claims()))}]}}

        if name == "validate_rule":
            claim = args["claim"]
            trials = args.get("trials", 30)
            if claim == "all":
                results = run_all(n_trials=trials)
                text = "\n".join(f"{r.claim_id}: {r.verdict} d={r.effect_size:.3f}" for r in results)
            else:
                report = run_claim(claim, n_trials=trials)
                if not report:
                    return {"jsonrpc": "2.0", "id": req_id,
                            "error": {"code": -1, "message": f"Claim '{claim}' not found"}}
                text = json.dumps({"claim": report.claim_id, "verdict": report.verdict,
                                   "effect_size": report.effect_size, "trials": report.total_trials,
                                   "errors": len(report.errors), "metrics": report.metrics})
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": text}]}}

        if name == "health_check":
            from layers.l1_gates import L1Gates
            from layers.l4_drift_predictor import DriftPredictor
            h = L1Gates().health.run()
            risk = DriftPredictor().assess()
            text = json.dumps({"health": "ok" if h.ok else "degraded",
                               "checks_passed": len(h.passed), "drift_risk": risk.risk_level,
                               "drift_score": risk.risk_score})
            return {"jsonrpc": "2.0", "id": req_id,
                    "result": {"content": [{"type": "text", "text": text}]}}

    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -2, "message": f"Unknown: {method}"}}


def main():
    for line in sys.stdin:
        if not (line := line.strip()):
            continue
        try:
            resp = handle_request(json.loads(line))
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()
        except json.JSONDecodeError:
            continue


if __name__ == "__main__":
    main()
