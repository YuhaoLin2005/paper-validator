#!/usr/bin/env python3
"""CLI entry for paper-validator (python -m paper_validator).

Imports: engine.query, config.defaults, state.store, state.flags
Data: uses StateStore for session tracking (JSON, schema in state/__init__.py)
"""

from __future__ import annotations

import argparse, sys, os

_pkg_root = os.path.dirname(os.path.abspath(__file__))
if _pkg_root not in sys.path:
    sys.path.insert(0, _pkg_root)


def cmd_interactive(args):
    from engine.query import run_query
    from config.defaults import EXECUTION_LAWS

    system_prompt = (
        "You are a paper validation agent for AI agent governance systems. "
        "Tools: read_file, write_file, bash, grep.\n"
        "Core principles:\n" +
        "\n".join(f"- {v}" for v in EXECUTION_LAWS.values()) +
        "\n\nBe direct. Default to executing."
    )

    print("paper-validator v0.1.0 — interactive mode")
    print("Type /exit to quit\n")

    while True:
        try:
            ui = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye."); break
        if not ui:
            continue
        if ui == "/exit":
            print("Goodbye."); break
        print("Thinking...")
        result = run_query(system_prompt=system_prompt, user_prompt=ui)
        print(f"\n{result['text']}\n")
        if result.get("usage"):
            u = result["usage"]
            print(f"[tokens: {u.get('total_tokens', '?')} | turns: {result['turns']}]\n")


def cmd_claim(args):
    from claims.runner import run_claim, run_all, list_claims, summarize
    from state.store import StateStore

    store = StateStore()

    if args.list_claims:
        print("Available claims:")
        for name in list_claims():
            print(f"  {name}")
        return

    if args.claim == "all":
        results = run_all(n_trials=args.trials, state_store=store)
        print(summarize(results))
        return

    report = run_claim(args.claim, n_trials=args.trials, state_store=store,
                       logprobs=args.logprobs)
    if report is None:
        print(f"Claim '{args.claim}' not found. Use --list to see available claims.")
        return

    print(f"\n{'='*50}")
    print(f"Claim: {report.claim_title}")
    print(f"Trials: {report.total_trials}, Errors: {len(report.errors)}")
    print(f"Verdict: {report.verdict}")
    if report.effect_size:
        print(f"Effect size: {report.effect_size:.3f}")
    print(f"Metrics: {report.metrics}")
    print(f"{'='*50}")


def cmd_health(args):
    from config.defaults import DEFAULT_RULES, DEFAULT_CONSTRAINT_PROBES, EVAL_PERSONAS
    from state.store import StateStore
    from state.flags import FlagManager

    store = StateStore()
    flags = FlagManager(store)
    print("paper-validator v0.1.0 — health check")
    print(f"  Rules: {len(DEFAULT_RULES)}")
    print(f"  Probes: {len(DEFAULT_CONSTRAINT_PROBES)}")
    print(f"  Personas: {len(EVAL_PERSONAS)}")
    print(f"  Flags: {flags.active_flags()}")
    print(f"  Sessions: {store.get_counter('sessions')}")
    print("  State: OK")


def main():
    parser = argparse.ArgumentParser(description="paper-validator")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("interactive", aliases=["i"], help="Interactive agent mode")
    pc = sub.add_parser("claim", aliases=["c"], help="Run claim experiment")
    pc.add_argument("--claim", default="l1-visibility", help="Claim name or 'all'")
    pc.add_argument("--trials", "-n", type=int, default=30)
    pc.add_argument("--list", dest="list_claims", action="store_true",
                    help="List available claims")
    pc.add_argument("--logprobs", action="store_true",
                    help="Request logprobs from API")
    sub.add_parser("health", aliases=["h"], help="Run health check")
    args = parser.parse_args()

    store = None
    try:
        from state.store import StateStore
        store = StateStore()
        store.increment("sessions")
    except Exception:
        pass

    try:
        if args.command in ("interactive", "i"):
            cmd_interactive(args)
        elif args.command in ("claim", "c"):
            cmd_claim(args)
        elif args.command in ("health", "h"):
            cmd_health(args)
        else:
            parser.print_help()
    finally:
        if store:
            store.log_audit("session_end", {"command": args.command or "help"})


if __name__ == "__main__":
    main()
