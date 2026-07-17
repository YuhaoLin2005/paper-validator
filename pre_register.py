"""Pre-Registration via SHA256 + Provider Timestamps
====================================================
Inspired by Dipankar Sarkar (DEV.to comment, 2026-07-15):
  "Hash the pre-registration (sha256 over the script header, the conditions,
   and the scoring regexes), then write that hash into every trial record you
   send. The provider timestamps those. You can still amend the commit afterward,
   but you can't amend 600 API records held by someone who isn't you."

Integrated as 'pre-register' CLI subcommand and importable PreRegistry class.
Registry: pre_registry.json. Each entry: {hash, timestamp, hypothesis, ...}
"""

from __future__ import annotations

import hashlib, json, os, re, time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


REGISTRY_PATH = Path(__file__).parent / "pre_registry.json"
HASH_BYTES = 16  # First 16 bytes of SHA256 → 32-char hex


@dataclass
class PreRegEntry:
    hash: str
    timestamp: str
    experiment_name: str
    hypothesis: str
    conditions_hash: str
    scoring_rules_hash: str
    script_path: str
    devto_comment_url: str = ""

    def to_record(self) -> dict:
        return {
            "pre_reg_hash": self.hash,
            "pre_reg_timestamp": self.timestamp,
            "experiment": self.experiment_name,
        }


# ── Core functions ──

def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_BYTES * 2]


def extract_pre_registration(script_path: str | Path) -> dict:
    """Extract pre-registration data from an experiment script."""
    script_path = Path(script_path)
    source = script_path.read_text(encoding="utf-8")

    doc_match = re.search(r'"""(.*?)"""', source, re.DOTALL)
    hypothesis = doc_match.group(1).strip() if doc_match else ""

    cond_match = re.search(r'CONDITIONS\s*=\s*(\{.*?\n\})', source, re.DOTALL)
    conditions_text = cond_match.group(1).strip() if cond_match else ""

    scoring_patterns = re.findall(
        r're\.(?:compile|search|findall|match)\s*\(\s*r?[\'"](.+?)[\'"]', source)
    scoring_text = "\n".join(sorted(set(scoring_patterns)))

    return {
        "experiment_name": script_path.stem,
        "hypothesis": hypothesis,
        "conditions_text": conditions_text,
        "scoring_text": scoring_text,
    }


def register_experiment(script_path: str | Path,
                        devto_comment_url: str = "") -> PreRegEntry:
    """Hash and register an experiment. Returns PreRegEntry."""
    data = extract_pre_registration(script_path)

    composite = (
        data["hypothesis"] + "\n---CONDITIONS---\n" +
        data["conditions_text"] + "\n---SCORING---\n" +
        data["scoring_text"]
    )
    exp_hash = _sha256(composite)

    entry = PreRegEntry(
        hash=exp_hash,
        timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        experiment_name=data["experiment_name"],
        hypothesis=data["hypothesis"][:200],
        conditions_hash=_sha256(data["conditions_text"]),
        scoring_rules_hash=_sha256(data["scoring_text"]),
        script_path=str(Path(script_path).resolve()),
        devto_comment_url=devto_comment_url,
    )

    _save_entry(entry)
    return entry


def verify_experiment(reg_hash: str, data_path: str | Path) -> dict:
    """Verify collected data against pre-registered hash."""
    registry = _load_registry()
    entry_data = registry.get(reg_hash)
    if not entry_data:
        return {"match": False, "error": f"Hash {reg_hash} not in registry"}

    data_path = Path(data_path)
    if not data_path.exists():
        return {"match": False, "error": f"Data not found: {data_path}"}

    data_text = data_path.read_text(encoding="utf-8")
    data_hash = _sha256(data_text)

    return {
        "match": data_hash == reg_hash,
        "entry": entry_data,
        "data_hash": data_hash,
    }


def embed_in_record(reg_hash: str, record: dict) -> dict:
    """Embed pre-registration metadata into an API call record."""
    registry = _load_registry()
    entry_data = registry.get(reg_hash, {})
    record["pre_registration"] = {
        "hash": reg_hash,
        "experiment": entry_data.get("experiment_name", ""),
        "registered_at": entry_data.get("timestamp", ""),
    }
    return record


def export_devto_comment(reg_hash: str) -> str:
    """Generate DEV.to comment template for public timestamp."""
    registry = _load_registry()
    entry = registry.get(reg_hash, {})
    return f"""**Pre-registration hash**: `{reg_hash}`

- **Experiment**: {entry.get('experiment_name', 'N/A')}
- **Hypothesis**: {entry.get('hypothesis', 'N/A')[:200]}
- **Conditions SHA256**: `{entry.get('conditions_hash', 'N/A')}`
- **Scoring rules SHA256**: `{entry.get('scoring_rules_hash', 'N/A')}`
- **Registered**: {entry.get('timestamp', 'N/A')}

This comment serves as a public timestamp. The SHA256 was computed over the
full experiment config (hypothesis + conditions + scoring regexes) before data
collection. Verify: `python pre_register.py verify --hash {reg_hash} --data <results.json>`
"""


# ── Registry I/O ──

def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def _save_entry(entry: PreRegEntry) -> None:
    registry = _load_registry()
    registry[entry.hash] = asdict(entry)
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")


def list_registrations() -> list[dict]:
    registry = _load_registry()
    return [
        {"hash": h, "name": e.get("experiment_name", ""),
         "timestamp": e.get("timestamp", ""),
         "devto_url": e.get("devto_comment_url", "")}
        for h, e in sorted(registry.items(),
                          key=lambda x: x[1].get("timestamp", ""), reverse=True)
    ]


# ── CLI ──

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SHA256 Pre-Registration")
    sp = p.add_subparsers(dest="cmd")

    sp.add_parser("list", help="List all pre-registrations")

    r = sp.add_parser("register", help="Register experiment")
    r.add_argument("--script", required=True)
    r.add_argument("--devto-url", default="")

    v = sp.add_parser("verify", help="Verify data against hash")
    v.add_argument("--hash", required=True)
    v.add_argument("--data", required=True)

    e = sp.add_parser("export", help="Export DEV.to comment")
    e.add_argument("--hash", required=True)

    a = p.parse_args()

    if a.cmd == "list":
        regs = list_registrations()
        if not regs:
            print("No pre-registrations found.")
        for r in regs:
            print(f"  {r['hash']}  {r['name']}  ({r['timestamp']})")
    elif a.cmd == "register":
        entry = register_experiment(a.script, a.devto_url)
        print(f"Registered: {entry.hash}")
        print(f"  {entry.experiment_name}  ({entry.timestamp})")
        print(f"  Export: python pre_register.py export --hash {entry.hash}")
    elif a.cmd == "verify":
        r = verify_experiment(a.hash, a.data)
        print("VERIFIED" if r["match"] else f"MISMATCH: {r.get('error')}")
    elif a.cmd == "export":
        print(export_devto_comment(a.hash))
    else:
        p.print_help()
