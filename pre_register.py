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


def embed_in_existing_data(data_path: str | Path,
                          reg_hash: str,
                          output_path: str | Path = "") -> dict:
    """Embed pre-registration hash into every record in existing data.
    Does NOT require API access — validates the embed→store→verify pipeline
    against collected trial data.

    Returns: {output, records_embedded, hash}
    """
    data_path = Path(data_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    registry = _load_registry()
    entry = registry.get(reg_hash, {})
    record_block = {
        "pre_reg_hash": reg_hash,
        "pre_reg_experiment": entry.get("experiment_name", ""),
        "pre_reg_timestamp": entry.get("timestamp", ""),
    }

    embedded = 0
    items = data if isinstance(data, list) else data.get("trials", data.get("results", []))
    for item in items:
        if isinstance(item, dict):
            item.update(record_block)
            embedded += 1

    # Also embed at top level for aggregate files
    if isinstance(data, dict):
        data["pre_registration"] = record_block

    output = Path(output_path) if output_path else data_path.with_suffix(".embedded.json")
    output.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    return {"output": str(output), "records_embedded": embedded, "hash": reg_hash}


def verify_embedded_records(data_path: str | Path, reg_hash: str = "") -> dict:
    """Verify all records in data file contain pre-registration hash.
    If reg_hash provided: also checks all hashes match (strict mode).

    Returns: {verified, total, has_hash, missing, wrong_hash, hash_found}
    """
    data_path = Path(data_path)
    data = json.loads(data_path.read_text(encoding="utf-8"))
    items = data if isinstance(data, list) else data.get("trials", data.get("results", []))
    total = 0
    has_hash = 0
    missing = 0
    wrong_hash = 0
    hashes_found = set()

    for item in items:
        if not isinstance(item, dict):
            continue
        total += 1
        h = item.get("pre_reg_hash", "")
        if h:
            has_hash += 1
            hashes_found.add(h)
            if reg_hash and h != reg_hash:
                wrong_hash += 1
        else:
            missing += 1

    result = {
        "verified": missing == 0 and wrong_hash == 0,
        "total": total,
        "has_hash": has_hash,
        "missing": missing,
        "wrong_hash": wrong_hash,
        "hashes_found": sorted(hashes_found),
    }
    if reg_hash:
        result["hash_match"] = reg_hash in hashes_found
        result["hash_uniform"] = len(hashes_found) == 1 and reg_hash in hashes_found
    return result


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

    em = sp.add_parser("embed", help="Embed hash into existing trial data")
    em.add_argument("--data", required=True)
    em.add_argument("--hash", required=True)
    em.add_argument("--output", default="")

    ch = sp.add_parser("check-embedded", help="Verify all records contain hash")
    ch.add_argument("--data", required=True)
    ch.add_argument("--hash", default="")

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
    elif a.cmd == "embed":
        r = embed_in_existing_data(a.data, a.hash, a.output)
        print(f"Embedded {r['records_embedded']} records with hash {r['hash']}")
        print(f"Output: {r['output']}")
    elif a.cmd == "check-embedded":
        r = verify_embedded_records(a.data, a.hash)
        status = "ALL VERIFIED" if r["verified"] else "FAILED"
        detail = f"({r['has_hash']}/{r['total']} have hash"
        if r["missing"]:
            detail += f", {r['missing']} MISSING"
        if r["wrong_hash"]:
            detail += f", {r['wrong_hash']} WRONG HASH"
        detail += ")"
        print(f"{status} {detail}")
        if a.hash:
            print(f"Hash match: {'YES' if r.get('hash_match') else 'NO'}")
            print(f"Hash uniform: {'YES' if r.get('hash_uniform') else 'NO — multiple hashes found'}")
        if r["hashes_found"] and not (a.hash and r.get("hash_uniform")):
            print(f"Hash(es) found: {', '.join(r['hashes_found'])}")
    else:
        p.print_help()
