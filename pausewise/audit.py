"""Append-only, hash-chained audit log + completeness verifier.

Every step of every case is written to a JSONL log: task start, proposal,
uncertainty score, pause decision (with reason), state save, human request,
human decision, resume validation, state restore, action execution, task end.
Each record carries the SHA-256 of the previous record, so silent edits break
the chain and are detected by :func:`verify_audit`.

*Completeness* is measured, not asserted: the verifier checks that every
started task ended, and that every pause episode contains the full
pause -> save -> request -> decision -> validate -> restore chain in order.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

GENESIS = "0" * 64

PAUSE_CHAIN = [
    "pause_decision",
    "state_saved",
    "human_requested",
    "human_decision",
    "resume_validated",
    "state_restored",
]


def _hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class AuditLog:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seq = 0
        self.prev_hash = GENESIS
        if self.path.exists() and self.path.stat().st_size > 0:
            last = None
            with self.path.open() as fh:
                for line in fh:
                    last = json.loads(line)
            if last is not None:
                self.seq = last["seq"]
                self.prev_hash = last["hash"]

    def emit(self, event: str, run_id: str, case_id: str | None = None, **data) -> dict:
        self.seq += 1
        record = {
            "seq": self.seq,
            "ts": time.time(),
            "event": event,
            "run_id": run_id,
            "case_id": case_id,
            "data": data,
            "prev_hash": self.prev_hash,
        }
        record["hash"] = _hash({k: v for k, v in record.items() if k != "hash"})
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, sort_keys=True) + "\n")
        self.prev_hash = record["hash"]
        return record


def verify_audit(path: str | Path) -> dict:
    """Verify hash-chain integrity and audit-trail completeness."""
    records = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    chain_valid = True
    prev = GENESIS
    for rec in records:
        if rec.get("prev_hash") != prev:
            chain_valid = False
            break
        expect = _hash({k: v for k, v in rec.items() if k != "hash"})
        if expect != rec.get("hash"):
            chain_valid = False
            break
        prev = rec["hash"]

    # Completeness requirements.
    requirements = 0
    satisfied = 0
    by_case: dict[str, list[dict]] = {}
    for rec in records:
        if rec.get("case_id"):
            by_case.setdefault(rec["case_id"], []).append(rec)

    for case_id, events in by_case.items():
        kinds = [e["event"] for e in events]
        if "task_start" in kinds:
            requirements += 1
            if "task_end" in kinds:
                satisfied += 1
            # Every pause episode must carry the full chain, in order.
            pause_points = [
                i for i, e in enumerate(events)
                if e["event"] == "pause_decision" and e["data"].get("pause")
            ]
            for start in pause_points:
                requirements += 1
                tail = [e["event"] for e in events[start:]]
                pos = -1
                ok = True
                for needed in PAUSE_CHAIN:
                    try:
                        pos = tail.index(needed, pos + 1)
                    except ValueError:
                        ok = False
                        break
                if ok:
                    satisfied += 1

    return {
        "events": len(records),
        "chain_valid": chain_valid,
        "requirements": requirements,
        "requirements_satisfied": satisfied,
        "completeness": (satisfied / requirements) if requirements else 1.0,
    }
