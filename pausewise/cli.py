"""Command line interface.

  pausewise benchmark [--out results.json]
  pausewise demo      [--cases 12] [--policy calibrated|blanket|never] --out-dir DIR
  pausewise resume    --checkpoint CKPT --queue QUEUE.json --decision DECISION.json --audit AUDIT.jsonl
  pausewise verify-audit AUDIT.jsonl

``demo`` runs a queue until the agent pauses, then exits. A human reads the
request file, writes a decision JSON, and ``resume`` continues the *same* run
from the on-disk checkpoint — in a new process, exactly like a real approval
workflow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agent import PausewiseAgent
from .audit import AuditLog, verify_audit
from .benchmark import main as benchmark_main
from .calibration import ErrorScorer, choose_threshold
from .human import HumanDecision
from .policies import BlanketPolicy, CalibratedPolicy, NeverPausePolicy
from .tasks import Case, generate_cases

CAL_SEED = 20261026
CAL_SIZE = 800


def _build_policy(name: str, threshold: float = 0.30):
    if name == "never":
        return NeverPausePolicy()
    if name == "blanket":
        return BlanketPolicy(threshold=threshold)
    calibration = generate_cases(CAL_SIZE, CAL_SEED)
    scorer = ErrorScorer().fit(calibration)
    tau, _ = choose_threshold(calibration, scorer.predict(calibration))
    return CalibratedPolicy(scorer, tau)


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cases = generate_cases(args.cases, args.seed)
    (out / "queue.json").write_text(
        json.dumps([c.to_dict() for c in cases], indent=2)
    )
    policy = _build_policy(args.policy, args.threshold)
    (out / "config.json").write_text(
        json.dumps({"policy": args.policy, "threshold": args.threshold}, indent=2)
    )
    agent = PausewiseAgent(
        policy=policy, audit_log=AuditLog(out / "audit.jsonl"), checkpoint_dir=out
    )
    status, payload = agent.run_until_pause(cases, run_id="demo")
    if status == "done":
        print(f"run completed without pausing: {len(payload)} tasks")
    else:
        print(f"PAUSED. checkpoint: {payload}")
        print(f"request:    {out / 'demo_request.json'}")
        print(
            "write a decision JSON, then: pausewise resume "
            f"--checkpoint {payload} --queue {out / 'queue.json'} "
            f"--decision <decision.json> --audit {out / 'audit.jsonl'}"
        )
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    ckpt = Path(args.checkpoint)
    out = ckpt.parent
    config = json.loads((out / "config.json").read_text())
    policy = _build_policy(config["policy"], config.get("threshold", 0.30))
    cases = [Case.from_dict(d) for d in json.loads(Path(args.queue).read_text())]
    decision = HumanDecision.from_dict(json.loads(Path(args.decision).read_text()))
    agent = PausewiseAgent(
        policy=policy, audit_log=AuditLog(args.audit), checkpoint_dir=out
    )
    status, payload = agent.resume(ckpt, cases, decision)
    if status == "done":
        errors = sum(1 for r in payload if r.final_wrong)
        print(f"run completed: {len(payload)} tasks, {errors} final errors")
        print(f"audit: {verify_audit(args.audit)}")
    else:
        print(f"PAUSED again. checkpoint: {payload}")
        print(f"request: {out / 'demo_request.json'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pausewise")
    sub = parser.add_subparsers(dest="command", required=True)

    p_bench = sub.add_parser("benchmark", help="run the three-policy benchmark")
    p_bench.add_argument("--out", default="results.json")
    p_bench.add_argument("--calibration-size", type=int, default=CAL_SIZE)
    p_bench.add_argument("--test-size", type=int, default=1500)

    p_demo = sub.add_parser("demo", help="run a queue until the agent pauses")
    p_demo.add_argument("--cases", type=int, default=12)
    p_demo.add_argument("--seed", type=int, default=5)
    p_demo.add_argument("--policy", choices=["calibrated", "blanket", "never"],
                        default="calibrated")
    p_demo.add_argument("--threshold", type=float, default=0.30)
    p_demo.add_argument("--out-dir", default="pausewise_demo")

    p_resume = sub.add_parser("resume", help="resume a paused run from its checkpoint")
    p_resume.add_argument("--checkpoint", required=True)
    p_resume.add_argument("--queue", required=True)
    p_resume.add_argument("--decision", required=True)
    p_resume.add_argument("--audit", required=True)

    p_verify = sub.add_parser("verify-audit", help="verify an audit log")
    p_verify.add_argument("path")

    args = parser.parse_args(argv)
    if args.command == "benchmark":
        benchmark_main(args.out, {"calibration_size": args.calibration_size,
                                   "test_size": args.test_size})
        return 0
    if args.command == "demo":
        return cmd_demo(args)
    if args.command == "resume":
        return cmd_resume(args)
    if args.command == "verify-audit":
        print(json.dumps(verify_audit(args.path), indent=2))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
