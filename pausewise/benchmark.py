"""Benchmark: never-pause vs blanket fixed threshold vs pausewise calibrated.

All three policies run the *same* test cases through the real agent machinery
(real checkpoints, real audit logs, simulated human with stated parameters),
so the comparison is paired and every number below is measured, not modelled.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .agent import PausewiseAgent
from .audit import AuditLog, verify_audit
from .calibration import ErrorScorer, auc, choose_threshold, expected_calibration_error
from .human import SimulatedHuman
from .policies import BlanketPolicy, CalibratedPolicy, NeverPausePolicy
from .tasks import Case, generate_cases

CONFIG = {
    "calibration_size": 800,
    "calibration_seed": 20261026,
    "test_size": 1500,
    "test_seed": 20261027,
    "blanket_threshold": 0.30,
    "human_accuracy": 0.90,
    "human_false_override_rate": 0.03,
    "human_mean_response_s": 42.0,
    "human_seed": 777,
    "cost_error": 10.0,
    "cost_pause": 3.0,
}


def _run_policy(policy, cases: list[Case], run_id: str, workdir: Path, cfg: dict) -> dict:
    audit_path = workdir / f"audit_{run_id}.jsonl"
    agent = PausewiseAgent(
        policy=policy,
        audit_log=AuditLog(audit_path),
        checkpoint_dir=workdir / f"ckpt_{run_id}",
        human=SimulatedHuman(
            accuracy=cfg["human_accuracy"],
            false_override_rate=cfg["human_false_override_rate"],
            mean_response_s=cfg["human_mean_response_s"],
            seed=cfg["human_seed"],
        ),
    )
    results = agent.run(cases, run_id=run_id)
    audit = verify_audit(audit_path)
    # Mechanism slice: high-novelty cases are where the agent's raw
    # uncertainty is blind (see tasks.py), so this is where calibration
    # should earn its keep if the mechanism is real.
    slice_pairs = [(c, r) for c, r in zip(cases, results) if c.novelty >= 0.7]
    high_novelty = {
        "n": len(slice_pairs),
        "agent_errors": sum(1 for c, _ in slice_pairs if c.agent_wrong),
        "final_errors": sum(1 for _, r in slice_pairs if r.final_wrong),
        "pauses": sum(1 for _, r in slice_pairs if r.paused),
        "errors_prevented": sum(1 for _, r in slice_pairs if r.error_prevented),
    }
    n = len(results)
    pauses = sum(1 for r in results if r.paused)
    prevented = sum(1 for r in results if r.error_prevented)
    final_errors = sum(1 for r in results if r.final_wrong)
    completed = sum(1 for r in results if r.status == "completed")
    human_time = sum(r.human_time_s for r in results)
    return {
        "policy": policy.name,
        "tasks": n,
        "agent_errors_without_intervention": sum(1 for r in results if r.agent_wrong),
        "final_errors": final_errors,
        "error_rate": final_errors / n,
        "error_rate_completed_only": (final_errors / completed) if completed else 0.0,
        "pauses": pauses,
        "pause_rate": pauses / n,
        "total_human_time_s": round(human_time, 1),
        "mean_human_response_s": round(human_time / pauses, 2) if pauses else 0.0,
        "errors_prevented": prevented,
        "pause_precision": (prevented / pauses) if pauses else 0.0,
        "false_overrides": sum(1 for r in results if r.false_override),
        "aborted": sum(1 for r in results if r.status == "aborted"),
        "audit_events": audit["events"],
        "audit_chain_valid": audit["chain_valid"],
        "audit_completeness": audit["completeness"],
        "high_novelty_slice": high_novelty,
    }


def run_benchmark(cfg: dict | None = None) -> dict:
    cfg = {**CONFIG, **(cfg or {})}
    calibration = generate_cases(cfg["calibration_size"], cfg["calibration_seed"])
    test = generate_cases(cfg["test_size"], cfg["test_seed"])

    scorer = ErrorScorer().fit(calibration)
    cal_probs = scorer.predict(calibration)
    tau, tau_stats = choose_threshold(
        calibration,
        cal_probs,
        human_accuracy=cfg["human_accuracy"],
        cost_error=cfg["cost_error"],
        cost_pause=cfg["cost_pause"],
    )

    test_labels = [c.agent_wrong for c in test]
    test_probs = scorer.predict(test)
    scorer_quality = {
        "auc_raw_uncertainty_on_test": auc([c.raw_uncertainty for c in test], test_labels),
        "auc_calibrated_scorer_on_test": auc(test_probs, test_labels),
        "ece_calibrated_scorer_on_test": expected_calibration_error(test_probs, test_labels),
        "base_error_rate_test": sum(test_labels) / len(test_labels),
        "base_error_rate_calibration": sum(c.agent_wrong for c in calibration) / len(calibration),
    }

    policies = [
        NeverPausePolicy(),
        BlanketPolicy(threshold=cfg["blanket_threshold"]),
        CalibratedPolicy(scorer, tau),
    ]
    with tempfile.TemporaryDirectory(prefix="pausewise_bench_") as tmp:
        workdir = Path(tmp)
        runs = [
            _run_policy(policy, test, f"bench_{policy.name}", workdir, cfg)
            for policy in policies
        ]

    return {
        "config": cfg,
        "calibration": {"tau": tau, **tau_stats},
        "scorer_quality": scorer_quality,
        "policies": runs,
    }


def main(out_path: str = "results.json", cfg: dict | None = None) -> dict:
    results = run_benchmark(cfg)
    Path(out_path).write_text(json.dumps(results, indent=2))
    cal = results["calibration"]
    q = results["scorer_quality"]
    print(f"calibrated tau = {cal['tau']}  (calibration pause rate {cal['calibration_pause_rate']:.2%})")
    print(
        f"error-prediction AUC: raw uncertainty {q['auc_raw_uncertainty_on_test']:.3f} "
        f"-> calibrated scorer {q['auc_calibrated_scorer_on_test']:.3f} "
        f"(ECE {q['ece_calibrated_scorer_on_test']:.3f})"
    )
    header = f"{'policy':<26}{'err_rate':>9}{'pauses':>8}{'prevented':>10}{'pause_prec':>11}{'human_min':>10}{'audit':>7}"
    print(header)
    for r in results["policies"]:
        print(
            f"{r['policy']:<26}{r['error_rate']:>9.3f}{r['pauses']:>8}"
            f"{r['errors_prevented']:>10}{r['pause_precision']:>11.3f}"
            f"{r['total_human_time_s'] / 60:>10.1f}{r['audit_completeness']:>7.2f}"
        )
    print(f"wrote {out_path}")
    return results


if __name__ == "__main__":
    main()
