import json
from pathlib import Path

from pausewise.agent import PausewiseAgent, RunState, validate_decision
from pausewise.audit import AuditLog, verify_audit
from pausewise.calibration import ErrorScorer, auc, choose_threshold
from pausewise.human import HumanDecision, SimulatedHuman
from pausewise.policies import BlanketPolicy, CalibratedPolicy, NeverPausePolicy
from pausewise.tasks import ACTIONS, generate_cases


def _calibrated_policy():
    cal = generate_cases(400, 1234)
    scorer = ErrorScorer(epochs=200).fit(cal)
    tau, _ = choose_threshold(cal, scorer.predict(cal))
    return CalibratedPolicy(scorer, tau)


def test_scorer_beats_raw_uncertainty_at_predicting_errors():
    cal = generate_cases(500, 101)
    test = generate_cases(500, 202)
    scorer = ErrorScorer(epochs=200).fit(cal)
    labels = [c.agent_wrong for c in test]
    assert auc(scorer.predict(test), labels) > auc(
        [c.raw_uncertainty for c in test], labels
    )


def test_threshold_selection_is_interior_and_reduces_cost():
    cal = generate_cases(500, 101)
    scorer = ErrorScorer(epochs=200).fit(cal)
    probs = scorer.predict(cal)
    tau, stats = choose_threshold(cal, probs, cost_error=10.0, cost_pause=1.0)
    assert 0.0 < tau < 1.0
    # Cheaper than pausing on everything or nothing, on the calibration set.
    n = len(cal)
    errors = sum(c.agent_wrong for c in cal)
    cost_never = 10.0 * errors / n
    cost_always = (1.0 * n + 10.0 * errors * 0.1) / n
    assert stats["expected_cost_per_case"] <= min(cost_never, cost_always)


def test_never_pause_baseline_has_zero_pauses_and_base_error_rate(tmp_path):
    cases = generate_cases(60, 303)
    agent = PausewiseAgent(
        NeverPausePolicy(), AuditLog(tmp_path / "a.jsonl"), tmp_path / "ck",
        human=SimulatedHuman(),
    )
    results = agent.run(cases, run_id="t-never")
    assert all(not r.paused for r in results)
    assert sum(r.final_wrong for r in results) == sum(c.agent_wrong for c in cases)


def test_pause_resume_round_trip_with_fresh_agent_instance(tmp_path):
    # An always-pause blanket policy forces a pause on the very first case.
    cases = generate_cases(10, 404)
    audit = tmp_path / "a.jsonl"
    agent = PausewiseAgent(BlanketPolicy(threshold=0.0), AuditLog(audit), tmp_path)
    status, ckpt = agent.run_until_pause(cases, run_id="t-rt")
    assert status == "paused"
    state = RunState.from_json(Path(ckpt).read_text())
    assert state.pending is not None and state.next_index == 0

    # A brand-new agent instance (new process in real life) resumes from disk.
    pending_case = state.pending["case"]
    decision = HumanDecision(
        "correct", corrected_action=pending_case["correct_action"],
        reviewer="test-human",
    )
    agent2 = PausewiseAgent(BlanketPolicy(threshold=1.01), AuditLog(audit), tmp_path)
    status, results = agent2.resume(ckpt, cases, decision)
    assert status == "done"
    assert len(results) == len(cases)
    first = results[0]
    assert first.paused and not first.final_wrong  # the human's correction landed


def test_invalid_decision_falls_back_to_reject_and_is_logged(tmp_path):
    cases = generate_cases(5, 505)
    audit = tmp_path / "a.jsonl"
    agent = PausewiseAgent(BlanketPolicy(threshold=0.0), AuditLog(audit), tmp_path)
    status, ckpt = agent.run_until_pause(cases, run_id="t-inv")
    assert status == "paused"
    bad = HumanDecision("correct", corrected_action="not_a_real_action")
    agent2 = PausewiseAgent(BlanketPolicy(threshold=1.01), AuditLog(audit), tmp_path)
    status, results = agent2.resume(ckpt, cases, bad)
    assert status == "done"
    assert results[0].status == "aborted"  # invalid decision never executes
    report = verify_audit(audit)
    assert report["chain_valid"]


def test_validate_decision_schema():
    case = generate_cases(1, 606)[0]
    ok, _ = validate_decision(HumanDecision("approve"), case)
    assert ok
    ok, _ = validate_decision(
        HumanDecision("correct", corrected_action=case.correct_action), case
    )
    # correct_action differs from proposal only when the agent is wrong; if the
    # proposal already equals it, the "correction" is invalid by design.
    assert ok == (case.proposed_action != case.correct_action)
    ok, _ = validate_decision(HumanDecision("bogus"), case)
    assert not ok
    ok, _ = validate_decision(
        HumanDecision("approve", corrected_action=ACTIONS[0]), case
    )
    assert not ok


def test_audit_completeness_after_paused_run(tmp_path):
    cases = generate_cases(40, 707)
    audit = tmp_path / "a.jsonl"
    agent = PausewiseAgent(
        BlanketPolicy(threshold=0.5), AuditLog(audit), tmp_path / "ck",
        human=SimulatedHuman(seed=1),
    )
    agent.run(cases, run_id="t-audit")
    report = verify_audit(audit)
    assert report["chain_valid"]
    assert report["completeness"] == 1.0


def test_audit_tamper_breaks_chain(tmp_path):
    cases = generate_cases(10, 808)
    audit = tmp_path / "a.jsonl"
    agent = PausewiseAgent(
        NeverPausePolicy(), AuditLog(audit), tmp_path / "ck",
        human=SimulatedHuman(),
    )
    agent.run(cases, run_id="t-tamper")
    lines = audit.read_text().splitlines()
    rec = json.loads(lines[3])
    rec["data"]["action"] = "delete_record"  # silent edit
    lines[3] = json.dumps(rec, sort_keys=True)
    audit.write_text("\n".join(lines) + "\n")
    assert not verify_audit(audit)["chain_valid"]


def test_calibrated_run_end_to_end(tmp_path):
    cases = generate_cases(120, 909)
    audit = tmp_path / "a.jsonl"
    agent = PausewiseAgent(
        _calibrated_policy(), AuditLog(audit), tmp_path / "ck",
        human=SimulatedHuman(seed=2),
    )
    results = agent.run(cases, run_id="t-cal")
    assert len(results) == len(cases)
    assert verify_audit(audit)["completeness"] == 1.0


def test_cli_demo_and_resume_in_subprocesses(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "demo"
    r1 = subprocess.run(
        [sys.executable, "-m", "pausewise", "demo", "--cases", "8", "--seed", "3",
         "--policy", "blanket", "--threshold", "0.0", "--out-dir", str(out)],
        capture_output=True, text=True,
    )
    assert r1.returncode == 0, r1.stderr
    assert "PAUSED" in r1.stdout
    request = json.loads((out / "demo_request.json").read_text())
    decision = {
        "decision": "correct",
        "corrected_action": request["case"]["correct_action"],
        "reviewer": "cli-test",
    }
    (out / "decision.json").write_text(json.dumps(decision))
    r2 = subprocess.run(
        [sys.executable, "-m", "pausewise", "resume",
         "--checkpoint", str(out / "demo_checkpoint.json"),
         "--queue", str(out / "queue.json"),
         "--decision", str(out / "decision.json"),
         "--audit", str(out / "audit.jsonl")],
        capture_output=True, text=True,
    )
    assert r2.returncode == 0, r2.stderr
    # Remaining cases use blanket threshold 0.0, so it pauses again — that is
    # expected; the point is the cross-process resume worked and logged.
    assert "PAUSED again" in r2.stdout or "run completed" in r2.stdout
    report = verify_audit(out / "audit.jsonl")
    assert report["chain_valid"] and report["events"] > 0
