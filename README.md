# pausewise

**Calibrated human-in-the-loop pausing for agents: pause when uncertainty actually predicts mistakes.**

Most human-in-the-loop agents pause on a hand-picked confidence threshold — or on everything that looks risky. Both waste the human: the blanket threshold interrupts constantly and *still* misses the errors the model's self-reported confidence can't see. pausewise calibrates the pause decision instead:

1. An **error scorer** is fit on a calibration set of past cases with known outcomes, learning which signals actually predict the agent being wrong.
2. The **pause threshold is chosen on that calibration set** by explicit cost minimisation (error cost vs. interruption cost, discounted by how often the human actually catches errors) — not guessed.
3. At run time the agent pauses only when the calibrated P(error) clears that threshold, saves its full state to an on-disk checkpoint, and resumes only after the human's decision passes schema validation.
4. **Everything is audit-logged** in a hash-chained JSONL trail, and audit *completeness* is a measured metric, not a claim.

Pure Python standard library. No dependencies, no APIs, no cost.

## Measured results

Scripted benchmark, 1,500 test tasks with known ground truth, simulated human approver (90% error-catch rate, 3% false-override rate, 42 s response-time parameter). All three policies run the *same* tasks through the same agent machinery. Full data: `results.json`, method and limits: `PAPER.md`.

| Policy | Final error rate | Pauses | Errors prevented | Pause precision | Human time |
|---|---|---|---|---|---|
| Never pause | 0.367 | 0 | 0 | — | 0 min |
| Blanket fixed threshold (raw ≥ 0.30) | 0.140 | 906 | 361 | 0.398 | 723.8 min |
| **pausewise calibrated (τ = 0.29)** | **0.123** | **870** | **381** | **0.438** | **677.5 min** |

Pause precision = errors prevented per pause. Calibrated pausing beats the blanket threshold on every metric at once: fewer interruptions, fewer final errors, more errors prevented, less human time. Audit-trail completeness was **1.00** (hash chain valid) for all three policies — 13,352 events for the calibrated run.

Where the win comes from: the agent's raw uncertainty is blind to *novelty* (a stated property of the simulation). On the high-novelty slice (455 tasks, 218 would-be errors), the blanket threshold leaves 81 errors standing; the calibrated policy leaves **42**, preventing 182 errors vs. the blanket's 142. The error scorer's error-prediction AUC is 0.734 vs. 0.642 for raw uncertainty (ECE 0.025). The threshold chosen on the calibration set predicted a pause precision of 0.437; the held-out test measured 0.438.

## Quickstart

Run the benchmark (writes `results.json`):

```bash
python -m pausewise benchmark
```

Drive a real pause/resume by hand — two processes, exactly like a production approval flow:

```bash
# process 1: run a queue until the agent pauses, then exit
python -m pausewise demo --cases 12 --out-dir demo_out
# read demo_out/demo_request.json, write your decision to decision.json:
#   {"decision": "correct", "corrected_action": "deny_refund", "reviewer": "you"}
# process 2: resume the same run from its on-disk checkpoint
python -m pausewise resume --checkpoint demo_out/demo_checkpoint.json \
    --queue demo_out/queue.json --decision decision.json \
    --audit demo_out/audit.jsonl
# verify the audit trail afterwards
python -m pausewise verify-audit demo_out/audit.jsonl
```

An invalid decision (unknown action, approve-with-correction, …) never executes: it is logged, and the task falls back to the safe default of no action.

## Library use

```python
from pausewise import (AuditLog, CalibratedPolicy, ErrorScorer,
                       PausewiseAgent, SimulatedHuman, choose_threshold,
                       generate_cases)

calibration = generate_cases(800, seed=20261026)      # your labelled history
scorer = ErrorScorer().fit(calibration)
tau, _ = choose_threshold(calibration, scorer.predict(calibration),
                          human_accuracy=0.90, cost_error=10.0, cost_pause=3.0)

agent = PausewiseAgent(
    policy=CalibratedPolicy(scorer, tau),
    audit_log=AuditLog("audit.jsonl"),
    checkpoint_dir="checkpoints",
    human=SimulatedHuman(accuracy=0.90),   # or your own review() channel
)
results = agent.run(generate_cases(1500, seed=20261027), run_id="run-1")
```

For a real human, drop the `human=` argument and use `run_until_pause()` / `resume()` (or the CLI) so a person supplies the decision between processes.

## How it works

- `pausewise/tasks.py` — the scripted task generator (documented simulation; the agent's error probability is a logistic function of ambiguity, evidence gap and novelty, and its raw uncertainty is deliberately blind to novelty).
- `pausewise/calibration.py` — logistic error scorer, AUC/ECE, cost-based threshold selection.
- `pausewise/policies.py` — never-pause, blanket fixed-threshold, and calibrated policies.
- `pausewise/agent.py` — the agent: pause → checkpoint save → human request → decision validation → state restore → resume.
- `pausewise/audit.py` — hash-chained audit log and a completeness verifier.
- `pausewise/human.py` — the simulated oracle approver with stated parameters; swap in any object with a `review()` method.
- `pausewise/benchmark.py` — the paired three-policy benchmark.

## Relation to pauseproof

[pauseproof](https://github.com/habib-analyst/pauseproof) is the forensic layer: tamper-evident audit for HITL agents with a fixed pause threshold. pausewise is the decision layer it doesn't have: *when* to pause, calibrated and measured. They are complementary — calibrate the pausing with pausewise, prove what happened with pauseproof-grade logging.

## Honest limits

The human is simulated with assumed parameters; the tasks are scripted from a stated generative model, not a live LLM; calibration and test data come from the same generator, so no distribution shift is tested. The threshold's cost units are a modelling choice. See `PAPER.md` for the full list before citing any number.

## License

MIT © Habib Ur Rehman
