# pausewise: Calibrated Pausing for Human-in-the-Loop Agents

**Habib Ur Rehman** — 2026. Code and full result data: this repository (`results.json`).

## Abstract

Human-in-the-loop (HITL) agents typically pause for human approval on a fixed,
hand-picked confidence threshold, or on every action a rule flags as risky.
Neither asks the only question that matters: *does this uncertainty signal
actually predict mistakes, at this threshold, on this workload?* pausewise
treats pausing as a calibrated decision. An error scorer is fit on a
calibration set of cases with known outcomes; the pause threshold is then
selected on that same calibration set by explicit cost minimisation, using the
human approver's measured/assumed catch rate. On a paired benchmark of 1,500
scripted tasks with known ground truth, calibrated pausing dominates the
standard blanket fixed-threshold baseline on every headline metric
simultaneously: final error rate 0.123 vs. 0.140, pauses 870 vs. 906, errors
prevented 381 vs. 361, pause precision 0.438 vs. 0.398, and 6.4% less human
time — with a complete, hash-chained audit trail (completeness 1.00) as a
measured property of the run, not an assertion.

## 1. Method

### 1.1 The task simulation (stated openly)

Each task is one consequential agent action (refund approval, data export,
record deletion, …) drawn from a documented generative model
(`pausewise/tasks.py`). A case has three difficulty features — ambiguity,
evidence gap, novelty — each uniform on [0, 1], plus a stakes feature. The
agent's latent error probability is

  p_err = σ(−3.8 + 2.2·ambiguity + 2.0·evidence_gap + 1.8·novelty + ε),  ε ~ N(0, 0.4),

and the agent's proposal is wrong with probability p_err. The base error rate
is 34.8% on the calibration set and 36.7% on the test set — a hard queue, in
the spirit of benchmarks where frontier agents collapse once information is
incomplete.

The agent's self-reported uncertainty is *deliberately miscalibrated* in one
specific, realistic way: it is a function of ambiguity and evidence gap only,
shifted toward overconfidence, and **blind to novelty**. Agents being
confidently wrong on out-of-distribution requests is the failure mode under
study; it is planted openly so the mechanism can be measured, and §3 reports a
slice analysis that isolates exactly it.

### 1.2 The calibrated pause decision

An error scorer (logistic regression over [raw uncertainty, ambiguity,
evidence gap, novelty, stakes], standard-library implementation, deterministic
full-batch gradient descent) is fit on the calibration set (n = 800, seed
20261026). The pause threshold τ is chosen on the calibration set to minimise
expected cost per case,

  cost(τ) = n_paused·C_pause + C_error·(uncaught errors + paused errors·(1 − h)),

with C_error = 10, C_pause = 3 (human attention priced at 30% of an error —
a stated modelling choice), and h = 0.90 the human catch rate. This yielded
**τ = 0.29**, a calibration pause rate of 56.1%, and a calibration-estimated
pause precision of 0.437.

### 1.3 The agent

At each task the agent scores the proposal, applies the policy, and on a pause:
serialises its full run state to a JSON checkpoint on disk, writes a human
request file, and stops. Resuming reloads the checkpoint, **validates the
human's decision against the action schema** (unknown actions,
approve-with-correction, and no-op "corrections" are rejected and fall back to
the safe default of no action), restores state, and continues. This works
across processes (`pausewise demo` / `pausewise resume` CLI); the benchmark
exercises the same save/restore path in-process. Every step is written to a
SHA-256 hash-chained JSONL audit log.

### 1.4 Policies compared (paired, same 1,500 test tasks, seed 20261027)

- **never_pause** — the agent never asks a human.
- **blanket_fixed_threshold** — pause whenever raw uncertainty ≥ 0.30, the
  conventional hand-picked setup.
- **pausewise_calibrated** — pause when calibrated P(error) ≥ τ = 0.29.

### 1.5 The simulated human

An oracle with stated parameters: catches a wrong proposal with probability
0.90; overrides a correct proposal with a wrong action with probability 0.03;
response time log-normal with a 42 s mean parameter (measured means ≈ 47 s).
Identical seed (777) for every policy.

### 1.6 Metrics

Final task error rate; number of pauses (human burden); total human response
time; errors prevented (wrong proposals the pause-and-review turned right);
**pause precision** = errors prevented / pauses; false overrides (correct
proposals the human broke); audit-trail completeness = fraction of required
audit events present in order (every started task ended; every pause episode
carrying the full pause → save → request → decision → validate → restore
chain), with hash-chain validity checked independently.

## 2. Results

| Policy | Error rate | Pauses | Prevented | Pause precision | False overrides | Human time | Audit completeness |
|---|---|---|---|---|---|---|---|
| Never pause | 0.367 | 0 | 0 | — | 0 | 0 s | 1.00 (9,002 events) |
| Blanket (raw ≥ 0.30) | 0.140 | 906 | 361 | 0.398 | 21 | 43,430 s | 1.00 (13,532 events) |
| **pausewise calibrated** | **0.123** | **870** | **381** | **0.438** | **16** | **40,652 s** | **1.00 (13,352 events)** |

Scorer quality on the held-out test set: the calibrated scorer predicts agent
errors with **AUC 0.734**, vs. **0.642** for the agent's raw uncertainty;
expected calibration error **0.025**. The calibration set's estimate of test
pause precision was 0.437; the test measured 0.438 — the calibration procedure
generalised essentially exactly within this simulation.

Relative to the blanket baseline, calibrated pausing gives 12% lower error
rate with 4% fewer pauses, 5.5% more errors prevented, 10% higher pause
precision, 24% fewer false overrides, and 6.4% less human time. No metric was
traded off against another.

## 3. Where the win comes from: the novelty slice

On test cases with novelty ≥ 0.7 (455 tasks, 218 would-be errors) — precisely
where raw uncertainty is blind:

| Policy | Final errors on slice | Errors prevented on slice | Pauses on slice |
|---|---|---|---|
| Never pause | 218 | 0 | 0 |
| Blanket | 81 | 142 | 277 |
| **pausewise calibrated** | **42** | **182** | 402 |

The blanket threshold cannot see these errors coming, so it under-pauses
exactly there (277/455) while over-pausing on routine cases elsewhere. The
calibrated scorer, having learned from the calibration set that novelty
predicts errors, concentrates its interruptions where they pay. This slice is
the mechanism, measured — not a narrative imposed on aggregate numbers.

## 4. Honest limits

- **Simulated human.** The approver is an oracle with assumed accuracy (0.90),
  false-override rate (0.03) and response-time distribution. Real reviewers
  vary by fatigue, expertise and interface; every headline number is
  conditional on these parameters, and the threshold itself depends on h.
  Sensitivity to the cost ratio was checked during development: cheaper
  pauses (C_pause = 2) make the calibrated policy pause more than the blanket
  baseline at a lower error rate; dearer pauses (C_pause = 5) push pause
  precision to ~0.66 at a higher error rate. The reported configuration is the
  middle of that frontier, not a cherry-picked endpoint — but it is one point
  on a frontier, and deployments should pick theirs from their own costs.
- **Scripted tasks, stated generator.** Cases come from the logistic
  generative model of §1.1, not from a live LLM or production logs. The
  planted novelty blind spot is the phenomenon under study; a real agent's
  miscalibration will have a different shape, which is exactly why the
  threshold must be re-fit per deployment rather than copied (τ = 0.29 is an
  output, not a recommendation).
- **No distribution shift.** Calibration and test sets are drawn from the
  same generator. The near-exact precision estimate (0.437 → 0.438) would
  degrade under shift; recalibration cadence is future work.
- **Threshold model simplifications.** τ selection ignores false overrides
  (measured in the benchmark: 16–21 per 1,500 tasks) and treats h as constant.
- **One domain, one scorer.** A single synthetic ops-action domain and a
  logistic scorer. The contribution is the calibration procedure and the
  measured comparison, not the scorer's sophistication.

## 5. Relation to prior work

Learning-to-defer and selective prediction (Madras et al.; Mozannar &
Sontag; conformal deferral variants) study *classifier* abstention/deferral,
largely as training objectives. Agent frameworks (e.g. LangGraph interrupts)
provide pause/resume mechanics with manual or fixed triggers, and no
calibration of when to interrupt or measurement of pause quality. Habib Ur
Rehman's own pauseproof covers the complementary forensic-audit layer with a
fixed threshold. pausewise's contribution is the combination aimed at working
agent systems: a calibration-set-fitted error scorer, a cost-chosen threshold
that accounts for human catch rate, real checkpoint save/restore with
validated resume, and pause precision plus audit completeness as first-class
measured outcomes against the baselines practitioners actually use.

## 6. Reproducibility

Deterministic seeds throughout; standard library only; Python ≥ 3.10.

```bash
python -m pausewise benchmark   # regenerates results.json (~40 s, CPU)
python -m pytest                # 10 tests
```
