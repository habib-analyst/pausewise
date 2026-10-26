"""Calibration: learn *when uncertainty actually predicts mistakes*.

An :class:`ErrorScorer` (plain logistic regression, standard library only) is
fit on a calibration set of cases whose outcomes are known. The pause
threshold is then chosen on that same calibration set by explicit cost
minimisation: pausing costs human time, an uncaught error costs more, and a
pause only prevents an error if the human catches it (hence the human
accuracy term). Nothing about the threshold is hand-picked or copied from
another project.
"""
from __future__ import annotations

import math

from .tasks import Case

FEATURE_NAMES = ["raw_uncertainty", "ambiguity", "evidence_gap", "novelty", "stakes"]


def _sigmoid(z: float) -> float:
    if z < -35:
        return 0.0
    if z > 35:
        return 1.0
    return 1.0 / (1.0 + math.exp(-z))


class ErrorScorer:
    """Logistic regression from case features to P(agent is wrong)."""

    def __init__(self, epochs: int = 400, lr: float = 0.5, l2: float = 1e-3):
        self.epochs = epochs
        self.lr = lr
        self.l2 = l2
        self.mean: list[float] = []
        self.std: list[float] = []
        self.weights: list[float] = []
        self.bias: float = 0.0

    def fit(self, cases: list[Case]) -> "ErrorScorer":
        X = [c.scorer_features() for c in cases]
        y = [1.0 if c.agent_wrong else 0.0 for c in cases]
        d = len(X[0])
        n = len(X)
        self.mean = [sum(row[j] for row in X) / n for j in range(d)]
        self.std = [
            math.sqrt(sum((row[j] - self.mean[j]) ** 2 for row in X) / n) or 1.0
            for j in range(d)
        ]
        Xs = [[(row[j] - self.mean[j]) / self.std[j] for j in range(d)] for row in X]
        w = [0.0] * d
        b = 0.0
        for _ in range(self.epochs):
            gw = [0.0] * d
            gb = 0.0
            for row, target in zip(Xs, y):
                p = _sigmoid(sum(wj * xj for wj, xj in zip(w, row)) + b)
                err = p - target
                for j in range(d):
                    gw[j] += err * row[j]
                gb += err
            for j in range(d):
                w[j] -= self.lr * (gw[j] / n + self.l2 * w[j])
            b -= self.lr * (gb / n)
        self.weights = w
        self.bias = b
        return self

    def predict_one(self, case: Case) -> float:
        row = case.scorer_features()
        z = self.bias + sum(
            wj * (xj - m) / s
            for wj, xj, m, s in zip(self.weights, row, self.mean, self.std)
        )
        return _sigmoid(z)

    def predict(self, cases: list[Case]) -> list[float]:
        return [self.predict_one(c) for c in cases]


def auc(scores: list[float], labels: list[bool]) -> float:
    """Rank-based AUC with tie handling (Mann-Whitney)."""
    pos = [s for s, lab in zip(scores, labels) if lab]
    neg = [s for s, lab in zip(scores, labels) if not lab]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for q in neg:
            if p > q:
                wins += 1.0
            elif p == q:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def expected_calibration_error(probs: list[float], labels: list[bool], bins: int = 10) -> float:
    n = len(probs)
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, p in enumerate(probs) if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not idx:
            continue
        avg_p = sum(probs[i] for i in idx) / len(idx)
        avg_y = sum(1.0 if labels[i] else 0.0 for i in idx) / len(idx)
        ece += (len(idx) / n) * abs(avg_p - avg_y)
    return ece


def choose_threshold(
    cases: list[Case],
    probs: list[float],
    human_accuracy: float = 0.90,
    cost_error: float = 10.0,
    cost_pause: float = 3.0,
) -> tuple[float, dict]:
    """Pick the pause threshold minimising expected cost on the calibration set.

    Expected cost of a threshold tau, per case:
      - paused: cost_pause, plus the error cost for errors the human misses
        (probability 1 - human_accuracy for genuinely wrong proposals).
      - not paused: full error cost for wrong proposals.
    False overrides (a human breaking a correct proposal) are not modelled
    here; they are measured in the benchmark and discussed in PAPER.md.
    """
    best_tau, best_cost, best_stats = 0.5, float("inf"), {}
    for step in range(1, 100):
        tau = step / 100.0
        pauses = 0
        paused_errors = 0
        unpaused_errors = 0
        for c, p in zip(cases, probs):
            if p >= tau:
                pauses += 1
                if c.agent_wrong:
                    paused_errors += 1
            elif c.agent_wrong:
                unpaused_errors += 1
        expected_cost = (
            cost_pause * pauses
            + cost_error * (unpaused_errors + paused_errors * (1.0 - human_accuracy))
        ) / len(cases)
        if expected_cost < best_cost:
            best_cost = expected_cost
            best_tau = tau
            best_stats = {
                "tau": tau,
                "expected_cost_per_case": expected_cost,
                "calibration_pause_rate": pauses / len(cases),
                "calibration_paused_errors": paused_errors,
                "calibration_pause_precision_estimate": (
                    paused_errors * human_accuracy / pauses if pauses else 0.0
                ),
            }
    return best_tau, best_stats
