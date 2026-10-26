"""Pause policies: the three strategies the benchmark compares."""
from __future__ import annotations

from dataclasses import dataclass

from .calibration import ErrorScorer
from .tasks import Case


@dataclass
class PauseDecision:
    pause: bool
    score: float  # the uncertainty/error score the policy used
    reason: str


class NeverPausePolicy:
    """Baseline (a): the agent never asks a human anything."""

    name = "never_pause"

    def decide(self, case: Case) -> PauseDecision:
        return PauseDecision(False, case.raw_uncertainty, "policy=never_pause")


class BlanketPolicy:
    """Baseline (b): pause on everything above a fixed low raw threshold.

    This is what most hand-rolled HITL setups do: one hand-picked number on
    the model's self-reported confidence.
    """

    name = "blanket_fixed_threshold"

    def __init__(self, threshold: float = 0.30):
        self.threshold = threshold

    def decide(self, case: Case) -> PauseDecision:
        return PauseDecision(
            case.raw_uncertainty >= self.threshold,
            case.raw_uncertainty,
            f"raw_uncertainty={case.raw_uncertainty:.3f} vs fixed threshold={self.threshold}",
        )


class CalibratedPolicy:
    """Pausewise: pause when the *calibrated* error probability clears a
    threshold that was itself chosen on a calibration set by cost
    minimisation."""

    name = "pausewise_calibrated"

    def __init__(self, scorer: ErrorScorer, tau: float):
        self.scorer = scorer
        self.tau = tau

    def decide(self, case: Case) -> PauseDecision:
        p = self.scorer.predict_one(case)
        return PauseDecision(
            p >= self.tau,
            p,
            f"calibrated P(error)={p:.3f} vs calibrated tau={self.tau}",
        )
