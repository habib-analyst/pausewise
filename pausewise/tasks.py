"""Scripted agent tasks with known ground truth.

Each case is one consequential action an agent is about to take (refund,
export, record deletion, ...). The case generator is a *documented simulation*:
the agent's probability of being wrong is a logistic function of three
observable difficulty features (ambiguity, evidence gap, novelty) plus noise.

The agent's own self-reported uncertainty is deliberately *miscalibrated* in a
specific, realistic way: it is driven by ambiguity and evidence gap only and
is blind to novelty — the classic failure mode where an agent is confidently
wrong on out-of-distribution requests. That blind spot is the phenomenon
pausewise exists to catch, and it is stated openly here and in PAPER.md rather
than hidden.
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass

ACTIONS = [
    "approve_refund",
    "deny_refund",
    "issue_credit",
    "export_data",
    "update_record",
    "delete_record",
    "close_account",
    "escalate_ticket",
]


def _sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


@dataclass
class Case:
    case_id: str
    ambiguity: float
    evidence_gap: float
    novelty: float
    stakes: float
    p_error: float  # latent ground-truth error probability used by the generator
    agent_wrong: bool  # ground truth: is the agent's proposal wrong?
    raw_uncertainty: float  # the agent's own (miscalibrated) uncertainty signal
    proposed_action: str
    correct_action: str

    def scorer_features(self) -> list[float]:
        return [
            self.raw_uncertainty,
            self.ambiguity,
            self.evidence_gap,
            self.novelty,
            self.stakes,
        ]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Case":
        return cls(**d)


def generate_cases(n: int, seed: int) -> list[Case]:
    """Deterministically generate ``n`` scripted cases."""
    rng = random.Random(seed)
    cases: list[Case] = []
    for i in range(n):
        amb = rng.random()
        gap = rng.random()
        nov = rng.random()
        stakes = rng.random()
        noise = rng.gauss(0.0, 0.4)
        p_err = _sigmoid(-3.8 + 2.2 * amb + 2.0 * gap + 1.8 * nov + noise)
        wrong = rng.random() < p_err
        # The agent's raw uncertainty ignores novelty entirely and is shifted
        # toward overconfidence. This is the miscalibration under study.
        raw_u = _sigmoid(-2.0 + 1.5 * amb + 1.2 * gap + rng.gauss(0.0, 0.5))
        raw_u = min(0.99, max(0.01, raw_u))
        correct = rng.choice(ACTIONS)
        if wrong:
            proposed = rng.choice([a for a in ACTIONS if a != correct])
        else:
            proposed = correct
        cases.append(
            Case(
                case_id=f"case-{i:05d}",
                ambiguity=amb,
                evidence_gap=gap,
                novelty=nov,
                stakes=stakes,
                p_error=p_err,
                agent_wrong=wrong,
                raw_uncertainty=raw_u,
                proposed_action=proposed,
                correct_action=correct,
            )
        )
    return cases
