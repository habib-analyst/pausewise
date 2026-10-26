"""Human approvers.

The benchmark uses a simulated oracle with *stated* parameters: it catches a
wrong proposal with probability ``accuracy``, occasionally overrides a correct
proposal with a wrong one (``false_override_rate``), and takes a
log-normally distributed amount of time to respond. Real deployments plug in
any object with the same ``review(case, proposed_action)`` method — see the
``pausewise demo`` / ``pausewise resume`` CLI for a file-based human channel
where an actual person writes the decision.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from .tasks import ACTIONS, Case


@dataclass
class HumanDecision:
    decision: str  # "approve" | "correct" | "reject"
    corrected_action: str | None = None
    response_s: float = 0.0
    reviewer: str = "simulated-human"
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "corrected_action": self.corrected_action,
            "response_s": self.response_s,
            "reviewer": self.reviewer,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "HumanDecision":
        return cls(
            decision=d["decision"],
            corrected_action=d.get("corrected_action"),
            response_s=float(d.get("response_s", 0.0)),
            reviewer=d.get("reviewer", "human"),
            note=d.get("note", ""),
        )


class SimulatedHuman:
    """Oracle approver with stated accuracy and response-time cost."""

    def __init__(
        self,
        accuracy: float = 0.90,
        false_override_rate: float = 0.03,
        mean_response_s: float = 42.0,
        seed: int = 777,
    ):
        self.accuracy = accuracy
        self.false_override_rate = false_override_rate
        self.mean_response_s = mean_response_s
        self.rng = random.Random(seed)

    def review(self, case: Case, proposed_action: str) -> HumanDecision:
        response_s = self.rng.lognormvariate(math.log(self.mean_response_s), 0.45)
        if case.agent_wrong:
            if self.rng.random() < self.accuracy:
                return HumanDecision(
                    "correct",
                    corrected_action=case.correct_action,
                    response_s=response_s,
                    note="caught the error",
                )
            return HumanDecision(
                "approve", response_s=response_s, note="missed the error"
            )
        # Proposal is correct.
        if self.rng.random() < self.false_override_rate:
            wrong = self.rng.choice([a for a in ACTIONS if a != case.correct_action])
            return HumanDecision(
                "correct",
                corrected_action=wrong,
                response_s=response_s,
                note="false override of a correct proposal",
            )
        return HumanDecision("approve", response_s=response_s, note="confirmed correct")
