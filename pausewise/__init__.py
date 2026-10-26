"""pausewise — calibrated human-in-the-loop pausing for agents.

Pause when uncertainty actually predicts mistakes: the pause threshold is
learned from a calibration set by measuring error-prediction accuracy, not
hand-picked. Real pause/resume state save/restore, a full hash-chained audit
trail, and a benchmark that measures task error rate, human burden, and pause
precision (errors prevented per pause) against never-pause and
blanket-threshold baselines.
"""

from .agent import PausewiseAgent, RunState, validate_decision
from .audit import AuditLog, verify_audit
from .calibration import ErrorScorer, choose_threshold
from .human import HumanDecision, SimulatedHuman
from .policies import BlanketPolicy, CalibratedPolicy, NeverPausePolicy
from .tasks import Case, generate_cases

__version__ = "0.1.0"

__all__ = [
    "PausewiseAgent",
    "RunState",
    "validate_decision",
    "AuditLog",
    "verify_audit",
    "ErrorScorer",
    "choose_threshold",
    "HumanDecision",
    "SimulatedHuman",
    "BlanketPolicy",
    "CalibratedPolicy",
    "NeverPausePolicy",
    "Case",
    "generate_cases",
]
