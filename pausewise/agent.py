"""The pausewise agent: uncertainty -> pause -> human -> validated resume.

The pause/resume machinery is real, not simulated. When the policy decides to
pause, the agent serialises its full run state (progress, results so far, the
pending case and the pause rationale) to a JSON checkpoint on disk and writes
a human-request file. Resuming — whether in the same process via a human
channel, or in a *different* process via ``pausewise resume`` — loads that
checkpoint back, validates the human's decision against the task schema, and
only then continues. Invalid decisions never execute: they are logged and the
task falls back to the safe default (reject / no action).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .audit import AuditLog
from .human import HumanDecision, SimulatedHuman
from .policies import PauseDecision
from .tasks import ACTIONS, Case


@dataclass
class CaseResult:
    case_id: str
    paused: bool
    agent_wrong: bool
    final_wrong: bool
    status: str  # "completed" | "aborted"
    error_prevented: bool = False
    false_override: bool = False
    human_time_s: float = 0.0


@dataclass
class RunState:
    run_id: str
    next_index: int = 0
    pending: dict | None = None  # {"case": {...}, "score": float, "reason": str}
    results: list[dict] = field(default_factory=list)
    pauses: int = 0
    total_human_time_s: float = 0.0

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, text: str) -> "RunState":
        d = json.loads(text)
        return cls(**d)


def validate_decision(decision: HumanDecision, case: Case) -> tuple[bool, str]:
    """Schema-validate a human decision before it is allowed to resume a run."""
    if decision.decision not in ("approve", "correct", "reject"):
        return False, f"unknown decision {decision.decision!r}"
    if decision.decision == "correct":
        if decision.corrected_action not in ACTIONS:
            return False, f"corrected_action {decision.corrected_action!r} not in action set"
        if decision.corrected_action == case.proposed_action:
            return False, "correction identical to the proposal is not a correction"
    if decision.decision == "approve" and decision.corrected_action is not None:
        return False, "approve must not carry a corrected_action"
    return True, "ok"


class PausewiseAgent:
    def __init__(
        self,
        policy,
        audit_log: AuditLog,
        checkpoint_dir: str | Path,
        human: SimulatedHuman | None = None,
    ):
        self.policy = policy
        self.audit = audit_log
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.human = human

    # -- checkpoint helpers -------------------------------------------------
    def _checkpoint_path(self, run_id: str) -> Path:
        return self.checkpoint_dir / f"{run_id}_checkpoint.json"

    def _save_checkpoint(self, state: RunState) -> Path:
        path = self._checkpoint_path(state.run_id)
        path.write_text(state.to_json())
        return path

    def _restore(self, path: Path) -> RunState:
        return RunState.from_json(path.read_text())

    # -- core loop ----------------------------------------------------------
    def _begin_case(self, state: RunState, case: Case) -> bool:
        """Score a case and apply the pause policy.

        Returns True if the case paused (state.pending is set and a checkpoint
        has been saved), False if it executed immediately.
        """
        run_id = state.run_id
        self.audit.emit("task_start", run_id, case.case_id)
        self.audit.emit(
            "proposal", run_id, case.case_id,
            action=case.proposed_action, raw_uncertainty=case.raw_uncertainty,
        )
        decision: PauseDecision = self.policy.decide(case)
        self.audit.emit(
            "uncertainty_scored", run_id, case.case_id,
            policy=self.policy.name, score=decision.score,
        )
        self.audit.emit(
            "pause_decision", run_id, case.case_id,
            pause=decision.pause, score=decision.score, reason=decision.reason,
        )
        if not decision.pause:
            self._execute(state, case, final_action=case.proposed_action, paused=False)
            return False
        state.pending = {
            "case": case.to_dict(),
            "score": decision.score,
            "reason": decision.reason,
        }
        state.pauses += 1
        ckpt = self._save_checkpoint(state)
        self.audit.emit("state_saved", run_id, case.case_id, checkpoint=str(ckpt))
        request = {
            "run_id": run_id,
            "case": case.to_dict(),
            "proposed_action": case.proposed_action,
            "score": decision.score,
            "reason": decision.reason,
            "checkpoint": str(ckpt),
            "instructions": (
                "Write a JSON decision file: {\"decision\": \"approve\"|\"correct\"|\"reject\", "
                "\"corrected_action\": <required iff correct, one of "
                + ", ".join(ACTIONS)
                + ">, \"reviewer\": \"<your name>\"} and resume with: "
                "pausewise resume --checkpoint <checkpoint> --queue <queue.json> "
                "--decision <decision.json> --audit <audit.jsonl>"
            ),
        }
        (self.checkpoint_dir / f"{run_id}_request.json").write_text(
            json.dumps(request, indent=2)
        )
        self.audit.emit("human_requested", run_id, case.case_id, checkpoint=str(ckpt))
        return True

    def _execute(
        self,
        state: RunState,
        case: Case,
        final_action: str | None,
        paused: bool,
        human_time_s: float = 0.0,
    ) -> None:
        if final_action is None:
            result = CaseResult(
                case_id=case.case_id, paused=paused, agent_wrong=case.agent_wrong,
                final_wrong=False, status="aborted", human_time_s=human_time_s,
            )
            self.audit.emit("task_end", state.run_id, case.case_id, status="aborted")
        else:
            final_wrong = final_action != case.correct_action
            result = CaseResult(
                case_id=case.case_id,
                paused=paused,
                agent_wrong=case.agent_wrong,
                final_wrong=final_wrong,
                status="completed",
                error_prevented=bool(case.agent_wrong and paused and not final_wrong),
                false_override=bool((not case.agent_wrong) and paused and final_wrong),
                human_time_s=human_time_s,
            )
            self.audit.emit(
                "action_executed", state.run_id, case.case_id, action=final_action
            )
            self.audit.emit(
                "task_end", state.run_id, case.case_id,
                status="completed", final_wrong=final_wrong,
            )
        state.results.append(asdict(result))
        state.total_human_time_s += human_time_s

    def _apply_decision(self, state: RunState, decision: HumanDecision) -> None:
        """Validate a human decision and resume the pending case from disk state."""
        pending = state.pending
        assert pending is not None, "no pending case to resume"
        case = Case.from_dict(pending["case"])
        run_id = state.run_id
        self.audit.emit(
            "human_decision", run_id, case.case_id,
            decision=decision.decision, corrected_action=decision.corrected_action,
            reviewer=decision.reviewer, response_s=decision.response_s,
        )
        ok, why = validate_decision(decision, case)
        self.audit.emit(
            "resume_validated", run_id, case.case_id, ok=ok, detail=why
        )
        if not ok:
            # Safe default: an invalid decision never executes an action.
            decision = HumanDecision("reject", reviewer=decision.reviewer,
                                     note=f"invalid decision rejected: {why}")
        # Real restore: reload the state from the checkpoint file on disk and
        # confirm it is the state that was saved at pause time.
        restored = self._restore(self._checkpoint_path(run_id))
        assert restored.pending is not None
        assert restored.pending["case"]["case_id"] == case.case_id
        self.audit.emit("state_restored", run_id, case.case_id)
        state.pending = None
        state.next_index += 1
        if decision.decision == "approve":
            self._execute(state, case, case.proposed_action, paused=True,
                          human_time_s=decision.response_s)
        elif decision.decision == "correct":
            self._execute(state, case, decision.corrected_action, paused=True,
                          human_time_s=decision.response_s)
        else:  # reject
            self._execute(state, case, None, paused=True,
                          human_time_s=decision.response_s)

    # -- public drivers -----------------------------------------------------
    def run(self, cases: list[Case], run_id: str) -> list[CaseResult]:
        """Run to completion, consulting the attached human channel at pauses.

        The pause path still round-trips through the on-disk checkpoint:
        state is saved, the human is asked, the checkpoint is reloaded and
        verified, and only then does execution continue.
        """
        if self.human is None:
            raise ValueError("run() needs a human channel; use run_until_pause()/resume() otherwise")
        self.audit.emit("run_start", run_id, policy=self.policy.name, tasks=len(cases))
        state = RunState(run_id=run_id)
        while state.next_index < len(cases):
            case = cases[state.next_index]
            if self._begin_case(state, case):
                decision = self.human.review(case, case.proposed_action)
                self._apply_decision(state, decision)
            else:
                state.next_index += 1
        self.audit.emit("run_end", run_id, tasks=len(state.results), pauses=state.pauses)
        return [CaseResult(**r) for r in state.results]

    def run_until_pause(
        self, cases: list[Case], run_id: str
    ) -> tuple[str, Path | list[CaseResult]]:
        """Run until the first pause (or completion) with no human attached.

        Returns ("paused", checkpoint_path) or ("done", results).
        """
        self.audit.emit("run_start", run_id, policy=self.policy.name, tasks=len(cases))
        state = RunState(run_id=run_id)
        while state.next_index < len(cases):
            case = cases[state.next_index]
            if self._begin_case(state, case):
                return "paused", self._checkpoint_path(run_id)
            state.next_index += 1
        self.audit.emit("run_end", run_id, tasks=len(state.results), pauses=state.pauses)
        return "done", [CaseResult(**r) for r in state.results]

    def resume(
        self, checkpoint_path: str | Path, cases: list[Case], decision: HumanDecision
    ) -> tuple[str, Path | list[CaseResult]]:
        """Resume a paused run from its checkpoint with a validated decision.

        Continues until the next pause or completion; returns the same tuple
        as :meth:`run_until_pause`.
        """
        state = self._restore(Path(checkpoint_path))
        if state.pending is None:
            raise ValueError("checkpoint has no pending case")
        self._apply_decision(state, decision)
        while state.next_index < len(cases):
            case = cases[state.next_index]
            if self._begin_case(state, case):
                return "paused", self._checkpoint_path(state.run_id)
            state.next_index += 1
        self.audit.emit(
            "run_end", state.run_id, tasks=len(state.results), pauses=state.pauses
        )
        return "done", [CaseResult(**r) for r in state.results]
