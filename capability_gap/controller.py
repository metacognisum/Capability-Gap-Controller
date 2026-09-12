"""Assess, rank, execute, verify, diagnose again, and retain selected outcomes."""
from collections import Counter
from dataclasses import asdict
import time
from uuid import uuid4

from foresight import Controller as EvidenceController, State, observe
from foresight.core import supported

from .prediction import ForesightLayer
from .storage import CapabilityMap
from .types import ADAPTATIONS, Gap, Handler, Intervention, Kind, Policy, Task


FAILURE_GAPS = {
    "missing_knowledge": Gap.KNOWLEDGE, "reasoning_error": Gap.REASONING,
    "tool_error": Gap.TOOL, "context_overflow": Gap.CONTEXT,
    "domain_error": Gap.DOMAIN, "compute_exhausted": Gap.COMPUTE,
    "skill_error": Gap.PERSISTENT_SKILL,
}


class CapabilityController:
    def __init__(self, provider, capability_map: CapabilityMap,
                 interventions: tuple[Intervention, ...], handlers: dict[str, Handler],
                 policy: Policy = Policy()):
        if len({i.id for i in interventions}) != len(interventions):
            raise ValueError("Intervention IDs must be unique")
        if set(handlers) - {i.id for i in interventions}:
            raise ValueError("Handler has no registered intervention")
        if sum(i.kind == Kind.DIRECT for i in interventions) > 1:
            raise ValueError("Register at most one direct baseline intervention")
        self.layer = ForesightLayer(provider, capability_map)
        self.map, self.interventions, self.handlers, self.policy = capability_map, interventions, dict(handlers), policy

    def run(self, task: Task, *, initial_state: State = State(), run_id: str | None = None):
        run = run_id or str(uuid4())
        events = []
        p = self.policy
        state, total_cost, total_latency = initial_state, 0.0, 0.0
        attempts = Counter()
        stall, failures, last_confidence = 0, 0, None
        triggers = ["initial_assessment"]
        checker = EvidenceController(task.requirements)

        def progress(s):
            return sum(supported(r, s.evidence) for r in task.requirements) / len(task.requirements)

        def emit(event):
            self.map.event(run, len(events), event)
            events.append(event)

        emit({"event": "start", "run": run, "task": asdict(task),
              "policy": {**asdict(p), "approved": sorted(p.approved)}, "initial_state": asdict(state),
              "interventions": [asdict(i) for i in self.interventions]})
        reason = "step_budget"
        completed = checker.complete(state)
        for step in range(1, p.max_steps + 1):
            if completed:
                reason = "complete"
                break
            eligible, excluded = [], []
            adaptation_ready = self.map.adaptation_ready(task)
            for i in self.interventions:
                why = None
                if i.kind == Kind.ABSTAIN:
                    continue  # Always available as the policy's explicit outside option.
                if i.id not in self.handlers:
                    why = "no_handler"
                elif i.requires_approval and i.id not in p.approved:
                    why = "approval_required"
                elif i.kind in ADAPTATIONS and not adaptation_ready:
                    why = "gap_not_persistent"
                elif i.risk > p.max_risk:
                    why = "risk_limit"
                elif i.cost > p.max_cost - total_cost or i.latency > p.max_latency - total_latency:
                    why = "budget"
                elif attempts[i.id] >= p.max_attempts_per_intervention:
                    why = "attempt_limit"
                if why:
                    excluded.append({"intervention": i.id, "reason": why})
                else:
                    eligible.append(i)
            emit({"event": "candidates", "step": step, "excluded": excluded,
                  "eligible": [i.id for i in eligible]})
            if not eligible:
                reason = "no_eligible_intervention"
                break
            started = time.monotonic()
            try:
                assessment = self.layer.provider.assess(task, state, tuple(events))
                emit({"event": "assessment", "step": step, "triggers": triggers,
                      "assessment": asdict(assessment)})
                ranked = self.layer.rank(task, state, assessment, tuple(eligible), tuple(events), p)
            except (ValueError, TypeError, KeyError) as error:
                emit({"event": "prediction_error", "step": step, "message": str(error)})
                reason = "prediction_error"
                break
            prediction_latency = time.monotonic() - started
            total_latency += prediction_latency
            emit({"event": "predictions", "step": step, "candidates": [asdict(r) for r in ranked],
                  "prediction_latency": prediction_latency})
            # Predictions with unsupported beliefs are not enough to authorize an intervention.
            allowed = [r for r in ranked if r.supported_forecast
                       and r.intervention.latency <= p.max_latency - total_latency]
            if not allowed:
                reason = "unsupported_forecasts" if not any(r.supported_forecast for r in ranked) else "latency_budget"
                break
            chosen = allowed[0]
            if chosen.utility < p.abstain_utility:
                emit({"event": "abstain", "step": step, "best_utility": chosen.utility})
                reason = "abstained"
                break
            i = chosen.intervention
            # A committed event exists before every handler side effect.
            emit({"event": "selected", "step": step, "intervention": i.id,
                  "gain": chosen.gain, "utility": chosen.utility})
            if i.kind in ADAPTATIONS:
                # Training completion is not task completion. A separate gate must
                # evaluate an artifact before an application registers it for use.
                request = {"event": "adaptation_requested", "intervention": i.id,
                           "kind": i.kind.value, "task": task.id, "model": task.model,
                           "family": task.family, "clusters": self.map.clusters(model=task.model, family=task.family)}
                emit(request)
                reason = "adaptation_pending"
                break
            actual = self.handlers[i.id](task, state, i)
            for e in actual.evidence:
                if (e.source, e.authoritative, e.current) != (i.source, i.authoritative, i.current):
                    raise ValueError("Observed evidence provenance must match the intervention registry")
            before = progress(state)
            transition = observe(state, chosen.estimate.forecast, actual.evidence,
                                 action_succeeded=actual.tool_succeeded)
            state = transition.state
            after = progress(state)
            completed = checker.complete(state)
            total_cost += actual.cost
            total_latency += actual.latency
            attempts[i.id] += 1
            stall = stall + 1 if after <= before else 0
            failures = failures + 1 if not actual.tool_succeeded else 0
            triggers = []
            if abs(chosen.estimate.expected_progress - after) > p.mismatch_threshold:
                triggers.append("progress_mismatch")
            if transition.missing_predictions or transition.unexpected_observations:
                triggers.append("outcome_mismatch")
            if stall >= p.stall_steps:
                triggers.append("progress_stalled")
            if failures >= p.failure_steps:
                triggers.append("repeated_tool_failure")
            if actual.confidence is not None:
                if (chosen.estimate.expected_confidence - actual.confidence > p.confidence_drop
                        or (last_confidence is not None and last_confidence - actual.confidence > p.confidence_drop)):
                    triggers.append("confidence_drop")
                last_confidence = actual.confidence
            if not triggers:
                triggers = ["continue_assessment"]
            gap = FAILURE_GAPS.get(actual.failure_code, Gap.UNKNOWN)
            # Diagnosis hypotheses are not used as ground-truth failure labels.
            self.map.record(run=run, step=step, task=task, intervention=i.id, kind=i.kind.value,
                success=completed, tool_success=actual.tool_succeeded, probability=chosen.adjusted_success,
                raw_probability=chosen.estimate.task_success, progress=after, cost=actual.cost,
                latency=actual.latency, gap=gap, failure_code=actual.failure_code if not completed else None)
            emit({"event": "observed", "step": step, "intervention": i.id, "actual": asdict(actual),
                  "progress": after, "progress_error": chosen.estimate.expected_progress - after,
                  "tool_success_brier": transition.brier_score,
                  "task_success_brier": (chosen.adjusted_success - int(completed)) ** 2,
                  "missing_predictions": transition.missing_predictions,
                  "unexpected_observations": transition.unexpected_observations,
                  "completed": completed, "triggers": triggers, "observed_gap": gap.value})
            if completed:
                reason = "complete"
                break
            if total_cost >= p.max_cost or total_latency >= p.max_latency:
                reason = "cost_budget" if total_cost >= p.max_cost else "latency_budget"
                break
        result = {"event": "finished", "run": run, "completed": completed,
                  "reason": "complete" if completed else reason, "steps": sum(attempts.values()),
                  "cost": total_cost, "latency": total_latency,
                  "budget_overrun": total_cost > p.max_cost or total_latency > p.max_latency,
                  "state": asdict(state),
                  "clusters": self.map.clusters(model=task.model, family=task.family)}
        emit(result)
        return {**result, "events": events}
