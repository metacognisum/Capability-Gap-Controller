"""Held-out adaptation gate with independently checked outputs and immutable reports."""
from dataclasses import dataclass
import hashlib
import json
from typing import Callable

from .storage import CapabilityMap
from .types import Gap, Task, number, text


@dataclass(frozen=True)
class Case:
    id: str
    input: str
    expected: str

    def __post_init__(self):
        for key in ("id", "input", "expected"):
            text(getattr(self, key), key)


@dataclass(frozen=True)
class Candidate:
    artifact: dict
    predict: Callable[[str], str]


class AdaptationGate:
    """Trainer receives training cases only. Evaluation truth stays in this gate.

    Exact string checking is the bundled verifier. Domain integrations should
    normalize outputs or implement an externally checked task suite.
    """
    def __init__(self, capability_map: CapabilityMap, *, min_gain=.2, min_test_cases=3):
        number(min_gain, "min_gain", 0, 1)
        if type(min_test_cases) is not int or min_test_cases < 1:
            raise ValueError("min_test_cases must be positive")
        self.map, self.min_gain, self.min_test_cases = capability_map, min_gain, min_test_cases

    def evaluate(self, ident: str, task: Task, gap: Gap, *, training: tuple[Case, ...],
                 heldout: tuple[Case, ...], regression: tuple[Case, ...], baseline,
                 trainer, approved: bool = False):
        text(ident, "adaptation ID")
        if approved is not True:
            raise ValueError("Adaptation requires explicit approval")
        matching = [c for c in self.map.clusters(model=task.model, family=task.family)
                    if c["gap"] == gap.value and c["adaptation_eligible"]]
        if not matching:
            raise ValueError("No eligible persistent gap for this model, family and category")
        if not training or len(heldout) < self.min_test_cases or not regression:
            raise ValueError("Provide training, sufficient held-out cases and a regression suite")
        all_cases = training + heldout + regression
        if len({c.id for c in all_cases}) != len(all_cases):
            raise ValueError("Case IDs must be disjoint across all splits")
        fingerprints = [hashlib.sha256(c.input.strip().encode()).hexdigest() for c in all_cases]
        if len(set(fingerprints)) != len(fingerprints):
            raise ValueError("Inputs must be disjoint across all splits")
        # Evaluate the original before training so training cannot change its measurements.
        def checks(predict, cases):
            results = []
            for case in cases:
                try:
                    result = predict(case.input)
                    results.append(type(result) is str and result == case.expected)
                except Exception:
                    results.append(False)
            return results
        baseline_target = checks(baseline, heldout)
        baseline_regression = checks(baseline, regression)
        candidate = trainer(training)
        # Artifact metadata must be serializable, not arbitrary executable code.
        serialized = json.dumps(candidate.artifact, sort_keys=True, allow_nan=False)
        after_target = checks(candidate.predict, heldout)
        after_regression = checks(candidate.predict, regression)
        gain = (sum(after_target) - sum(baseline_target)) / len(heldout)
        regressions = sum(before and not after for before, after in zip(baseline_regression, after_regression))
        accepted = gain >= self.min_gain and regressions == 0
        report = {"adaptation": ident, "parent_model": task.model, "family": task.family,
            "gap": gap.value, "accepted": accepted, "target_gain": gain,
            "baseline_target_passes": sum(baseline_target), "candidate_target_passes": sum(after_target),
            "heldout_count": len(heldout), "regression_count": len(regression), "regressions": regressions,
            "baseline_target": baseline_target, "candidate_target": after_target,
            "baseline_regression": baseline_regression, "candidate_regression": after_regression,
            "training_ids": [c.id for c in training], "heldout_ids": [c.id for c in heldout],
            "regression_ids": [c.id for c in regression], "input_fingerprints": fingerprints,
            "artifact": candidate.artifact, "artifact_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
            "new_model": task.model + "+" + ident if accepted else None,
            "note": "Small exact-match evaluation; acceptance is not statistical proof of general improvement."}
        self.map.save_adaptation(ident, task.model, task.family, gap,
                                 "accepted" if accepted else "rejected", report)
        return report
