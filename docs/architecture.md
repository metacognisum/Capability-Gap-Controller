# Architecture and semantics

## Layer responsibilities

`Task` names a task family and model version and declares verification requirements.
`Assessment` estimates direct-solve success and possible gaps. `Intervention` is an
application-owned description with cost, latency, risk, source metadata and approval
requirements. Candidate generation means selecting from this registry, not executing
arbitrary model-suggested APIs.

`PredictionProvider` supplies assessment and intervention estimates. The built-in
JSON provider accepts a completion callable. `ForesightLayer` reuses the installed
Foresight package's strict forecast parser, metadata validation and support checks.
It attaches capability-map adjustments and ranks candidates. The outer controller
owns intervention policy, execution, re-diagnosis and persistence.

The model does not set costs, authority or freshness. Unsupported forecast beliefs
and unsupported next decisions make a candidate ineligible. A plausible forecast
still cannot satisfy completion: only handler observations enter the evidence state.

## Distinct prediction targets

| Field | Meaning | Observed target |
| --- | --- | --- |
| `Assessment.success_now` | Direct solve would complete from the current state | Counterfactual unless direct is executed |
| `Estimate.task_success` | Task requirements pass immediately after this intervention | Observed verifier result after the selected intervention |
| `Forecast.success_probability` | The tool/handler executes successfully | Executor's `tool_succeeded` |
| `expected_progress` | Fraction of requirements supported after the next step | Fraction computed from actual evidence |
| `expected_confidence` | Actor's expected self-reported confidence | Optional actor confidence, not truth |

Success after a whole retry sequence is not the same as success after one
intervention. A useful information-gathering step may not immediately finish a
task. The current utility does not model long-horizon option value; that is a
research extension. Task success estimates cannot exceed tool success estimates.

## Utility and abstention

The policy ranks `adjusted_probability - success_now - weighted_costs`. Source
support is a hard candidate filter. Risk is an application-supplied normalized
score, not an inferred safety guarantee. Cost/latency weights have units and must
be chosen for the application; their defaults are explicit heuristics.

When direct is a candidate, its version-scoped empirical adjustment also adjusts
the baseline probability. Otherwise the provider's current direct-solve estimate
is used. `abstain_utility` is a reservation utility in the same scale; abstention
is never scored as task success.

Budget filtering uses declared per-intervention estimates. Actual executor charges
are recorded, and a subsequent step stops if a budget is exhausted. An action can
overrun its estimate: the terminal report exposes `budget_overrun`. Applications
must enforce hard spending/time limits inside handlers. Predictor elapsed time is
included; predictor token costs are not automatically charged. No claim of a hard
wall-clock deadline is made.

## Reassessment signals

The provider sees the entire observed history on each cycle. Trigger labels
distinguish initial assessment, progress mismatch, missing/unexpected outcomes,
stalls, repeated tool failure and unexpected confidence drops. A mismatch is a
signal to investigate, not a confirmed capability diagnosis.

Only application-issued failure codes map to observed gap categories. Unknown codes
remain `unknown`; the model's hypothesis does not become a training label. The
registry currently recognizes `missing_knowledge`, `reasoning_error`, `tool_error`,
`context_overflow`, `domain_error`, `compute_exhausted`, and `skill_error`.

## Capability map

SQLite stores attempts, run events and immutable adaptation reports. A selected
forecast is committed before invoking its handler. Each `(run, step)` outcome is
unique. A trace with no `finished` event is incomplete, not a verified successful
run. A crash between handler execution and persistence requires manual reconciliation;
the runtime does not claim exactly-once execution or safe automatic resumption.

Prediction adjustment uses:

```text
adjusted = (4 × current_forecast + sum(per_task_mean_success)) / (4 + distinct_tasks)
```

Only prior outcomes with the same model, family and intervention are included.
The current task ID is excluded. Repeated attempts on one task count as one task
for this adjustment, although attempt-level Brier scores remain visible. These
selected-action statistics are biased by the routing policy and task mix. They
are not calibrated posterior guarantees or randomized intervention effects.

Clusters use exact `(model, family, gap, failure_code)` keys. This version does not
perform embedding clustering. Adaptation eligibility requires at least three
unresolved distinct tasks, **each** failing with two distinct intervention kinds,
and a reasoning, domain or persistent-skill gap. A later verified solution removes
that task from unresolved support. Tool/infrastructure and unknown failures cannot
justify neural training under this default policy.

## Adaptation boundary

Training kinds require explicit approval and persistent-gap eligibility. If selected,
the controller returns `adaptation_pending`; it does not call a trainer as an ordinary
task-solving handler. `collect_failures` exports examples for labeling only after
approval and eligibility. Missing correct answers stay null.

`AdaptationGate` gives a trainer only training cases, computes the baseline first,
then checks candidate outputs on separate held-out and regression cases. Input
fingerprints and IDs must be disjoint. Exact input matching catches simple leakage,
not semantic paraphrases or a malicious trainer with access to external files.

Acceptance requires the configured target gain and zero lost baseline passes on
the regression suite. The default minimum of three held-out cases supports a demo,
not a research conclusion. Reports retain per-case pass/fail results, artifact hash
and a new version ID. Old-model statistics are preserved. Applying that version to
new tasks produces new capability-map entries and subsequent forecast adjustments.
