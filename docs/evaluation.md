# Evaluation and claim boundaries

## Reproduce the mechanism demonstration

```bash
capability-gap demo --output runs/evaluation-01
capability-gap inspect runs/evaluation-01/capabilities.sqlite3
```

The demo uses deterministic priors and a deliberately limited arithmetic actor.
It is useful for verifying flow and invariants, not estimating an LLM's capability.
There are seven runs: three ordinary tasks, three persistent-gap tasks, and one
post-adaptation task. Four runs complete; three deliberately expose the skill gap.

The adapter demonstration fits `y = scale*x + offset` from three labeled conversions,
then tests three unseen conversions and two preexisting addition capabilities.
The baseline passes 0/3 targeted cases; the fitted skill passes 3/3 and retains both
regression passes. The accepted artifact is used on a fourth unseen conversion.
These small constructed results demonstrate the gate, not generalization to new
task families or model improvement.

## What tests enforce

- Predicted evidence cannot complete a task.
- Unknown handlers, missing approvals, unsupported beliefs and invalid probabilities
  do not authorize execution.
- A failure can trigger another strategy; retries and budgets bound the run.
- Predictions are persisted before handlers are called.
- Tool success and task success are measured as separate events.
- Retrying one task cannot create three independent persistent-gap examples.
- Solved tasks and infrastructure failures do not justify training under the default policy.
- Calibration is scoped to model version, family and selected intervention.
- Train/evaluation leakage, no improvement and regressions block adaptation acceptance.

## Experiments required before stronger claims

1. Use a fixed acting model with externally verifiable tasks spanning knowledge,
   reasoning, tools and domain-specific skills. Keep a held-out task set.
2. Record predictions in shadow mode before allowing them to affect routing. Measure
   task-success Brier scores and progress forecast error against independent outcomes.
3. Compare direct solving, fixed tool-first routing, ordinary evidence validation,
   uncertainty-only routing, outcome-only forecasting, and full gap-aware control.
4. Equalize total inference, retrieval, tool and training budgets. Include latency,
   token spend, human effort and intervention failures. Current demo cost units are
   declared examples, not dollar estimates or total compute accounting.
5. Randomize intervention assignment in a suitable subset to estimate causal gains;
   selected-action history alone is biased and cannot identify counterfactual success.
6. Evaluate adaptation on fresh targeted tasks and broad regression tasks, repeat
   across seeds/models, and report paired uncertainty intervals.

The idea is weakened if a simple deterministic router performs equally well, if
gap labels do not predict useful interventions, if prediction overhead exceeds the
benefit, or if adaptation gains disappear on unseen tasks. The present implementation
does not establish novelty, general capability diagnosis, or improved LLM performance.
