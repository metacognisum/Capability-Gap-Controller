# Integrating an agent

Install both packages as described in the README. Import application contracts from
`capability_gap`, and evidence primitives from `foresight`.

## Model and intervention adapters

```python
from foresight import Evidence, Requirement
from foresight.models import OllamaCompletion
from capability_gap import (
    CapabilityController, CapabilityMap, Execution, Intervention,
    JsonPredictionProvider, Kind, Policy, Task,
)

task = Task("task-001", "example-family", "YOUR_MODEL_VERSION", "Your task",
            (Requirement("solved", "yes"),))
strategy = Intervention("direct", Kind.DIRECT, "Run the application agent",
                        cost=1, latency=1, risk=0, source="application-verifier")

# Replace this function with an application-owned actor AND independent validator.
def handler(task, state, intervention):
    # In a real integration: answer = actor(task.prompt); passed = verifier(answer).
    # An empty observation deliberately cannot mark this example solved.
    return Execution((), tool_succeeded=True, cost=1, latency=1,
                     confidence=None, failure_code="reasoning_error")

provider = JsonPredictionProvider(OllamaCompletion("YOUR_LOCAL_MODEL"))
controller = CapabilityController(provider, CapabilityMap("runs/map.sqlite3"),
                                  (strategy,), {strategy.id: handler}, Policy(max_steps=2))
result = controller.run(task)
print(result["completed"], result["reason"])
```

For an executable model example with a real independent arithmetic check, use
`examples/live_arithmetic.py`. A model must already be installed in a running
Ollama instance. No downloads or API requests happen during the offline demo.

Any completion callable accepting a prompt and returning JSON can replace the
transport. JSON syntax, candidate IDs, probability ranges, evidence references and
application-owned metadata are checked. Invalid predictions stop before execution.
Model requests can still be slow or fail; configure transport timeouts and monitor
incomplete runs. Provider or handler programming errors propagate rather than being
silently labeled model incapability.

Task success must come from an independent verifier. For an actual successful result,
return evidence such as `Evidence("unique-observation-id", "solved", "yes",
intervention.source, intervention.authoritative, intervention.current)`. A model
asserting that it succeeded is not independent verification. Evidence IDs must be
unique for new observations. Foresight rejects conflicting reuse of an ID.

## Registering interventions

Each intervention has a stable ID. Register handlers only for available capabilities:
retrieval systems, calculators, API clients, extended-compute calls, specialist
models, example selection or a human-approval workflow. Unregistered IDs cannot
execute even if the predictor names them. There is no dynamic Python or shell
execution of model-generated code.

Human-contact and other consequential handlers should use `requires_approval=True`.
The application supplies approved IDs through `Policy(approved=frozenset({...}))`.
This is a runtime policy boundary, not an interactive UI or security sandbox. The
controller does not itself contact people, purchase compute or start GPU jobs.

The built-in kinds `skill`, `adapter`, `lora` and `fine_tune` mean creating or changing
a capability. They return an adaptation request when selected; route it to the gate.
Using an already accepted skill belongs in a direct/tool/specialist handler registered
under the new model or agent version.

## Adaptation

`AdaptationGate.evaluate` accepts training, held-out and regression `Case` tuples,
an existing baseline predictor and a trainer. A trainer receives training cases
only and returns `Candidate(artifact_metadata, predict_callable)`.

An external LoRA integration can implement this interface by training an adapter,
returning its version/checksum metadata, and exposing inference through the callable.
This repository does not implement LoRA training. The gate will evaluate outputs;
it does not trust a trainer's claimed accuracy.

Use `CapabilityMap.collect_failures(model, family, gap, output, approved=True)` to
export eligible local failures for human or externally verified labeling. These
records are not ready-made supervised targets. Do not train on unverified answers.

Accepted reports include `new_model`. The application explicitly registers the
accepted artifact and uses that version for new tasks. Rejected artifacts must not
be promoted. The included demo illustrates fitting and applying a numeric skill
without neural model training.

## Operational limits

- SQLite is a local store, not a distributed task coordinator. Keep task IDs stable
  across retries and use a new run ID for each execution.
- No unexecuted candidate receives a success/failure label.
- Provider prompts and traces can contain task data. Control access to the database
  and exports; use an approved endpoint for any external completion provider.
- The source metadata and failure codes are only as trustworthy as application adapters.
- A handler may overrun its declared cost or duration. It must enforce external limits.
- There is no automatic resume after a crash, source retraction or semantic truth checker.
