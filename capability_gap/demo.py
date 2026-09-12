"""Offline executable examples. The actor and predictor are explicitly scripted."""
from decimal import Decimal
import json
from pathlib import Path
import re

from foresight import Belief, Evidence, Forecast, Requirement

from .adaptation import AdaptationGate, Candidate, Case
from .controller import CapabilityController
from .storage import CapabilityMap
from .types import Assessment, Estimate, Execution, Gap, Intervention, Kind, Policy, Task


def arithmetic(prompt: str, *, multiply=False) -> str:
    match = re.fullmatch(r"(-?\d+)\s*([+*])\s*(-?\d+)", prompt.strip())
    if not match or (match[2] == "*" and not multiply):
        return "unsupported"
    a, b = int(match[1]), int(match[3])
    return str(a + b if match[2] == "+" else a * b)


def registry():
    # Metadata is application-owned. No side effect occurs merely by registering a kind.
    kinds = [(Kind.DIRECT, .1, .1), (Kind.RETRIEVAL, .3, .2), (Kind.TOOL, .4, .2),
             (Kind.COMPUTE, .5, .3), (Kind.SPECIALIST, 1, .5), (Kind.EXAMPLES, .3, .2),
             (Kind.HUMAN, 5, 10), (Kind.SKILL, 2, 2), (Kind.ADAPTER, 4, 5),
             (Kind.LORA, 6, 10), (Kind.FINETUNE, 8, 20)]
    return tuple(Intervention(k.value, k, f"Application-registered {k.value} strategy", cost, latency,
                             .05, k.value, requires_approval=k in {Kind.HUMAN, Kind.SKILL,
                             Kind.ADAPTER, Kind.LORA, Kind.FINETUNE}) for k, cost, latency in kinds)


class DemoPredictor:
    """Fixture priors know task families but cannot access executor answer keys."""
    def assess(self, task, state, history):
        failed = [e for e in history if e["event"] == "observed" and not e["completed"]]
        gap = {"lookup": Gap.KNOWLEDGE, "arithmetic": Gap.REASONING,
               "conversion": Gap.PERSISTENT_SKILL}.get(task.family, Gap.UNKNOWN)
        if failed:
            gap = Gap(failed[-1]["observed_gap"])
        return Assessment(.1 if failed else (.9 if task.family == "arithmetic" else .25),
                          (gap,), ("Direct solver may lack the required operation or source",))

    def predict(self, task, state, assessment, interventions, history):
        observed = [e for e in history if e["event"] == "observed"]
        failed = {e["intervention"] for e in observed if not e["completed"]}
        preferred = {"lookup": Kind.RETRIEVAL, "conversion": Kind.COMPUTE,
                     "arithmetic": Kind.TOOL if observed else Kind.DIRECT}.get(task.family, Kind.DIRECT)
        estimates = []
        for i in interventions:
            p = .05 if i.id in failed else (.9 if i.kind == preferred else .35)
            predicted = Evidence(f"predicted:{len(observed)}:{i.id}", "solved", "yes", i.source,
                                 i.authoritative, i.current)
            forecast = Forecast(i.id, (predicted,), (Belief("solved", "yes", (predicted.id,)),),
                                "complete", .99, i.cost)
            estimates.append(Estimate(i.id, p, .95, .9, forecast))
        return tuple(estimates)


def verified_execution(task, state, intervention, answer, expected, *, failure_code):
    # Independent environment check: incorrect actor output never yields solved=yes.
    evidence = ()
    if answer == expected:
        evidence = (Evidence(f"verified:{task.id}:{intervention.id}:{len(state.evidence)}", "solved", "yes",
                             intervention.source, intervention.authoritative, intervention.current),)
    return Execution(evidence, True, intervention.cost, intervention.latency,
                     confidence=.9 if evidence else .2, failure_code=None if evidence else failure_code)


def make_handlers(expected: str, corpus: Path):
    def direct(task, state, i):
        return verified_execution(task, state, i, arithmetic(task.prompt), expected,
                                  failure_code="skill_error" if task.family == "conversion" else "reasoning_error")
    def tool(task, state, i):
        return verified_execution(task, state, i, arithmetic(task.prompt, multiply=True), expected,
                                  failure_code="skill_error" if task.family == "conversion" else "tool_error")
    def retrieval(task, state, i):
        try:
            answer = json.loads(corpus.read_text(encoding="utf-8")).get(task.prompt, "unknown")
        except (OSError, ValueError):
            return Execution((), False, i.cost, i.latency, .1, "tool_error")
        return verified_execution(task, state, i, answer, expected, failure_code="missing_knowledge")
    return {"direct": direct, "tool": tool, "compute": direct, "retrieval": retrieval}


def learned_affine(artifact, prompt):
    match = re.fullmatch(r"convert (-?\d+)", prompt)
    if not match:
        return arithmetic(prompt)
    result = Decimal(artifact["scale"]) * Decimal(match[1]) + Decimal(artifact["offset"])
    return str(int(result)) if result == int(result) else str(result.normalize())


def train_affine(cases):
    """Fit two numeric parameters from examples; this is skill fitting, NOT LoRA."""
    pairs = [(Decimal(c.input.removeprefix("convert ")), Decimal(c.expected)) for c in cases]
    x1, y1 = pairs[0]
    x2, y2 = next((x, y) for x, y in pairs if x != x1)
    scale = (y2 - y1) / (x2 - x1)
    offset = y1 - scale * x1
    if any(scale * x + offset != y for x, y in pairs):
        raise ValueError("Training examples do not define an affine conversion")
    artifact = {"type": "affine_skill", "scale": str(scale), "offset": str(offset)}
    return Candidate(artifact, lambda prompt: learned_affine(artifact, prompt))


def run_demo(directory: Path):
    directory.mkdir(parents=True, exist_ok=False)
    capability_map = CapabilityMap(directory / "capabilities.sqlite3")
    corpus = directory / "reference.json"
    corpus.write_text(json.dumps({"Where is the example lab?": "Pune"}), encoding="utf-8")
    interventions = registry()
    runs = []
    # A hidden arithmetic limitation is exposed by real execution, then a tool resolves it.
    for ident, family, prompt, expected in (("sum", "arithmetic", "2 + 3", "5"),
            ("hidden-multiply", "arithmetic", "7 * 8", "56"),
            ("lookup", "lookup", "Where is the example lab?", "Pune")):
        task = Task(ident, family, "scripted-actor-v1", prompt, (Requirement("solved", "yes"),))
        controller = CapabilityController(DemoPredictor(), capability_map, interventions,
                                          make_handlers(expected, corpus))
        runs.append(controller.run(task))
    # Separate tasks and two unsuccessful remedies are required to propose adaptation.
    training = tuple(Case(f"train-{n}", f"convert {n}", str(2 * n + 3)) for n in (1, 2, 3))
    limited = tuple(i for i in interventions if i.kind in {Kind.DIRECT, Kind.COMPUTE})
    for case in training:
        task = Task(case.id, "conversion", "scripted-actor-v1", case.input, (Requirement("solved", "yes"),))
        runs.append(CapabilityController(DemoPredictor(), capability_map, limited,
            {k: v for k, v in make_handlers(case.expected, corpus).items() if k in {"direct", "compute"}},
            Policy(max_steps=2, max_attempts_per_intervention=1, abstain_utility=-1)).run(task))
    gate = AdaptationGate(capability_map)
    collection = capability_map.collect_failures(task.model, task.family, Gap.PERSISTENT_SKILL,
                                                 directory / "failures-for-labeling.jsonl", approved=True)
    adaptation = gate.evaluate("affine-skill-v1", task, Gap.PERSISTENT_SKILL, training=training,
        heldout=tuple(Case(f"heldout-{n}", f"convert {n}", str(2 * n + 3)) for n in (4, 5, 6)),
        regression=(Case("reg-sum-1", "2 + 4", "6"), Case("reg-sum-2", "5 + 6", "11")),
        baseline=arithmetic, trainer=train_affine, approved=True)
    (directory / "adaptation.json").write_text(json.dumps(adaptation, indent=2) + "\n", encoding="utf-8")
    if adaptation["accepted"]:
        artifact = adaptation["artifact"]
        (directory / "accepted-skill.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
        new_task = Task("post-adaptation", "conversion", adaptation["new_model"], "convert 8",
                        (Requirement("solved", "yes"),))
        i = next(i for i in interventions if i.kind == Kind.DIRECT)
        def adapted(task, state, intervention):
            return verified_execution(task, state, intervention, learned_affine(artifact, task.prompt), "19",
                                      failure_code="skill_error")
        runs.append(CapabilityController(DemoPredictor(), capability_map, (i,), {i.id: adapted},
                                        Policy(abstain_utility=-1)).run(new_task))
    for result in runs:
        with (directory / f"{result['run']}.jsonl").open("w", encoding="utf-8") as stream:
            for event in result["events"]:
                stream.write(json.dumps(event) + "\n")
    summary = {"label": "Deterministic mechanism demonstration; not measured LLM performance",
        "runs": [{k: r[k] for k in ("run", "completed", "reason", "steps", "cost")} for r in runs],
        "capability_map": capability_map.summary(), "clusters": capability_map.clusters(),
        "adaptation": adaptation, "failure_collection": collection,
        "limitations": "Scripted priors; exact-match verifier; affine skill fitting, not neural training. "
                        "Empirical prediction updates are selected-action statistics, not causal effects."}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
