"""Opt-in local LLM actor plus Foresight predictor, with independent arithmetic checks.

Run after installation: python examples/live_arithmetic.py --model YOUR_MODEL --output runs/live-01
The predictor never receives the answer key. This is still a tiny smoke example.
"""
import argparse
import json
from pathlib import Path
import time

from foresight import Requirement
from foresight.models import OllamaCompletion

from capability_gap import (CapabilityController, CapabilityMap, Execution, Intervention,
                            JsonPredictionProvider, Kind, Task)
from capability_gap.demo import arithmetic, verified_execution


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expression", default="137 * 29")
    args = parser.parse_args()
    expected = arithmetic(args.expression, multiply=True)
    if expected == "unsupported":
        parser.error("Use two integer operands with + or *")
    args.output.mkdir(parents=True, exist_ok=False)
    completion = OllamaCompletion(args.model)
    capability_map = CapabilityMap(args.output / "map.sqlite3")
    direct = Intervention("direct", Kind.DIRECT, "Ask the current model to solve arithmetic", 1, 1, 0, "actor")
    tool = Intervention("calculator", Kind.TOOL, "Run an exact integer calculator supporting + and *", .1, .01, 0, "calculator")
    task = Task("live-arithmetic", "integer-arithmetic", args.model, args.expression, (Requirement("solved", "yes"),))

    def actor(task, state, intervention):
        started = time.monotonic()
        try:
            data = json.loads(completion('Solve the expression. Return JSON {"answer":"integer"}. Expression: ' + task.prompt))
            if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
                raise ValueError("Invalid answer")
            result = verified_execution(task, state, intervention, data["answer"], expected, failure_code="reasoning_error")
            return Execution(result.evidence, True, intervention.cost, time.monotonic() - started,
                             failure_code=result.failure_code)
        except (ValueError, TypeError):
            return Execution((), False, intervention.cost, time.monotonic() - started, failure_code="tool_error")

    def calculator(task, state, intervention):
        return verified_execution(task, state, intervention, arithmetic(task.prompt, multiply=True), expected,
                                  failure_code="tool_error")

    controller = CapabilityController(JsonPredictionProvider(completion), capability_map,
        (direct, tool), {"direct": actor, "calculator": calculator})
    result = controller.run(task)
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("completed", "reason", "steps", "cost")}, indent=2))
    if not result["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
