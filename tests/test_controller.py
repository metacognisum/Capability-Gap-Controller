from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from foresight import Belief, Evidence, Forecast, Requirement, State

from capability_gap import (Assessment, CapabilityController, CapabilityMap, Estimate,
                            Execution, Gap, Intervention, Kind, Policy, Task)
from capability_gap.demo import DemoPredictor, make_handlers, registry, run_demo
from capability_gap.prediction import ForesightLayer, JsonPredictionProvider


class Base(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.map = CapabilityMap(self.root / "map.sqlite3")
        self.task = Task("task1", "arithmetic", "model-v1", "7 * 8", (Requirement("solved", "yes"),))
        self.direct = Intervention("direct", Kind.DIRECT, "Solve", .1, .1, .01, "direct")
        self.tool = Intervention("tool", Kind.TOOL, "Calculate", .4, .1, .01, "tool")

    def fail(self, task, state, i):
        return Execution((), True, i.cost, i.latency, .1, "reasoning_error")

    def succeed(self, task, state, i):
        return Execution((Evidence("actual", "solved", "yes", i.source, True, True),), True, i.cost, i.latency, .9)

    def controller(self, predictor=None, interventions=None, handlers=None, policy=None):
        return CapabilityController(predictor or DemoPredictor(), self.map,
            interventions if interventions is not None else (self.direct, self.tool),
            handlers if handlers is not None else {"direct": self.fail, "tool": self.succeed}, policy or Policy())


class ControllerTests(Base):
    def test_hidden_gap_switches_intervention_and_verifies(self):
        result = self.controller().run(self.task)
        self.assertTrue(result["completed"])
        selected = [e["intervention"] for e in result["events"] if e["event"] == "selected"]
        self.assertEqual(selected, ["direct", "tool"])
        assessments = [e for e in result["events"] if e["event"] == "assessment"]
        self.assertIn("progress_mismatch", assessments[1]["triggers"])
        self.assertIn("confidence_drop", assessments[1]["triggers"])
        self.assertEqual(result["state"]["evidence"][0]["id"], "actual")

    def test_forecast_cannot_complete_without_actual_evidence(self):
        result = self.controller(interventions=(self.direct,), handlers={"direct": self.fail},
                                 policy=Policy(max_steps=1)).run(self.task)
        self.assertFalse(result["completed"])
        self.assertEqual(result["state"]["evidence"], ())

    def test_unsupported_forecast_never_executes(self):
        class Unsupported(DemoPredictor):
            def predict(inner, *args):
                return tuple(replace(e, forecast=replace(e.forecast, evidence=())) for e in super().predict(*args))
        result = self.controller(predictor=Unsupported()).run(self.task)
        self.assertEqual(result["reason"], "unsupported_forecasts")
        self.assertEqual(result["steps"], 0)

    def test_unavailable_and_unapproved_handlers_are_excluded(self):
        human = Intervention("human", Kind.HUMAN, "Ask", 1, 1, 0, "human", requires_approval=True)
        result = self.controller(interventions=(human, self.tool), handlers={"human": self.succeed}).run(self.task)
        excluded = result["events"][1]["excluded"]
        self.assertIn({"intervention": "human", "reason": "approval_required"}, excluded)
        self.assertIn({"intervention": "tool", "reason": "no_handler"}, excluded)
        self.assertEqual(result["steps"], 0)

    def test_risk_and_budget_limits(self):
        result = self.controller(policy=Policy(max_risk=0)).run(self.task)
        self.assertEqual(result["reason"], "no_eligible_intervention")
        result = self.controller(policy=Policy(max_cost=.05)).run(self.task)
        self.assertEqual(result["steps"], 0)

    def test_abstention_is_not_success(self):
        result = self.controller(policy=Policy(abstain_utility=1)).run(self.task)
        self.assertEqual(result["reason"], "abstained")
        self.assertFalse(result["completed"])
        self.assertEqual(result["steps"], 0)

    def test_retries_bounded_and_stalls_reported(self):
        result = self.controller(interventions=(self.direct,), handlers={"direct": self.fail},
                                 policy=Policy(max_steps=5, abstain_utility=-1)).run(self.task)
        self.assertEqual(result["steps"], 2)
        observed = [e for e in result["events"] if e["event"] == "observed"]
        self.assertIn("progress_stalled", observed[-1]["triggers"])

    def test_repeated_tool_failures_distinct_from_task_failure(self):
        def failure(task, state, i):
            return Execution((), False, i.cost, i.latency, .1, "tool_error")
        result = self.controller(interventions=(self.direct,), handlers={"direct": failure},
                                 policy=Policy(abstain_utility=-1)).run(self.task)
        observed = [e for e in result["events"] if e["event"] == "observed"]
        self.assertIn("repeated_tool_failure", observed[-1]["triggers"])
        self.assertGreater(observed[0]["tool_success_brier"], .9)

    def test_forecast_written_before_handler(self):
        def inspect(task, state, i):
            events = self.map.events("fixed-run")
            self.assertEqual(events[-1]["event"], "selected")
            self.assertEqual(events[-2]["event"], "predictions")
            return self.succeed(task, state, i)
        result = self.controller(handlers={"direct": inspect, "tool": inspect}).run(self.task, run_id="fixed-run")
        self.assertTrue(result["completed"])
        self.assertEqual(self.map.events("fixed-run")[-1]["event"], "finished")
        with self.assertRaises(sqlite3.IntegrityError):
            self.controller().run(self.task, run_id="fixed-run")

    def test_provenance_forgery_stops_run(self):
        def forged(task, state, i):
            return Execution((Evidence("forged", "solved", "yes", "invented", True, True),), True, .1, .1)
        with self.assertRaises(ValueError):
            self.controller(handlers={"direct": forged, "tool": forged}).run(self.task)

    def test_invalid_prediction_executes_nothing(self):
        result = self.controller(predictor=JsonPredictionProvider(lambda _: "not json")).run(self.task)
        self.assertEqual(result["reason"], "prediction_error")
        self.assertEqual(result["steps"], 0)

    def test_already_complete_does_not_call_handler(self):
        state = State((Evidence("initial", "solved", "yes", "verified", True, True),))
        result = self.controller().run(self.task, initial_state=state)
        self.assertTrue(result["completed"])
        self.assertEqual(result["steps"], 0)

    def test_training_is_not_eligible_from_approval_alone(self):
        train = Intervention("train", Kind.LORA, "Train", 1, 1, 0, "train", requires_approval=True)
        result = self.controller(interventions=(train,), handlers={"train": self.succeed},
                                 policy=Policy(approved=frozenset({"train"}))).run(self.task)
        self.assertEqual(result["steps"], 0)
        self.assertEqual(result["events"][1]["excluded"][0]["reason"], "gap_not_persistent")

    def test_actual_budget_overrun_stops_further_work(self):
        def expensive(task, state, i):
            return Execution((), True, 20, 1, .1)
        result = self.controller(handlers={"direct": expensive, "tool": expensive}).run(self.task)
        self.assertEqual(result["steps"], 1)
        self.assertTrue(result["budget_overrun"])
        self.assertEqual(result["reason"], "cost_budget")


class ModelContractTests(Base):
    def test_model_output_reuses_foresight_and_registry(self):
        assessment = {"success_now": .3, "gaps": ["reasoning"], "failure_points": ["arithmetic"]}
        estimate = {"intervention": "direct", "task_success": .5, "expected_progress": 1,
            "expected_confidence": .8, "forecast": {"action": "direct", "evidence": [
            {"id": "p", "claim": "solved", "value": "yes"}], "beliefs": [
            {"claim": "solved", "value": "yes", "evidence_ids": ["p"]}],
            "next_decision": "complete", "success_probability": .9}}
        responses = iter((json.dumps(assessment), json.dumps({"predictions": [estimate]})))
        provider = JsonPredictionProvider(lambda _: next(responses))
        a = provider.assess(self.task, State(), ())
        ranked = ForesightLayer(provider, self.map).rank(self.task, State(), a, (self.direct,), (), Policy())
        self.assertEqual(ranked[0].estimate.forecast.cost, .1)
        self.assertEqual(ranked[0].estimate.forecast.evidence[0].source, "direct")
        self.assertTrue(ranked[0].supported_forecast)
        self.assertAlmostEqual(ranked[0].gain, .2)

    def test_probabilities_have_different_events(self):
        class Bad(DemoPredictor):
            def predict(inner, *args):
                return tuple(replace(e, task_success=1) for e in super().predict(*args))
        result = self.controller(predictor=Bad()).run(self.task)
        self.assertEqual(result["reason"], "prediction_error")

    def test_types_reject_nan_and_bool_numbers(self):
        for value in (float("nan"), float("inf"), True, -1, 1.1):
            with self.assertRaises(ValueError):
                Assessment(value, (Gap.UNKNOWN,), ())
        with self.assertRaises(ValueError):
            Intervention("train", Kind.LORA, "Train", 1, 1, 0, "source")
        with self.assertRaises(ValueError):
            Execution((), "yes", 1, 1)
        with self.assertRaises(ValueError):
            Policy(max_steps=True)


class DemoTests(Base):
    def test_end_to_end_recovery_and_adaptation(self):
        result = run_demo(self.root / "demo")
        self.assertEqual([r["completed"] for r in result["runs"]], [True, True, True, False, False, False, True])
        self.assertEqual(result["runs"][1]["steps"], 2)
        self.assertTrue(result["adaptation"]["accepted"])
        self.assertEqual(result["adaptation"]["candidate_target_passes"], 3)
        self.assertEqual(result["adaptation"]["regressions"], 0)
        self.assertEqual(result["failure_collection"]["records"], 6)
        with (self.root / "demo" / "failures-for-labeling.jsonl").open() as stream:
            self.assertTrue(all(json.loads(line)["verified_target"] is None for line in stream))
        with self.assertRaises(FileExistsError):
            run_demo(self.root / "demo")
