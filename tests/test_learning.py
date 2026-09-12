from dataclasses import replace
import sqlite3

from capability_gap import Gap, Kind, Intervention, Policy
from capability_gap.adaptation import AdaptationGate, Candidate, Case
from capability_gap.demo import arithmetic, train_affine
from tests.test_controller import Base


class LearningBase(Base):
    def record(self, task, intervention="direct", kind="direct", success=False, gap=Gap.PERSISTENT_SKILL,
               failure_code="skill_error", step=1, run=None):
        self.map.record(run=run or task.id + intervention, step=step, task=task,
            intervention=intervention, kind=kind, success=success, tool_success=True,
            probability=.8, raw_probability=.8, progress=int(success), cost=.1, latency=.1,
            gap=gap, failure_code=None if success else failure_code)

    def persistent(self):
        for n in range(3):
            task = replace(self.task, id=f"failure-{n}")
            self.record(task)
            self.record(task, "compute", "compute")


class StorageTests(LearningBase):
    def test_data_collection_requires_persistence_and_approval(self):
        with self.assertRaises(ValueError):
            self.map.collect_failures(self.task.model, self.task.family, Gap.PERSISTENT_SKILL,
                                      self.root / "data.jsonl", approved=True)
        self.persistent()
        with self.assertRaises(ValueError):
            self.map.collect_failures(self.task.model, self.task.family, Gap.PERSISTENT_SKILL,
                                      self.root / "data.jsonl")

    def test_training_selection_requests_evaluation_instead_of_claiming_success(self):
        self.persistent()
        train = Intervention("train", Kind.LORA, "Train", 1, 1, 0, "trainer", requires_approval=True)
        def must_not_run(*args):
            self.fail("Training must go through the adaptation gate")
        result = self.controller(interventions=(train,), handlers={"train": must_not_run},
            policy=Policy(approved=frozenset({"train"}), abstain_utility=-1)).run(self.task)
        self.assertEqual(result["reason"], "adaptation_pending")
        self.assertFalse(result["completed"])
        self.assertEqual(result["steps"], 0)

    def test_selected_outcome_updates_predictions_only_in_scope(self):
        self.record(self.task)
        future = replace(self.task, id="future")
        self.assertAlmostEqual(self.map.calibrate(future, "direct", .8), .64)
        self.assertEqual(self.map.calibrate(future, "tool", .8), .8)
        self.assertEqual(self.map.calibrate(replace(future, model="model-v2"), "direct", .8), .8)
        self.assertEqual(self.map.calibrate(replace(future, family="other"), "direct", .8), .8)
        self.assertEqual(self.map.calibrate(self.task, "direct", .8), .8)
        reopened = type(self.map)(self.root / "map.sqlite3")
        self.assertEqual(reopened.summary()[0]["attempts"], 1)

    def test_duplicate_attempt_rejected(self):
        self.record(self.task)
        with self.assertRaises(sqlite3.IntegrityError):
            self.record(self.task)

    def test_retries_cannot_manufacture_persistence_or_sample_size(self):
        for n in range(5):
            self.record(self.task, step=n + 1)
        self.assertFalse(self.map.adaptation_ready(self.task))
        self.assertEqual(self.map.clusters()[0]["distinct_tasks"], 1)
        self.assertAlmostEqual(self.map.calibrate(replace(self.task, id="future"), "direct", .8), .64)

    def test_each_persistent_task_needs_multiple_remedies(self):
        for n in range(3):
            self.record(replace(self.task, id=str(n)))
        self.record(replace(self.task, id="0"), "compute", "compute")
        self.assertFalse(self.map.adaptation_ready(self.task))

    def test_resolved_failures_do_not_justify_training(self):
        self.persistent()
        self.assertTrue(self.map.adaptation_ready(self.task))
        self.record(replace(self.task, id="failure-0"), "tool", "tool", success=True)
        self.assertFalse(self.map.adaptation_ready(self.task))

    def test_infrastructure_failures_do_not_trigger_training(self):
        for n in range(3):
            task = replace(self.task, id=str(n))
            self.record(task, gap=Gap.TOOL, failure_code="tool_error")
            self.record(task, "compute", "compute", gap=Gap.TOOL, failure_code="tool_error")
        self.assertFalse(self.map.adaptation_ready(self.task))


class AdaptationTests(LearningBase):
    def setUp(self):
        super().setUp()
        self.train = tuple(Case(f"train-{n}", f"convert {n}", str(n * 2 + 3)) for n in (1, 2, 3))
        self.heldout = tuple(Case(f"test-{n}", f"convert {n}", str(n * 2 + 3)) for n in (4, 5, 6))
        self.regression = (Case("regression", "2 + 3", "5"),)

    def evaluate(self, **changes):
        kwargs = dict(training=self.train, heldout=self.heldout, regression=self.regression,
                      baseline=arithmetic, trainer=train_affine, approved=True)
        kwargs.update(changes)
        return AdaptationGate(self.map).evaluate("adaptation-1", self.task, Gap.PERSISTENT_SKILL, **kwargs)

    def test_requires_persistence_and_approval(self):
        with self.assertRaises(ValueError):
            self.evaluate()
        self.persistent()
        with self.assertRaises(ValueError):
            self.evaluate(approved=False)

    def test_heldout_gain_accepts_new_version_preserving_old_map(self):
        self.persistent()
        report = self.evaluate()
        self.assertTrue(report["accepted"])
        self.assertEqual(report["new_model"], "model-v1+adaptation-1")
        self.assertEqual(report["target_gain"], 1)
        self.assertEqual(self.map.adaptations()[0]["status"], "accepted")
        self.assertEqual(self.map.summary()[0]["model"], "model-v1")

    def test_no_gain_rejects(self):
        self.persistent()
        report = self.evaluate(trainer=lambda _: Candidate({"type": "noop"}, arithmetic))
        self.assertFalse(report["accepted"])
        self.assertIsNone(report["new_model"])

    def test_regression_blocks_promotion_even_with_target_gain(self):
        self.persistent()
        def regressing(training):
            learned = train_affine(training)
            return Candidate(learned.artifact, lambda prompt: learned.predict(prompt) if prompt.startswith("convert") else "bad")
        report = self.evaluate(trainer=regressing)
        self.assertEqual(report["target_gain"], 1)
        self.assertEqual(report["regressions"], 1)
        self.assertFalse(report["accepted"])

    def test_duplicate_ids_and_inputs_rejected_before_training(self):
        self.persistent()
        for cases in ((replace(self.heldout[0], id=self.train[0].id),) + self.heldout[1:],
                      (replace(self.heldout[0], input=self.train[0].input),) + self.heldout[1:]):
            with self.assertRaises(ValueError):
                self.evaluate(heldout=cases)

    def test_insufficient_evaluation_data_rejected(self):
        self.persistent()
        with self.assertRaises(ValueError):
            self.evaluate(heldout=self.heldout[:1])
        with self.assertRaises(ValueError):
            self.evaluate(regression=())
