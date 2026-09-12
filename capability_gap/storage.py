"""Version-scoped empirical map. Selected outcomes only, never counterfactual labels."""
from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3

from .types import Gap, Task, text, number


class CapabilityMap:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path == ":memory:":
            raise ValueError("Use a file path for a persistent capability map")
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS attempts (
                    run TEXT NOT NULL, step INTEGER NOT NULL,
                    task TEXT NOT NULL, family TEXT NOT NULL, model TEXT NOT NULL,
                    intervention TEXT NOT NULL, kind TEXT NOT NULL,
                    success INTEGER NOT NULL, tool_success INTEGER NOT NULL,
                    probability REAL NOT NULL, raw_probability REAL NOT NULL,
                    progress REAL NOT NULL, cost REAL NOT NULL, latency REAL NOT NULL,
                    gap TEXT NOT NULL, failure_code TEXT,
                    PRIMARY KEY (run, step));
                CREATE INDEX IF NOT EXISTS capability_scope ON attempts(model, family, intervention);
                CREATE TABLE IF NOT EXISTS events (
                    run TEXT NOT NULL, sequence INTEGER NOT NULL, payload TEXT NOT NULL,
                    PRIMARY KEY(run, sequence));
                CREATE TABLE IF NOT EXISTS adaptations (
                    id TEXT PRIMARY KEY, model TEXT NOT NULL, family TEXT NOT NULL,
                    gap TEXT NOT NULL, status TEXT NOT NULL, report TEXT NOT NULL);
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def event(self, run: str, sequence: int, payload: dict):
        with self.connection() as db:
            db.execute("INSERT INTO events VALUES (?, ?, ?)",
                       (run, sequence, json.dumps(payload, allow_nan=False)))

    def events(self, run: str) -> list[dict]:
        with self.connection() as db:
            return [json.loads(r[0]) for r in db.execute(
                "SELECT payload FROM events WHERE run=? ORDER BY sequence", (run,))]

    def record(self, *, run, step, task: Task, intervention, kind, success, tool_success,
               probability, raw_probability, progress, cost, latency, gap: Gap, failure_code):
        for value, name in ((run, "run"), (intervention, "intervention"), (kind, "kind")):
            text(value, name)
        if type(step) is not int or step < 1:
            raise ValueError("step must be a positive integer")
        if type(success) is not bool or type(tool_success) is not bool:
            raise ValueError("Outcome flags must be booleans")
        for value, name in ((probability, "probability"), (raw_probability, "raw_probability"), (progress, "progress")):
            number(value, name, 0, 1)
        number(cost, "cost")
        number(latency, "latency")
        if not isinstance(gap, Gap):
            raise ValueError("gap must be a Gap")
        with self.connection() as db:
            db.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                       (run, step, task.id, task.family, task.model, intervention, kind,
                        int(success), int(tool_success), probability, raw_probability,
                        progress, cost, latency, gap.value, failure_code))

    def calibrate(self, task: Task, intervention: str, prior: float) -> float:
        """Shrink current forecast toward empirical successes with prior weight four.

        Average per distinct task first so retries cannot manufacture large support.
        This is a heuristic, not a causal intervention-effect estimator.
        """
        number(prior, "prior", 0, 1)
        with self.connection() as db:
            row = db.execute("""SELECT COUNT(*) AS n, SUM(rate) AS wins FROM (
                SELECT task, AVG(success) AS rate FROM attempts
                WHERE model=? AND family=? AND intervention=? AND task<>? GROUP BY task)
                """, (task.model, task.family, intervention, task.id)).fetchone()
        return (4 * prior + (row["wins"] or 0)) / (4 + row["n"])

    def summary(self) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("""SELECT model, family, intervention, COUNT(*) AS attempts,
                COUNT(DISTINCT task) AS distinct_tasks, AVG(success) AS observed_success_rate,
                AVG((probability-success)*(probability-success)) AS task_success_brier,
                AVG(cost) AS mean_cost, AVG(latency) AS mean_latency
                FROM attempts GROUP BY model, family, intervention ORDER BY model, family, intervention""")
            return [dict(r) for r in rows]

    def clusters(self, *, model=None, family=None) -> list[dict]:
        with self.connection() as db:
            rows = db.execute("""SELECT model, family, gap, failure_code, COUNT(*) AS failures,
                COUNT(DISTINCT task) AS distinct_tasks, COUNT(DISTINCT kind) AS remedies
                FROM attempts WHERE success=0 AND failure_code IS NOT NULL
                AND (? IS NULL OR model=?) AND (? IS NULL OR family=?)
                GROUP BY model, family, gap, failure_code ORDER BY model, family, gap, failure_code""",
                              (model, model, family, family))
            result = [dict(r) for r in rows]
            for row in result:
                eligible_tasks = db.execute("""SELECT COUNT(*) FROM (
                    SELECT a.task FROM attempts a WHERE a.model=? AND a.family=?
                    AND a.gap=? AND a.failure_code=? AND a.success=0
                    AND NOT EXISTS (SELECT 1 FROM attempts b WHERE b.model=a.model
                        AND b.family=a.family AND b.task=a.task AND b.success=1)
                    GROUP BY a.task HAVING COUNT(DISTINCT a.kind)>=2)
                    """, (row["model"], row["family"], row["gap"], row["failure_code"])).fetchone()[0]
                row["unresolved_multi_remedy_tasks"] = eligible_tasks
                row["adaptation_eligible"] = (eligible_tasks >= 3
                    and row["gap"] in {Gap.REASONING.value, Gap.DOMAIN.value, Gap.PERSISTENT_SKILL.value})
        return result

    def adaptation_ready(self, task: Task) -> bool:
        return any(c["adaptation_eligible"] for c in self.clusters(model=task.model, family=task.family))

    def save_adaptation(self, ident: str, model: str, family: str, gap: Gap, status: str, report: dict):
        text(ident, "adaptation ID")
        with self.connection() as db:
            db.execute("INSERT INTO adaptations VALUES (?,?,?,?,?,?)",
                       (ident, model, family, gap.value, status, json.dumps(report, allow_nan=False)))

    def adaptations(self):
        with self.connection() as db:
            return [{**dict(row), "report": json.loads(row["report"])}
                    for row in db.execute("SELECT * FROM adaptations ORDER BY id")]

    def collect_failures(self, model: str, family: str, gap: Gap, output: str | Path, *, approved=False):
        """Export local diagnosis examples for labeling, not unverified training targets."""
        if approved is not True:
            raise ValueError("Failure-data collection requires explicit approval")
        eligible = [c for c in self.clusters(model=model, family=family)
                    if c["gap"] == gap.value and c["adaptation_eligible"]]
        if not eligible:
            raise ValueError("No persistent eligible cluster")
        codes = {c["failure_code"] for c in eligible}
        with self.connection() as db:
            rows = [dict(r) for r in db.execute("""SELECT a.* FROM attempts a
                WHERE a.model=? AND a.family=? AND a.gap=? AND a.success=0
                AND NOT EXISTS (SELECT 1 FROM attempts b WHERE b.model=a.model
                    AND b.family=a.family AND b.task=a.task AND b.success=1)
                ORDER BY a.task, a.run, a.step""", (model, family, gap.value))]
        records = []
        for row in rows:
            if row["failure_code"] not in codes:
                continue
            history = self.events(row["run"])
            start = next((e for e in history if e["event"] == "start"), None)
            if start is None:
                continue
            records.append({"task": start["task"], "intervention": row["intervention"],
                            "gap": gap.value, "failure_code": row["failure_code"],
                            "run": row["run"], "step": row["step"],
                            "verified_target": None, "status": "requires_labeling"})
        with Path(output).open("x", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        return {"records": len(records), "output": str(output), "status": "requires_labeling"}
