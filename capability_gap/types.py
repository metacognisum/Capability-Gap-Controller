"""Application-owned contracts and explicit meanings for predicted probabilities."""
from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Callable

from foresight import Evidence, Forecast, Requirement, State


def number(value, name, low=0, high=math.inf):
    if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError(f"{name} must be a finite number in [{low}, {high}]")


def text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")


class Gap(str, Enum):
    KNOWLEDGE = "knowledge"
    REASONING = "reasoning"
    TOOL = "tool"
    CONTEXT = "context"
    DOMAIN = "domain"
    COMPUTE = "compute"
    PERSISTENT_SKILL = "persistent_skill"
    UNKNOWN = "unknown"


class Kind(str, Enum):
    DIRECT = "direct"
    RETRIEVAL = "retrieval"
    TOOL = "tool"
    COMPUTE = "compute"
    SPECIALIST = "specialist"
    EXAMPLES = "examples"
    HUMAN = "human"
    SKILL = "skill"
    ADAPTER = "adapter"
    LORA = "lora"
    FINETUNE = "fine_tune"
    ABSTAIN = "abstain"


ADAPTATIONS = frozenset((Kind.SKILL, Kind.ADAPTER, Kind.LORA, Kind.FINETUNE))


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    model: str
    prompt: str
    requirements: tuple[Requirement, ...]

    def __post_init__(self):
        for key in ("id", "family", "model", "prompt"):
            text(getattr(self, key), key)
        if not self.requirements or len({r.claim for r in self.requirements}) != len(self.requirements):
            raise ValueError("Task requirements must be nonempty with unique claims")


@dataclass(frozen=True)
class Assessment:
    success_now: float
    gaps: tuple[Gap, ...]
    failure_points: tuple[str, ...]

    def __post_init__(self):
        number(self.success_now, "success_now", 0, 1)
        if not self.gaps or any(not isinstance(g, Gap) for g in self.gaps):
            raise ValueError("Assessment needs at least one gap hypothesis (unknown is allowed)")
        for point in self.failure_points:
            text(point, "failure point")


@dataclass(frozen=True)
class Intervention:
    id: str
    kind: Kind
    description: str
    cost: float
    latency: float
    risk: float
    source: str
    authoritative: bool = True
    current: bool = True
    requires_approval: bool = False

    def __post_init__(self):
        for key in ("id", "description", "source"):
            text(getattr(self, key), key)
        if not isinstance(self.kind, Kind):
            raise ValueError("kind must be a Kind")
        number(self.cost, "cost")
        number(self.latency, "latency")
        number(self.risk, "risk", 0, 1)
        for key in ("authoritative", "current", "requires_approval"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be boolean")
        if self.kind in ADAPTATIONS and not self.requires_approval:
            raise ValueError("Training and skill-creation interventions require explicit approval")


@dataclass(frozen=True)
class Estimate:
    intervention: str
    task_success: float
    expected_progress: float
    expected_confidence: float
    forecast: Forecast

    def __post_init__(self):
        text(self.intervention, "intervention")
        for key in ("task_success", "expected_progress", "expected_confidence"):
            number(getattr(self, key), key, 0, 1)
        if self.forecast.action != self.intervention:
            raise ValueError("Forecast action must match intervention")


@dataclass(frozen=True)
class Execution:
    """Executor observations; confidence is actor-reported, never proof of success."""
    evidence: tuple[Evidence, ...]
    tool_succeeded: bool
    cost: float
    latency: float
    confidence: float | None = None
    failure_code: str | None = None

    def __post_init__(self):
        State(self.evidence)
        if type(self.tool_succeeded) is not bool:
            raise ValueError("tool_succeeded must be boolean")
        if not self.tool_succeeded and self.evidence:
            raise ValueError("A failed tool must not supply partial evidence")
        number(self.cost, "actual cost")
        number(self.latency, "actual latency")
        if self.confidence is not None:
            number(self.confidence, "confidence", 0, 1)
        if self.failure_code is not None:
            text(self.failure_code, "failure_code")


@dataclass(frozen=True)
class Policy:
    max_steps: int = 6
    max_cost: float = 10
    max_latency: float = 60
    max_risk: float = .3
    cost_weight: float = .05
    latency_weight: float = .005
    risk_weight: float = .5
    mismatch_threshold: float = .3
    confidence_drop: float = .25
    stall_steps: int = 2
    failure_steps: int = 2
    max_attempts_per_intervention: int = 2
    abstain_utility: float = -.2
    approved: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self):
        for key in ("max_steps", "stall_steps", "failure_steps", "max_attempts_per_intervention"):
            if type(getattr(self, key)) is not int or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("max_cost", "max_latency", "cost_weight", "latency_weight", "risk_weight"):
            number(getattr(self, key), key)
        for key in ("max_risk", "mismatch_threshold", "confidence_drop"):
            number(getattr(self, key), key, 0, 1)
        number(self.abstain_utility, "abstain_utility", -1, 1)


Handler = Callable[[Task, State, Intervention], Execution]
