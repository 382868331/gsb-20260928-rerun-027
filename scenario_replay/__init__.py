"""确定性自动驾驶场景回放系统。"""
from .scenario import Scenario, Vehicle, Pedestrian, Signal, ScriptedEvent
from .engine import (
    EngineState,
    PerceptionRecord,
    ReplayEngine,
    ResponseRecord,
)
from .storage import save_run, load_run
from .branching import BranchChanges, branch_from
from .comparison import (
    BranchComparison,
    CollisionSnapshot,
    Divergence,
    compare_at,
    first_divergence,
    first_response_comparison,
)

__all__ = [
    "Scenario",
    "Vehicle",
    "Pedestrian",
    "Signal",
    "ScriptedEvent",
    "ReplayEngine",
    "EngineState",
    "PerceptionRecord",
    "ResponseRecord",
    "save_run",
    "load_run",
    "BranchChanges",
    "branch_from",
    "BranchComparison",
    "CollisionSnapshot",
    "compare_at",
    "Divergence",
    "first_divergence",
    "first_response_comparison",
]
