"""确定性自动驾驶场景回放系统。"""
from .scenario import Scenario, Vehicle, Pedestrian, Signal, ScriptedEvent
from .engine import ReplayEngine, EngineState
from .storage import save_run, load_run
from .branching import BranchChanges, branch_from
from .comparison import (
    BranchComparison,
    CollisionSnapshot,
    FirstDivergence,
    compare_at,
    first_divergence,
)

__all__ = [
    "Scenario",
    "Vehicle",
    "Pedestrian",
    "Signal",
    "ScriptedEvent",
    "ReplayEngine",
    "EngineState",
    "save_run",
    "load_run",
    "BranchChanges",
    "branch_from",
    "BranchComparison",
    "CollisionSnapshot",
    "compare_at",
    "FirstDivergence",
    "first_divergence",
]
