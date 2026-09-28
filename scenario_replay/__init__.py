"""Deterministic autonomous-driving scenario replay system.

The same :class:`~scenario_replay.clock.ScenarioClock` drives vehicles,
pedestrians and traffic signals.  Participant state is snapshotted every
tick so the clock can be paused and relocated to an earlier moment, while
already-processed events are never triggered twice.
"""

from .clock import ScenarioClock
from .engine import ReplayEngine, EngineConfig
from .events import Event, EventRecord
from .participants import Participant, Vehicle, Pedestrian, TrafficSignal
from .scenario import Scenario, ScenarioDefinitionError, load_scenario
from .records import save_run, load_run, verify_run, RunRecord

__all__ = [
    "ScenarioClock", "ReplayEngine", "EngineConfig",
    "Event", "EventRecord",
    "Participant", "Vehicle", "Pedestrian", "TrafficSignal",
    "Scenario", "ScenarioDefinitionError", "load_scenario",
    "save_run", "load_run", "verify_run", "RunRecord",
]

__version__ = "1.0.0"
