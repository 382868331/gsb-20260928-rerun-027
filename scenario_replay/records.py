"""Persist scenario version, seed, event log and per-tick positions."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .events import EventRecord
from .scenario import SCENARIO_SCHEMA_VERSION, Scenario

RUN_RECORD_VERSION = "1.0"


@dataclass
class RunRecord:
    record_version: str
    schema_version: str
    scenario_name: str
    seed: int
    step_seconds: float
    end_tick: int
    end_time: float
    events: list[dict] = field(default_factory=list)
    positions: list[dict] = field(default_factory=list)
    scenario: Optional[dict] = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunRecord":
        return cls(**d)


def build_run_record(engine, include_scenario: bool = True) -> RunRecord:
    scenario = engine.scenario
    return RunRecord(
        record_version=RUN_RECORD_VERSION,
        schema_version=SCENARIO_SCHEMA_VERSION,
        scenario_name=scenario.name, seed=scenario.seed,
        step_seconds=scenario.step_seconds, end_tick=engine.clock.tick,
        end_time=engine.clock.time,
        events=[r.to_dict() for r in engine.log],
        positions=engine.positions,
        scenario=scenario.to_dict() if include_scenario else None)


def save_run(engine, path: str | Path) -> Path:
    path = Path(path)
    record = build_run_record(engine)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record.to_dict(), fh, indent=2, sort_keys=True)
    return path


def load_run(path: str | Path) -> RunRecord:
    with open(path, "r", encoding="utf-8") as fh:
        return RunRecord.from_dict(json.load(fh))


def verify_run(path: str | Path) -> dict[str, Any]:
    """Re-run a saved record from its pinned version and seed and compare.

    Returns a report with event and position equality.  A clean replay must
    reproduce both the event records (ids, order, tick, sequence) and every
    participant position exactly, proving the run is deterministic.
    """
    from .engine import ReplayEngine

    record = load_run(path)
    if record.scenario is None:
        raise ValueError("record lacks the embedded scenario; cannot replay")
    scenario = Scenario.from_dict(record.scenario)
    engine = ReplayEngine(scenario)
    # Replay in lockstep so an interruption stops the rerun at exactly the
    # same tick as the original run.
    while engine.clock.tick < record.end_tick and not engine.interrupted:
        engine.step()
    replay_events = [r.to_dict() for r in engine.log]
    replay_positions = engine.positions
    events_match = replay_events == record.events
    positions_match = replay_positions == record.positions
    return {"events_match": events_match, "positions_match": positions_match,
            "recorded_events": len(record.events),
            "replayed_events": len(replay_events),
            "recorded_ticks": len(record.positions),
            "replayed_ticks": len(replay_positions),
            "ok": events_match and positions_match}
