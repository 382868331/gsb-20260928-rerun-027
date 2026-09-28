"""Declarative scenario definitions loaded from JSON.

A scenario pins the schema version, timestep and RNG seed so any run can be
reproduced byte-for-byte.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .participants import Pedestrian, TrafficSignal, Vehicle

SCENARIO_SCHEMA_VERSION = "1.0"


class ScenarioDefinitionError(ValueError):
    pass


@dataclass
class Scenario:
    name: str
    step_seconds: float = 0.5
    seed: int = 0
    duration: float = 20.0
    schema_version: str = SCENARIO_SCHEMA_VERSION
    participants: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    initial_participants: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scenario":
        try:
            sc = cls(
                name=data["name"],
                step_seconds=float(data.get("step_seconds", 0.5)),
                seed=int(data.get("seed", 0)),
                duration=float(data.get("duration", 20.0)),
                schema_version=str(data.get("schema_version",
                                       SCENARIO_SCHEMA_VERSION)),
            )
        except KeyError as exc:
            raise ScenarioDefinitionError(f"missing field {exc}") from exc
        if sc.step_seconds <= 0:
            raise ScenarioDefinitionError("step_seconds must be positive")
        if sc.duration <= 0:
            raise ScenarioDefinitionError("duration must be positive")
        for item in data.get("participants", []):
            sc.participants[item["id"]] = build_participant(item)
        sc.initial_participants = list(data.get("initial_participants") or
                                      data.get("participants", []))
        sc.events = list(data.get("events", []))
        return sc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "step_seconds": self.step_seconds,
            "seed": self.seed,
            "duration": self.duration,
            "participants": self.initial_participants,
            "initial_participants": self.initial_participants,
            "events": self.events,
        }

    def end_tick(self) -> int:
        return round(self.duration / self.step_seconds)


def build_participant(item: dict[str, Any]):
    kind = item.get("kind", "vehicle")
    common = {"pid": item["id"], "x": float(item.get("x", 0.0)),
              "y": float(item.get("y", 0.0)),
              "heading": float(item.get("heading", 0.0))}
    if kind == "vehicle":
        return Vehicle(speed=float(item.get("speed", 0.0)),
                       accel=float(item.get("accel", 0.0)),
                       jitter=float(item.get("jitter", 0.0)), **common)
    if kind == "pedestrian":
        return Pedestrian(walk_speed=float(item.get("walk_speed", 1.2)),
                          hesitation=float(item.get("hesitation", 0.0)),
                          **common)
    if kind == "signal":
        sig = TrafficSignal(phase=item.get("phase", "red"),
                            phase_ticks=int(item.get("phase_ticks", 4)),
                            auto=bool(item.get("auto", True)), **common)
        return sig
    raise ScenarioDefinitionError(f"unknown participant kind {kind!r}")


def load_scenario(path: str | Path) -> Scenario:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return Scenario.from_dict(data)
