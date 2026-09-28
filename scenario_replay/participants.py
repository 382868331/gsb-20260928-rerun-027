"""Scene participants: vehicles, pedestrians and traffic signals.

Each participant integrates its motion with the shared scenario clock using
a fixed timestep, so the same sequence of ticks always produces the same
positions.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


@dataclass
class Participant:
    """Base class for anything that advances with the scenario clock."""

    pid: str
    kind: str = "participant"
    x: float = 0.0
    y: float = 0.0
    heading: float = 0.0

    def step(self, dt: float, rng: random.Random) -> None:
        """Advance one tick. Deterministic given ``rng`` state."""

    def state(self) -> dict:
        return {"id": self.pid, "kind": self.kind, "x": self.x, "y": self.y,
                "heading": self.heading}

    def restore_state(self, state: dict) -> None:
        self.x = float(state["x"])
        self.y = float(state["y"])
        self.heading = float(state["heading"])


@dataclass
class Vehicle(Participant):
    speed: float = 0.0
    accel: float = 0.0
    jitter: float = 0.0
    kind: str = "vehicle"
    _noise_x: float = field(default=0.0, init=False, repr=False)
    _noise_y: float = field(default=0.0, init=False, repr=False)

    def step(self, dt: float, rng: random.Random) -> None:
        self.speed += self.accel * dt
        dist = self.speed * dt
        self.x += dist * math.cos(self.heading)
        self.y += dist * math.sin(self.heading)
        if self.jitter:
            self._noise_x = rng.uniform(-self.jitter, self.jitter)
            self._noise_y = rng.uniform(-self.jitter, self.jitter)

    def observed_position(self) -> tuple[float, float]:
        """Position including the deterministic sensor jitter."""
        return self.x + self._noise_x, self.y + self._noise_y

    def state(self) -> dict:
        s = super().state()
        s.update({"speed": self.speed, "accel": self.accel,
                  "noise_x": self._noise_x, "noise_y": self._noise_y})
        return s

    def restore_state(self, state: dict) -> None:
        super().restore_state(state)
        self.speed = float(state["speed"])
        self.accel = float(state.get("accel", self.accel))
        self._noise_x = float(state.get("noise_x", 0.0))
        self._noise_y = float(state.get("noise_y", 0.0))


@dataclass
class Pedestrian(Participant):
    speed: float = 0.0
    walk_speed: float = 1.2
    hesitation: float = 0.0
    moving: bool = True
    kind: str = "pedestrian"

    def step(self, dt: float, rng: random.Random) -> None:
        if self.hesitation and rng.random() < self.hesitation:
            self.moving = False
        else:
            self.moving = True
        if self.moving:
            dist = self.walk_speed * dt
            self.x += dist * math.cos(self.heading)
            self.y += dist * math.sin(self.heading)

    def state(self) -> dict:
        s = super().state()
        s.update({"walk_speed": self.walk_speed, "moving": self.moving})
        return s

    def restore_state(self, state: dict) -> None:
        super().restore_state(state)
        self.walk_speed = float(state.get("walk_speed", self.walk_speed))
        self.moving = bool(state.get("moving", True))


PHASE_CYCLE = ("red", "green", "yellow")


@dataclass
class TrafficSignal(Participant):
    """Fixed-cycle signal; phase changes only at scheduled tick times."""

    phase: str = "red"
    phase_ticks: int = 4
    auto: bool = True
    kind: str = "signal"

    def step(self, dt: float, rng: random.Random) -> None:
        return None

    def set_phase(self, phase: str) -> None:
        if phase not in PHASE_CYCLE:
            raise ValueError(f"unknown signal phase {phase!r}")
        self.phase = phase

    def state(self) -> dict:
        return {"id": self.pid, "kind": self.kind, "phase": self.phase,
                "x": self.x, "y": self.y, "heading": self.heading,
                "auto": self.auto}

    def restore_state(self, state: dict) -> None:
        super().restore_state(state)
        self.phase = state["phase"]
        self.auto = bool(state.get("auto", self.auto))


def distance(a: Participant, b: Participant) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)
