"""Events evaluated on the shared clock with a fixed dispatch order.

Every event has a stable event_id.  Once an event has fired it is recorded
in the processed set; relocating the clock backwards restores an earlier
snapshot (including that set), so replaying forward never triggers the same
event twice within the retained history.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

Condition = Callable[[object], bool]
Effect = Callable[[object], Optional[dict]]


@dataclass
class Event:
    """A scheduled or rule-based scenario event.

    tick is the earliest tick at which the event may fire.  Scheduled
    events fire exactly on that tick; rule events are re-evaluated on every
    later tick until their condition holds.  order breaks ties between
    events firing on the same tick (lower first); event_id is the final
    tie-breaker, making the ordering at one instant fully deterministic.
    """

    event_id: str
    tick: int
    etype: str
    order: int = 100
    recurring: bool = False
    once: bool = True
    condition: Optional[Condition] = None
    effect: Optional[Effect] = None
    payload: dict = field(default_factory=dict)
    description: str = ""

    @property
    def sort_key(self) -> tuple:
        return (self.tick, self.order, self.event_id)

    def is_due(self, tick: int, engine) -> bool:
        if tick < self.tick:
            return False
        if self.condition is None:
            return tick == self.tick
        return bool(self.condition(engine))


@dataclass(frozen=True)
class EventRecord:
    """Immutable log entry produced when an event fires."""

    tick: int
    time: float
    seq: int
    event_id: str
    etype: str
    payload: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"tick": self.tick, "time": self.time, "seq": self.seq,
                "event_id": self.event_id, "etype": self.etype,
                "payload": self.payload}

    @classmethod
    def from_dict(cls, d: dict) -> "EventRecord":
        return cls(d["tick"], d["time"], d["seq"], d["event_id"],
                   d["etype"], d.get("payload", {}))


class InterruptScenario(Exception):
    """An event effect raises this to interrupt the scenario.

    The engine catches it, logs an interrupt record and pauses the clock.
    An interrupted run can be resumed explicitly.
    """

    def __init__(self, reason: str, payload: Optional[dict] = None):
        super().__init__(reason)
        self.reason = reason
        self.payload = payload or {}


def proximity(a_id: str, b_id: str, threshold: float) -> Condition:
    """True when two participants are within threshold metres."""
    from .participants import distance

    def _check(engine) -> bool:
        if a_id not in engine.participants.keys() or b_id not in engine.participants.keys():
            return False
        return distance(engine.participants[a_id],
                        engine.participants[b_id]) <= threshold

    return _check


def signal_is(signal_id: str, phase: str) -> Condition:
    def _check(engine) -> bool:
        sig = engine.participants.get(signal_id)
        return sig is not None and getattr(sig, "phase", None) == phase

    return _check
