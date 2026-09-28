"""Build Event objects from declarative scenario JSON entries."""

from __future__ import annotations

from typing import Any

from .events import Event, InterruptScenario, proximity, signal_is
from .participants import TrafficSignal


def build_event(spec: dict[str, Any], step_seconds: float) -> Event:
    at = float(spec.get("at", 0.0))
    tick = round(at / step_seconds)
    etype = spec["type"]
    event = Event(event_id=spec["id"], tick=tick, etype=etype,
                  order=int(spec.get("order", 100)),
                  description=spec.get("description", ""))
    if etype == "signal_change":
        target = spec["signal"]
        phase = spec["phase"]
        event.effect = _signal_effect(target, phase)
    elif etype == "speed_change":
        vehicle_id = spec["vehicle"]
        speed = float(spec.get("speed", 0.0))
        accel = spec.get("accel")
        event.effect = _speed_effect(vehicle_id, speed, accel)
    elif etype == "proximity_warning":
        event.condition = proximity(spec["a"], spec["b"],
                                    float(spec["threshold"]))
        event.effect = _proximity_payload(spec["a"], spec["b"])
    elif etype == "interrupt":
        reason = spec.get("reason", "manual interrupt")
        if "a" in spec and "b" in spec:
            event.condition = proximity(spec["a"], spec["b"],
                                        float(spec["threshold"]))
        event.effect = _interrupt_effect(reason)
    elif etype == "wait_for_signal":
        event.condition = signal_is(spec["signal"], spec["phase"])
    else:
        raise ValueError(f"unknown event type {etype!r}")
    return event


def _signal_effect(signal_id: str, phase: str):
    def _apply(engine):
        sig = engine.participants[signal_id]
        assert isinstance(sig, TrafficSignal)
        sig.set_phase(phase)
        sig.auto = False
        return {"signal": signal_id, "phase": phase}
    return _apply


def _speed_effect(vehicle_id: str, speed: float, accel):
    def _apply(engine):
        veh = engine.participants[vehicle_id]
        veh.speed = speed
        if accel is not None:
            veh.accel = float(accel)
        return {"vehicle": vehicle_id, "speed": speed}
    return _apply


def _proximity_payload(a_id: str, b_id: str):
    from .participants import distance
    def _apply(engine):
        return {"a": a_id, "b": b_id,
                "distance": distance(engine.participants[a_id],
                                     engine.participants[b_id])}
    return _apply


def _interrupt_effect(reason: str):
    def _apply(engine):
        raise InterruptScenario(reason)
    return _apply
