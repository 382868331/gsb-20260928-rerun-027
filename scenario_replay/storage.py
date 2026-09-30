"""场景版本、运行种子与运行状态的保存/加载（JSON）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .engine import CollisionRecord, EngineState, EventRecord, ReplayEngine
from .scenario import Pedestrian, Scenario, ScriptedEvent, Signal, Vehicle

if TYPE_CHECKING:
    pass


def scenario_to_dict(scenario: Scenario) -> dict:
    return {
        "name": scenario.name,
        "version": scenario.version,
        "seed": scenario.seed,
        "tick_ms": scenario.tick_ms,
        "vehicles": [_actor_to_dict(a) for a in scenario.vehicles],
        "pedestrians": [_actor_to_dict(a) for a in scenario.pedestrians],
        "signals": [
            {"signal_id": s.signal_id, "phases": [list(p) for p in s.phases]}
            for s in scenario.signals
        ],
        "events": [
            {
                "event_id": e.event_id,
                "tick_ms": e.tick_ms,
                "seq": e.seq,
                "action": e.action,
            }
            for e in scenario.events
        ],
        "branch_id": scenario.branch_id,
        "parent_id": scenario.parent_id,
        "branch_point_ms": scenario.branch_point_ms,
        "reaction_params": {
            actor_id: dict(params)
            for actor_id, params in scenario.reaction_params.items()
        },
    }


def scenario_from_dict(data: dict) -> Scenario:
    vehicles = tuple(_actor_from_dict(d, Vehicle) for d in data["vehicles"])
    peds = tuple(_actor_from_dict(d, Pedestrian) for d in data["pedestrians"])
    signals = tuple(
        Signal(s["signal_id"], tuple((p[0], p[1]) for p in s["phases"]))
        for s in data["signals"]
    )
    events = tuple(
        ScriptedEvent(e["event_id"], e["tick_ms"], e["action"], e.get("seq", 0))
        for e in data["events"]
    )
    return Scenario(
        name=data["name"],
        version=data["version"],
        seed=int(data["seed"]),
        vehicles=vehicles,
        pedestrians=peds,
        signals=signals,
        events=events,
        tick_ms=int(data.get("tick_ms", 100)),
        branch_id=data.get("branch_id"),
        parent_id=data.get("parent_id"),
        branch_point_ms=data.get("branch_point_ms"),
        reaction_params={
            actor_id: {k: float(v) for k, v in params.items()}
            for actor_id, params in data.get("reaction_params", {}).items()
        },
    )


def save_run(engine: ReplayEngine, path: str | Path) -> Path:
    """保存场景定义（含版本与种子）以及当前运行状态和事件日志。"""
    payload = {
        "scenario": scenario_to_dict(engine.scenario),
        "state": {
            "tick_ms": engine.state.tick_ms,
            "distances": engine.state.distances,
            "speeds": engine.state.speeds,
            "speed_targets": engine.state.speed_targets,
            "signal_overrides": engine.state.signal_overrides,
            "reaction_params": engine.state.reaction_params,
            "interrupted": engine.state.interrupted,
            "paused": engine.paused,
        },
        "journal": [
            {
                "tick_ms": r.tick_ms,
                "event_id": r.event_id,
                "order_index": r.order_index,
                "action": r.action,
                "reason": r.reason,
            }
            for r in engine.journal
        ],
        "collisions": [
            {
                "tick_ms": c.tick_ms,
                "actor_a": c.actor_a,
                "actor_b": c.actor_b,
                "distance": c.distance,
                "position_a": list(c.position_a),
                "position_b": list(c.position_b),
                "reason": c.reason,
            }
            for c in engine.collisions
        ],
    }
    path = Path(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_run(path: str | Path) -> ReplayEngine:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    scenario = scenario_from_dict(payload["scenario"])
    engine = ReplayEngine(scenario)
    st = payload["state"]
    engine.state = EngineState(
        tick_ms=int(st["tick_ms"]),
        distances={k: float(v) for k, v in st["distances"].items()},
        speeds={k: float(v) for k, v in st["speeds"].items()},
        speed_targets={k: float(v) for k, v in st.get("speed_targets", {}).items()},
        signal_overrides=dict(st["signal_overrides"]),
        reaction_params={
            k: {pk: float(pv) for pk, pv in v.items()}
            for k, v in st.get("reaction_params", {}).items()
        },
        interrupted=bool(st["interrupted"]),
    )
    engine.paused = bool(st.get("paused", False))
    engine.journal = [
        EventRecord(
            tick_ms=int(r["tick_ms"]),
            event_id=r["event_id"],
            order_index=int(r["order_index"]),
            action=r["action"],
            reason=r.get("reason"),
        )
        for r in payload["journal"]
    ]
    engine._fired = {r.event_id for r in engine.journal}
    engine.collisions = [
        CollisionRecord(
            tick_ms=int(c["tick_ms"]),
            actor_a=c["actor_a"],
            actor_b=c["actor_b"],
            distance=float(c["distance"]),
            position_a=tuple(c["position_a"]),
            position_b=tuple(c["position_b"]),
            reason=c["reason"],
        )
        for c in payload.get("collisions", ())
    ]
    engine._collision_pairs = {c.pair for c in engine.collisions}
    return engine


def _actor_to_dict(actor) -> dict:
    return {
        "actor_id": actor.actor_id,
        "speed": actor.speed,
        "path": [[x, y] for x, y in actor.path],
        "radius": actor.radius,
    }


def _actor_from_dict(data: dict, cls):
    return cls(
        actor_id=data["actor_id"],
        speed=float(data["speed"]),
        path=tuple((float(x), float(y)) for x, y in data["path"]),
        radius=float(data.get("radius", cls.radius)),
    )
