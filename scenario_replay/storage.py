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
        "branch": scenario.branch,
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
        branch=data.get("branch"),
    )


def save_run(engine: ReplayEngine, path: str | Path) -> Path:
    """保存场景定义（含版本与种子）以及当前运行状态和事件日志。"""
    payload = {
        "scenario": scenario_to_dict(engine.scenario),
        "state": {
            "tick_ms": engine.state.tick_ms,
            "distances": engine.state.distances,
            "speeds": engine.state.speeds,
            "paths": {
                actor_id: [list(point) for point in path]
                for actor_id, path in engine.state.paths.items()
            },
            "signal_plans": {
                signal_id: [list(item) for item in plan]
                for signal_id, plan in engine.state.signal_plans.items()
            },
            "signal_overrides": engine.state.signal_overrides,
            "interrupted": engine.state.interrupted,
            "paused": engine.paused,
        },
        "journal": [
            {
                "tick_ms": r.tick_ms,
                "event_id": r.event_id,
                "order_index": r.order_index,
                "action": r.action,
            }
            for r in engine.journal
        ],
        "collisions": [
            {
                "tick_ms": record.tick_ms,
                "vehicle_id": record.vehicle_id,
                "pedestrian_id": record.pedestrian_id,
                "vehicle_position": list(record.vehicle_position),
                "pedestrian_position": list(record.pedestrian_position),
                "distance": record.distance,
            }
            for record in engine.collisions
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
        paths={
            actor_id: tuple((float(x), float(y)) for x, y in path)
            for actor_id, path in st.get("paths", {}).items()
        },
        signal_plans={
            signal_id: tuple((int(tick), str(phase)) for tick, phase in plan)
            for signal_id, plan in st.get("signal_plans", {}).items()
        },
        signal_overrides=dict(st["signal_overrides"]),
        interrupted=bool(st["interrupted"]),
    )
    engine.paused = bool(st.get("paused", False))
    engine.journal = [
        EventRecord(
            tick_ms=int(r["tick_ms"]),
            event_id=r["event_id"],
            order_index=int(r["order_index"]),
            action=r["action"],
        )
        for r in payload["journal"]
    ]
    engine._fired = {r.event_id for r in engine.journal}
    engine.collisions = [
        CollisionRecord(
            tick_ms=int(record["tick_ms"]),
            vehicle_id=record["vehicle_id"],
            pedestrian_id=record["pedestrian_id"],
            vehicle_position=tuple(float(v) for v in record["vehicle_position"]),
            pedestrian_position=tuple(
                float(v) for v in record["pedestrian_position"]
            ),
            distance=float(record["distance"]),
        )
        for record in payload.get("collisions", [])
    ]
    engine._collided_pairs = {
        (record.vehicle_id, record.pedestrian_id) for record in engine.collisions
    }
    return engine


def _actor_to_dict(actor) -> dict:
    return {
        "actor_id": actor.actor_id,
        "speed": actor.speed,
        "path": [[x, y] for x, y in actor.path],
    }


def _actor_from_dict(data: dict, cls):
    return cls(
        actor_id=data["actor_id"],
        speed=float(data["speed"]),
        path=tuple((float(x), float(y)) for x, y in data["path"]),
    )
