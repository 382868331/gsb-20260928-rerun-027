"""场景版本、运行种子与运行状态的保存/加载（JSON）。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from .engine import (
    CollisionRecord,
    EngineState,
    EventRecord,
    PerceptionRecord,
    ReplayEngine,
    ResponseRecord,
)
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
        "reaction_delays": {
            actor_id: dict(delays)
            for actor_id, delays in scenario.reaction_delays.items()
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
        reaction_delays={
            actor_id: {k: float(v) for k, v in delays.items()}
            for actor_id, delays in data.get("reaction_delays", {}).items()
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
            "reaction_delays": engine.state.reaction_delays,
            "observed_signal_phases": engine.state.observed_signal_phases,
            "observed_speeds": engine.state.observed_speeds,
            "active_actors": engine.state.active_actors,
            "pending_responses": [
                {
                    "response_id": r.response_id,
                    "vehicle_id": r.vehicle_id,
                    "stimulus": r.stimulus,
                    "source_id": r.source_id,
                    "trigger_tick_ms": r.trigger_tick_ms,
                    "due_tick_ms": r.due_tick_ms,
                    "action": r.action,
                    "status": r.status,
                    "order_index": r.order_index,
                    "perception_id": r.perception_id,
                    "superseded_by": r.superseded_by,
                    "reason": r.reason,
                }
                for r in engine.state.pending_responses
                if r.status == "scheduled"
            ],
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
        "perceptions": [
            {
                "perception_id": p.perception_id,
                "tick_ms": p.tick_ms,
                "vehicle_id": p.vehicle_id,
                "stimulus": p.stimulus,
                "source_id": p.source_id,
                "observed": p.observed,
                "response_id": p.response_id,
                "order_index": p.order_index,
                "seq": p.seq,
                "supersedes_response_id": p.supersedes_response_id,
            }
            for p in engine.perceptions
        ],
        "responses": [
            {
                "response_id": r.response_id,
                "vehicle_id": r.vehicle_id,
                "stimulus": r.stimulus,
                "source_id": r.source_id,
                "trigger_tick_ms": r.trigger_tick_ms,
                "due_tick_ms": r.due_tick_ms,
                "action": r.action,
                "status": r.status,
                "order_index": r.order_index,
                "perception_id": r.perception_id,
                "superseded_by": r.superseded_by,
                "reason": r.reason,
            }
            for r in engine.responses
        ],
    }
    if engine.pending_revision is not None:
        payload["pending_revision"] = _revision_to_dict(engine.pending_revision)
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
        reaction_delays={
            k: {pk: float(pv) for pk, pv in v.items()}
            for k, v in st.get("reaction_delays", {}).items()
        },
        observed_signal_phases=dict(st.get("observed_signal_phases", {})),
        observed_speeds={k: float(v) for k, v in st.get("observed_speeds", {}).items()},
        active_actors={
            actor.actor_id: bool(
                st.get("active_actors", {}).get(actor.actor_id, True)
            )
            for actor in scenario.ordered_actors()
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
    engine.perceptions = [
        PerceptionRecord(
            perception_id=p["perception_id"],
            tick_ms=int(p["tick_ms"]),
            vehicle_id=p["vehicle_id"],
            stimulus=p["stimulus"],
            source_id=p["source_id"],
            observed=dict(p["observed"]),
            response_id=p["response_id"],
            order_index=int(p["order_index"]),
            seq=int(p.get("seq", 0)),
            supersedes_response_id=p.get("supersedes_response_id"),
        )
        for p in payload.get("perceptions", ())
    ]
    engine.responses = [
        ResponseRecord(
            response_id=r["response_id"],
            vehicle_id=r["vehicle_id"],
            stimulus=r["stimulus"],
            source_id=r["source_id"],
            trigger_tick_ms=int(r["trigger_tick_ms"]),
            due_tick_ms=int(r["due_tick_ms"]),
            action=dict(r["action"]),
            status=r.get("status", "executed"),
            order_index=int(r.get("order_index", 0)),
            perception_id=r.get("perception_id"),
            superseded_by=r.get("superseded_by"),
            reason=r.get("reason"),
        )
        for r in payload.get("responses", ())
    ]
    engine.state.pending_responses = [
        ResponseRecord(
            response_id=r["response_id"],
            vehicle_id=r["vehicle_id"],
            stimulus=r["stimulus"],
            source_id=r["source_id"],
            trigger_tick_ms=int(r["trigger_tick_ms"]),
            due_tick_ms=int(r["due_tick_ms"]),
            action=dict(r["action"]),
            status=r.get("status", "scheduled"),
            order_index=int(r.get("order_index", 0)),
            perception_id=r.get("perception_id"),
            superseded_by=r.get("superseded_by"),
            reason=r.get("reason"),
        )
        for r in st.get("pending_responses", ())
    ]
    engine._handled_perceptions = {p.perception_id for p in engine.perceptions}
    engine._handled_responses = {r.response_id for r in engine.responses}
    if "pending_revision" in payload:
        engine.pending_revision = _revision_from_dict(payload["pending_revision"])
    return engine


def _revision_to_dict(revision) -> dict:
    return {
        "revision_id": revision.revision_id,
        "created_tick_ms": revision.created_tick_ms,
        "base_scheme_id": revision.base_scheme_id,
        "status": revision.status,
        "windows": [
            {
                "window_id": w.window_id,
                "vehicle_id": w.vehicle_id,
                "stimulus": w.stimulus,
                "source_id": w.source_id,
                "earliest_ms": w.earliest_ms,
                "latest_ms": w.latest_ms,
                "delay_ms": w.delay_ms,
                "observed": dict(w.observed),
                "response_action": (
                    dict(w.response_action)
                    if w.response_action is not None
                    else None
                ),
            }
            for w in revision.windows
        ],
        "constraints": [
            {
                "kind": c.kind,
                "first_id": c.first_id,
                "second_id": c.second_id,
            }
            for c in revision.constraints
        ],
    }


def _revision_from_dict(data: dict):
    from .uncertainty import PerceptionWindow, RevisionRequest, WindowConstraint

    return RevisionRequest(
        revision_id=data["revision_id"],
        created_tick_ms=int(data.get("created_tick_ms", 0)),
        base_scheme_id=data.get("base_scheme_id"),
        status=data.get("status", "pending"),
        windows=tuple(
            PerceptionWindow(
                window_id=w["window_id"],
                vehicle_id=w["vehicle_id"],
                stimulus=w["stimulus"],
                source_id=w["source_id"],
                earliest_ms=int(w["earliest_ms"]),
                latest_ms=int(w["latest_ms"]),
                delay_ms=int(w["delay_ms"]),
                observed=dict(w.get("observed", {})),
                response_action=(
                    dict(w["response_action"])
                    if w.get("response_action") is not None
                    else None
                ),
            )
            for w in data["windows"]
        ),
        constraints=tuple(
            WindowConstraint(
                kind=c["kind"],
                first_id=c["first_id"],
                second_id=c["second_id"],
            )
            for c in data.get("constraints", ())
        ),
    )


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
