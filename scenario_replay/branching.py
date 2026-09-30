"""从任意时刻建立自包含的分支试验。

分支点之前的路径、事件与碰撞记录原样保留；分支点之后的行人路线、
信号灯计划和车辆反应参数以内聚场景数据描述，因此分支可单独保存重开。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from typing import Mapping, Optional, Tuple

from .engine import CollisionRecord, EngineState, EventRecord, ReplayEngine
from .scenario import (
    Pedestrian,
    Scenario,
    ScriptedEvent,
    Signal,
    Vec2,
)


@dataclass(frozen=True)
class BranchChanges:
    vehicle_routes: Mapping[str, Tuple[Vec2, ...]] = field(default_factory=dict)
    pedestrian_routes: Mapping[str, Tuple[Vec2, ...]] = field(default_factory=dict)
    pedestrian_speeds: Mapping[str, float] = field(default_factory=dict)
    signal_plans: Mapping[str, Tuple[Tuple[int, str], ...]] = field(default_factory=dict)
    vehicle_reaction_params: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    extra_events: Tuple[ScriptedEvent, ...] = ()
    drop_event_ids: Tuple[str, ...] = ()


def branch_from(
    engine: ReplayEngine,
    branch_id: str,
    changes: BranchChanges,
    at_ms: Optional[int] = None,
) -> ReplayEngine:
    branch_point = engine.now_ms if at_ms is None else int(at_ms)
    if branch_point % engine.scenario.tick_ms != 0:
        raise ValueError("branch point must align with scenario tick_ms")
    engine.seek(branch_point)

    source = engine.scenario
    known_vehicles = {vehicle.actor_id for vehicle in source.vehicles}
    known_pedestrians = {pedestrian.actor_id for pedestrian in source.pedestrians}
    for actor_id in changes.vehicle_routes:
        if actor_id not in known_vehicles:
            raise KeyError(f"unknown vehicle: {actor_id}")
    for actor_id in changes.pedestrian_routes:
        if actor_id not in known_pedestrians:
            raise KeyError(f"unknown pedestrian: {actor_id}")
    vehicles = tuple(
        _replace_vehicle(vehicle, engine, changes.vehicle_routes.get(vehicle.actor_id))
        for vehicle in source.vehicles
    )
    pedestrians = tuple(
        _replace_pedestrian(pedestrian, engine, changes)
        for pedestrian in source.pedestrians
    )
    signals = tuple(
        _replace_signal(signal, engine, branch_point, changes.signal_plans.get(signal.signal_id))
        for signal in source.signals
    )

    drop_ids = set(changes.drop_event_ids)
    for signal_id in changes.signal_plans:
        drop_ids.update(
            event.event_id
            for event in source.events
            if event.tick_ms > branch_point
            and event.action.get("type") == "set_signal"
            and event.action.get("signal_id") == signal_id
        )

    branch_events = [
        event for event in source.ordered_events() if event.event_id not in drop_ids
    ]
    for event in changes.extra_events:
        if event.tick_ms < branch_point:
            raise ValueError("branch events cannot occur before the branch point")
        branch_events.append(event)

    setup_records: list[EventRecord] = []
    for actor_id, speed in changes.pedestrian_speeds.items():
        if actor_id not in known_pedestrians:
            raise KeyError(f"unknown pedestrian: {actor_id}")
        event = ScriptedEvent(
            event_id=f"{branch_id}:ped-speed:{actor_id}",
            tick_ms=branch_point,
            action={
                "type": "set_speed",
                "actor_id": actor_id,
                "speed": float(speed),
                "reason": f"分支 {branch_id} 修改行人后续速度",
            },
            seq=-9_000,
        )
        branch_events.append(event)
        setup_records.append(
            EventRecord(
                tick_ms=branch_point,
                event_id=event.event_id,
                order_index=len(engine.journal) + len(setup_records),
                action=dict(event.action),
                reason=event.action["reason"],
            )
        )
    for actor_id in changes.vehicle_reaction_params:
        if actor_id not in known_vehicles:
            raise KeyError(f"unknown vehicle: {actor_id}")

    scenario = Scenario(
        name=source.name,
        version=f"{source.version}+branch.{branch_id}",
        seed=source.seed,
        vehicles=tuple(vehicles),
        pedestrians=tuple(pedestrians),
        signals=tuple(signals),
        events=tuple(branch_events),
        tick_ms=source.tick_ms,
        branch_id=branch_id,
        parent_id=source.branch_id or source.name,
        branch_point_ms=branch_point,
        reaction_params={k: dict(v) for k, v in changes.vehicle_reaction_params.items()},
    )

    branch = ReplayEngine(scenario)
    branch.state = _clone_branch_state(engine.state, changes)
    # 事件归属：分支点及之前的记录是共享事实，原样保留；
    # 父场景在分支点之后触发的事件不属于本分支，不得带入。
    branch.journal = [
        record for record in engine.journal if record.tick_ms <= branch_point
    ]
    # 分支点及之前的场景事件视为已处理：向后跳转再重放时
    # 只重建副作用，不得再次登记/累计。
    branch._fired = {record.event_id for record in branch.journal}
    branch._fired.update(
        event.event_id for event in source.events if event.tick_ms <= branch_point
    )
    existing_fired = set(engine._fired)
    setup_by_id = {record.event_id: record for record in setup_records}
    for event in sorted(branch_events, key=lambda e: (e.tick_ms, e.seq, e.event_id)):
        if event.tick_ms != branch_point or event.event_id in existing_fired:
            continue
        branch._apply_action(branch.state, event)
        if event.event_id not in branch._fired:
            record = setup_by_id.get(event.event_id)
            if record is None:
                record = EventRecord(
                    tick_ms=branch_point,
                    event_id=event.event_id,
                    order_index=len(branch.journal),
                    action=dict(event.action),
                    reason=event.action.get("reason"),
                )
            branch.journal.append(record)
            branch._fired.add(event.event_id)
    branch.collisions = [
        _clone_collision(record)
        for record in engine.collisions
        if record.tick_ms <= branch_point
    ]
    branch._collision_pairs = {record.pair for record in branch.collisions}
    branch.paused = False
    return branch


def _replace_vehicle(
    vehicle,
    engine: ReplayEngine,
    new_route: Optional[Tuple[Vec2, ...]],
):
    if new_route is None:
        return vehicle
    return _reroute(vehicle, engine, new_route)


def _replace_pedestrian(
    pedestrian: Pedestrian,
    engine: ReplayEngine,
    changes: BranchChanges,
) -> Pedestrian:
    route = changes.pedestrian_routes.get(pedestrian.actor_id)
    if route is None:
        return pedestrian
    return _reroute(pedestrian, engine, route)


def _reroute(actor, engine: ReplayEngine, new_route: Tuple[Vec2, ...]):
    """保留分支点之前已行驶的轨迹前缀，之后接续替换后的路线。"""
    distance = engine.state.distances[actor.actor_id]
    prefix = _truncate_path(actor.path, distance)
    suffix = tuple(new_route)
    if suffix and _distance(prefix[-1], suffix[0]) < 1e-12:
        suffix = suffix[1:]
    return replace(actor, path=(*prefix, *suffix))


def _replace_signal(
    signal: Signal,
    engine: ReplayEngine,
    branch_point: int,
    new_phases: Optional[Tuple[Tuple[int, str], ...]],
) -> Signal:
    if new_phases is None:
        return signal
    phases = [(start, phase) for start, phase in signal.phases if start < branch_point]
    phases.append((branch_point, engine.signal_phase(signal.signal_id)))
    for start, phase in new_phases:
        if start < branch_point:
            raise ValueError("signal plan changes cannot occur before the branch point")
        if start == branch_point:
            phases[-1] = (start, phase)
        elif start not in {existing_start for existing_start, _ in phases}:
            phases.append((start, phase))
        else:
            phases = [
                (existing_start, phase if existing_start == start else existing_phase)
                for existing_start, existing_phase in phases
            ]
    return Signal(signal.signal_id, tuple(phases))


def _clone_branch_state(state: EngineState, changes: BranchChanges) -> EngineState:
    cloned = state.clone()
    for actor_id, params in changes.vehicle_reaction_params.items():
        cloned.reaction_params.setdefault(actor_id, {}).update(params)
    return cloned


def _clone_collision(record: CollisionRecord) -> CollisionRecord:
    return CollisionRecord(
        tick_ms=record.tick_ms,
        actor_a=record.actor_a,
        actor_b=record.actor_b,
        distance=record.distance,
        position_a=record.position_a,
        position_b=record.position_b,
        reason=record.reason,
    )


def _truncate_path(path: Tuple[Vec2, ...], distance: float) -> Tuple[Vec2, ...]:
    remaining = distance
    points: list[Vec2] = [path[0]]
    for start, end in zip(path, path[1:]):
        length = _distance(start, end)
        if remaining <= length:
            if length == 0:
                points.append(end)
            else:
                ratio = remaining / length
                points.append(
                    (
                        start[0] + (end[0] - start[0]) * ratio,
                        start[1] + (end[1] - start[1]) * ratio,
                    )
                )
            return tuple(points)
        remaining -= length
        points.append(end)
    return tuple(points)


def _distance(a: Vec2, b: Vec2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])
