"""从任意时刻建立独立分支。

分支点之前，分支与父场景使用完全相同的参与者、路线、事件与信号计划，
因此分支点及以前的轨迹/事件保持原样；分支修改从 branch_tick_ms 这一 tick
的状态转移开始生效。返回的 Scenario 是完整自足的定义，可独立保存重开。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from .scenario import Scenario, ScriptedEvent, Vec2


@dataclass(frozen=True)
class BranchChanges:
    # actor_id -> 分支点之后的新折线（世界坐标）
    pedestrian_paths: Dict[str, Tuple[Vec2, ...]] = field(default_factory=dict)
    # signal_id -> 分支点之后生效的新相位计划；时刻使用绝对毫秒
    signal_plans: Dict[str, Tuple[Tuple[int, str], ...]] = field(default_factory=dict)
    # actor_id -> 分支点起的车辆反应速度（米/秒）
    vehicle_speeds: Dict[str, float] = field(default_factory=dict)


def create_branch(
    parent: Scenario,
    branch_tick_ms: int,
    changes: BranchChanges,
    name: str,
) -> Scenario:
    """在 branch_tick_ms 建立分支；修改事件在分支 tick 按固定顺序生效。

    顺序：先替换车辆反应参数，再替换行人路线，最后替换信号灯计划，
    然后才执行该 tick 原有的普通事件，保证同刻“变灯与刹车”比较确定。
    """
    if branch_tick_ms < 0 or branch_tick_ms % parent.tick_ms != 0:
        raise ValueError("branch_tick_ms must align with scenario tick_ms")
    _validate_actors(parent, changes)
    _validate_signals(parent, changes)

    branch_events = list(parent.events)
    base_seq = -10000  # 分支改动先于该 tick 的普通脚本事件

    for index, (actor_id, speed) in enumerate(
        sorted(changes.vehicle_speeds.items())
    ):
        branch_events.append(
            ScriptedEvent(
                event_id=f"branch:{name}:speed:{actor_id}",
                tick_ms=branch_tick_ms,
                action={"type": "set_speed", "actor_id": actor_id, "speed": speed},
                seq=base_seq + index,
            )
        )

    path_offset = len(changes.vehicle_speeds)
    for index, (actor_id, path) in enumerate(
        sorted(changes.pedestrian_paths.items())
    ):
        branch_events.append(
            ScriptedEvent(
                event_id=f"branch:{name}:path:{actor_id}",
                tick_ms=branch_tick_ms,
                action={
                    "type": "set_path",
                    "actor_id": actor_id,
                    "path": [list(point) for point in path],
                },
                seq=base_seq + path_offset + index,
            )
        )

    signal_offset = path_offset + len(changes.pedestrian_paths)
    for index, (signal_id, phases) in enumerate(
        sorted(changes.signal_plans.items())
    ):
        ordered = tuple(
            sorted(((int(tick), str(phase)) for tick, phase in phases), key=lambda p: p[0])
        )
        branch_events.append(
            ScriptedEvent(
                event_id=f"branch:{name}:signal:{signal_id}",
                tick_ms=branch_tick_ms,
                action={
                    "type": "set_signal_plan",
                    "signal_id": signal_id,
                    "phases": [list(item) for item in ordered],
                },
                seq=base_seq + signal_offset + index,
            )
        )

    parent_meta = parent.branch or {}
    return Scenario(
        name=parent.name,
        version=parent.version,
        seed=parent.seed,
        vehicles=parent.vehicles,
        pedestrians=parent.pedestrians,
        signals=parent.signals,
        events=tuple(branch_events),
        tick_ms=parent.tick_ms,
        branch={
            "name": name,
            "parent": parent_meta.get("name", "root"),
            "branch_tick_ms": branch_tick_ms,
        },
    )


def _validate_actors(parent: Scenario, changes: BranchChanges) -> None:
    ped_ids = {actor.actor_id for actor in parent.pedestrians}
    vehicle_ids = {actor.actor_id for actor in parent.vehicles}
    for actor_id, path in changes.pedestrian_paths.items():
        if actor_id not in ped_ids:
            raise KeyError(f"branch modifies unknown pedestrian: {actor_id}")
        if not path:
            raise ValueError(f"pedestrian path must not be empty: {actor_id}")
    for actor_id in changes.vehicle_speeds:
        if actor_id not in vehicle_ids:
            raise KeyError(f"branch modifies unknown vehicle: {actor_id}")


def _validate_signals(parent: Scenario, changes: BranchChanges) -> None:
    ids = {signal.signal_id for signal in parent.signals}
    for signal_id, phases in changes.signal_plans.items():
        if signal_id not in ids:
            raise KeyError(f"branch modifies unknown signal: {signal_id}")
        if not phases:
            raise ValueError(f"signal plan must not be empty: {signal_id}")

