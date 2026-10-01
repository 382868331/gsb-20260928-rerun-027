"""反应方案对比：首次响应差异、最小间距与冲突前事件链。

从共同历史派生的多个反应方案（分支）共享同一时钟，这里提供
面向"感知-延迟-响应"实验的对比工具。
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

from .engine import ReplayEngine, Vec2


@dataclass(frozen=True)
class GapRecord:
    """两参与者在区间内的最小间距。"""

    tick_ms: int
    actor_a: str
    actor_b: str
    distance: float
    position_a: Vec2
    position_b: Vec2


@dataclass(frozen=True)
class ChainEntry:
    """事件链条目：感知事实、（事件/响应）动作与碰撞按固定顺序合并。"""

    tick_ms: int
    kind: str  # "perception" / "event" / "collision"
    label: str
    detail: Optional[str]


@dataclass(frozen=True)
class ResponseDifference:
    """两个反应方案在感知/响应上的第一处差异。"""

    tick_ms: int
    vehicles: Tuple[str, ...]
    entries_a: Tuple[Tuple, ...]
    entries_b: Tuple[Tuple, ...]


def minimum_gap(
    engine: ReplayEngine,
    actor_a: str,
    actor_b: str,
    until_ms: int,
    start_ms: int = 0,
) -> GapRecord:
    """逐 tick 重放，返回 [start_ms, until_ms] 内两参与者的最小间距。"""
    tick_ms = engine.scenario.tick_ms
    best: Optional[GapRecord] = None
    for tick in range(start_ms, until_ms + 1, tick_ms):
        engine.seek(tick)
        position_a = engine.position_of(actor_a)
        position_b = engine.position_of(actor_b)
        distance = math.hypot(
            position_a[0] - position_b[0], position_a[1] - position_b[1]
        )
        if best is None or distance < best.distance:
            best = GapRecord(
                tick_ms=tick,
                actor_a=actor_a,
                actor_b=actor_b,
                distance=distance,
                position_a=position_a,
                position_b=position_b,
            )
    return best


def event_chain_before(engine: ReplayEngine, tick_ms: int) -> Tuple[ChainEntry, ...]:
    """给定时刻（如首次冲突）之前的事件链：感知、动作与碰撞按
    (tick, 类别, 顺序号) 全序合并，感知事实与实际行动分别成条目。"""
    entries: list[ChainEntry] = []
    for record in engine.journal:
        if record.tick_ms < tick_ms:
            entries.append(
                ChainEntry(record.tick_ms, "event", record.event_id, record.reason)
            )
    for perception in engine.perceptions:
        if perception.tick_ms < tick_ms:
            entries.append(
                ChainEntry(
                    perception.tick_ms,
                    "perception",
                    f"{perception.vehicle_id} 感知 {perception.stimulus}",
                    perception.detail,
                )
            )
    for collision in engine.collisions:
        if collision.tick_ms < tick_ms:
            entries.append(
                ChainEntry(
                    collision.tick_ms,
                    "collision",
                    f"{collision.actor_a} 与 {collision.actor_b}",
                    collision.reason,
                )
            )
    kind_order = {"perception": 0, "event": 1, "collision": 2}
    entries.sort(key=lambda e: (e.tick_ms, kind_order[e.kind], e.label))
    return tuple(entries)


def first_response_difference(
    engine_a: ReplayEngine,
    engine_b: ReplayEngine,
    until_ms: int,
    start_ms: int = 0,
) -> Optional[ResponseDifference]:
    """逐 tick 对比两个方案的感知事实与响应动作，返回第一处差异。"""
    tick_ms = engine_a.scenario.tick_ms
    if tick_ms != engine_b.scenario.tick_ms:
        raise ValueError("engines must share the same tick_ms")
    for tick in range(start_ms, until_ms + 1, tick_ms):
        engine_a.seek(tick)
        engine_b.seek(tick)
        snapshot_a = _response_snapshot(engine_a, tick)
        snapshot_b = _response_snapshot(engine_b, tick)
        if snapshot_a != snapshot_b:
            vehicles = tuple(
                sorted(
                    {
                        entry[1]
                        for entry in snapshot_a + snapshot_b
                        if entry[0] in ("perception", "response")
                    }
                )
            )
            return ResponseDifference(
                tick_ms=tick,
                vehicles=vehicles,
                entries_a=snapshot_a,
                entries_b=snapshot_b,
            )
    return None


def _response_snapshot(engine: ReplayEngine, tick_ms: int) -> Tuple[Tuple, ...]:
    entries: list[Tuple] = []
    for perception in engine.perceptions:
        if perception.tick_ms <= tick_ms:
            entries.append(
                (
                    "perception",
                    perception.vehicle_id,
                    perception.tick_ms,
                    perception.stimulus,
                    perception.detail,
                )
            )
    for record in engine.journal:
        if record.event_id.startswith("react:") and record.tick_ms <= tick_ms:
            entries.append(
                (
                    "response",
                    record.action.get("actor_id"),
                    record.tick_ms,
                    record.event_id,
                )
            )
    return tuple(entries)

