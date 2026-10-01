"""两个分支在相同时刻的位置、碰撞与触发原因对比。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .engine import CollisionRecord, ReplayEngine, Vec2


@dataclass(frozen=True)
class CollisionSnapshot:
    pairs: Tuple[Tuple[str, str], ...]
    reasons: Dict[Tuple[str, str], str]


@dataclass(frozen=True)
class BranchComparison:
    tick_ms: int
    positions_match: bool
    collisions_match: bool
    trigger_reasons_match: bool
    positions_a: Dict[str, Vec2]
    positions_b: Dict[str, Vec2]
    collisions_a: CollisionSnapshot
    collisions_b: CollisionSnapshot
    reasons_a: Dict[str, Optional[str]]
    reasons_b: Dict[str, Optional[str]]

    @property
    def equal(self) -> bool:
        return (
            self.positions_match
            and self.collisions_match
            and self.trigger_reasons_match
        )


@dataclass(frozen=True)
class Divergence:
    """两个分支的第一处行为差异。"""

    tick_ms: int
    # 差异维度："progress"（推进中断）/ "positions" / "signals" / "collisions"
    aspects: Tuple[str, ...]
    actors: Tuple[str, ...] = ()
    signals: Tuple[str, ...] = ()


def first_divergence(
    engine_a: ReplayEngine,
    engine_b: ReplayEngine,
    until_ms: int,
    start_ms: int = 0,
) -> Optional[Divergence]:
    """从 start_ms 起逐 tick 对比两个引擎，返回第一处行为差异。

    对比推进进度、参与者位置、信号灯相位与碰撞集合；
    全部一致时返回 None。两个引擎须使用相同的 tick_ms。
    """
    tick_ms = engine_a.scenario.tick_ms
    if tick_ms != engine_b.scenario.tick_ms:
        raise ValueError("engines must share the same tick_ms")
    for tick in range(start_ms, until_ms + 1, tick_ms):
        engine_a.seek(tick)
        engine_b.seek(tick)
        if engine_a.now_ms != engine_b.now_ms:
            return Divergence(tick_ms=tick, aspects=("progress",))
        aspects: list = []
        positions_a = engine_a.positions()
        positions_b = engine_b.positions()
        actors = tuple(
            actor_id
            for actor_id in positions_a
            if positions_a[actor_id] != positions_b.get(actor_id)
        )
        if actors:
            aspects.append("positions")
        phases_a = engine_a.signal_phases()
        phases_b = engine_b.signal_phases()
        signals = tuple(
            signal_id
            for signal_id in phases_a
            if phases_a[signal_id] != phases_b.get(signal_id)
        )
        if signals:
            aspects.append("signals")
        collisions_differ = (
            _collision_snapshot(engine_a, tick).pairs
            != _collision_snapshot(engine_b, tick).pairs
        )
        if collisions_differ:
            aspects.append("collisions")
        if aspects:
            return Divergence(
                tick_ms=tick,
                aspects=tuple(aspects),
                actors=actors,
                signals=signals,
            )
    return None


def compare_at(engine_a: ReplayEngine, engine_b: ReplayEngine, tick_ms: int) -> BranchComparison:
    engine_a.seek(tick_ms)
    engine_b.seek(tick_ms)
    positions_a = engine_a.positions()
    positions_b = engine_b.positions()
    collisions_a = _collision_snapshot(engine_a, tick_ms)
    collisions_b = _collision_snapshot(engine_b, tick_ms)
    reasons_a = _reasons_until(engine_a, tick_ms)
    reasons_b = _reasons_until(engine_b, tick_ms)
    return BranchComparison(
        tick_ms=tick_ms,
        positions_match=positions_a == positions_b,
        collisions_match=(collisions_a.pairs, collisions_a.reasons)
        == (collisions_b.pairs, collisions_b.reasons),
        trigger_reasons_match=reasons_a == reasons_b,
        positions_a=positions_a,
        positions_b=positions_b,
        collisions_a=collisions_a,
        collisions_b=collisions_b,
        reasons_a=reasons_a,
        reasons_b=reasons_b,
    )


def first_response_comparison(
    engine_a: ReplayEngine,
    engine_b: ReplayEngine,
    until_ms: int,
    vehicle_id: str,
    stimulus: Optional[str] = None,
) -> Dict[str, Optional[object]]:
    """比较两个方案中同一车辆的首个感知与实际响应时刻。"""
    engine_a.seek(until_ms)
    engine_b.seek(until_ms)

    def snapshot(engine: ReplayEngine) -> Dict[str, Optional[object]]:
        perception = next(
            (
                record
                for record in engine.perception_records()
                if record.vehicle_id == vehicle_id
                and (stimulus is None or record.stimulus == stimulus)
            ),
            None,
        )
        response = next(
            (
                record
                for record in engine.response_records()
                if record.vehicle_id == vehicle_id
                and (stimulus is None or record.stimulus == stimulus)
            ),
            None,
        )
        return {
            "perception_tick_ms": perception.tick_ms if perception else None,
            "response": response,
            "trigger_tick_ms": response.trigger_tick_ms if response else None,
            "due_tick_ms": response.due_tick_ms if response else None,
            "delay_ms": response.due_tick_ms - response.trigger_tick_ms if response else None,
        }

    return {
        "a": snapshot(engine_a),
        "b": snapshot(engine_b),
    }


def _collision_snapshot(engine: ReplayEngine, tick_ms: int) -> CollisionSnapshot:
    records: Tuple[CollisionRecord, ...] = tuple(
        record for record in engine.collisions if record.tick_ms <= tick_ms
    )
    return CollisionSnapshot(
        pairs=tuple(record.pair for record in records),
        reasons={record.pair: record.reason for record in records},
    )


def _reasons_until(engine: ReplayEngine, tick_ms: int) -> Dict[str, Optional[str]]:
    reasons: Dict[str, Optional[str]] = {}
    for record in engine.journal:
        if record.tick_ms <= tick_ms:
            reasons[record.event_id] = record.reason
    return reasons
