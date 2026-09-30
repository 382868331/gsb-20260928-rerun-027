"""两个分支在相同时刻的状态比较。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .engine import CollisionRecord
from .scenario import Vec2


@dataclass(frozen=True)
class BranchComparison:
    tick_ms: int
    positions_equal: bool
    position_deltas: Dict[str, float]
    left_collisions: Tuple[CollisionRecord, ...]
    right_collisions: Tuple[CollisionRecord, ...]
    collisions_equal: bool
    left_reasons: Tuple[str, ...]
    right_reasons: Tuple[str, ...]
    reasons_equal: bool

    @property
    def equal(self) -> bool:
        return (
            self.positions_equal
            and self.collisions_equal
            and self.reasons_equal
        )


def _distance(a: Vec2, b: Vec2) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def compare_at(left: ReplayEngine, right: ReplayEngine, tick_ms: int) -> BranchComparison:
    """把两个引擎静默定位到同一时刻后比较位置、碰撞和触发原因。

    seek 从完整场景确定性重建状态，不重复登记事件，因此可在任意时刻反复比较。
    """
    left.seek(tick_ms)
    right.seek(tick_ms)

    left_positions = left.positions()
    right_positions = right.positions()
    shared_ids = sorted(set(left_positions) & set(right_positions))
    deltas = {
        actor_id: _distance(left_positions[actor_id], right_positions[actor_id])
        for actor_id in shared_ids
    }
    left_collisions = left.collision_at(tick_ms)
    right_collisions = right.collision_at(tick_ms)
    left_reasons = left.triggered_reasons(tick_ms)
    right_reasons = right.triggered_reasons(tick_ms)

    def collision_key(record: CollisionRecord):
        return (record.vehicle_id, record.pedestrian_id)

    return BranchComparison(
        tick_ms=tick_ms,
        positions_equal=all(delta == 0.0 for delta in deltas.values())
        and set(left_positions) == set(right_positions),
        position_deltas=deltas,
        left_collisions=left_collisions,
        right_collisions=right_collisions,
        collisions_equal=[collision_key(c) for c in left_collisions]
        == [collision_key(c) for c in right_collisions],
        left_reasons=left_reasons,
        right_reasons=right_reasons,
        reasons_equal=left_reasons == right_reasons,
    )
