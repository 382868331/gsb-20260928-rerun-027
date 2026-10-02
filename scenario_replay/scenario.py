"""场景定义：参与者（车辆/行人）、信号灯与脚本事件。

场景对象是不可变的初始定义：运行时变化都由引擎状态承载，
因此同一场景配合同一种子从初始时刻重放必然得到相同结果。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Mapping, Tuple

Vec2 = Tuple[float, float]


@dataclass(frozen=True)
class Actor:
    actor_id: str
    path: Tuple[Vec2, ...]
    speed: float  # 米/秒，沿折线路径的标量速度
    radius: float = 1.0  # 碰撞半径（米）

    def position_at(self, distance: float) -> Vec2:
        """按已行驶距离在折线路径上线性插值，路径尽头后停住。"""
        if not self.path:
            raise ValueError("path must not be empty")
        remaining = distance
        points = self.path
        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]
            seg_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if remaining <= seg_len:
                if seg_len == 0:
                    return points[i]
                ratio = remaining / seg_len
                return (x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio)
            remaining -= seg_len
        return points[-1]


@dataclass(frozen=True)
class Vehicle(Actor):
    kind: str = field(default="vehicle", init=False)
    radius: float = 2.0


@dataclass(frozen=True)
class Pedestrian(Actor):
    kind: str = field(default="pedestrian", init=False)
    radius: float = 0.5


@dataclass(frozen=True)
class Signal:
    signal_id: str
    # 相位表：(开始时刻ms, 相位)，按时间升序
    phases: Tuple[Tuple[int, str], ...]

    def phase_at(self, tick_ms: int) -> str:
        current = self.phases[0][1]
        for start, phase in self.phases:
            if tick_ms >= start:
                current = phase
            else:
                break
        return current


@dataclass(frozen=True)
class ScriptedEvent:
    """脚本事件；seq 固定同一时刻内的先后顺序。"""

    event_id: str
    tick_ms: int
    action: dict
    seq: int = 0


@dataclass(frozen=True)
class PlannedPerception:
    """在指定 tick 注入的一条感知事实（不确定感知方案的具体取值）。

    与引擎自动感知不同：它表示"该方案假定驾驶员在 tick_ms 看到了
    stimulus"，并在 delay_ms 后调度响应；delay_ms 为 None 时只记录
    感知事实不调度动作。plan_id 在场景内唯一，保证重放确定性。
    """

    plan_id: str
    vehicle_id: str
    stimulus: str
    source_id: str
    tick_ms: int
    delay_ms: "int | None"
    observed: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Scenario:
    name: str
    version: str
    seed: int
    # 车辆先于行人推进；同类按列表顺序，构成固定推进顺序
    vehicles: Tuple[Vehicle, ...] = ()
    pedestrians: Tuple[Pedestrian, ...] = ()
    signals: Tuple[Signal, ...] = ()
    events: Tuple[ScriptedEvent, ...] = ()
    tick_ms: int = 100  # 固定积分步长（毫秒）
    # 分支元数据；主场景为 None。分支的完整定义仍内聚在 Scenario 中，
    # 因此分支存档不依赖内存中的父场景。
    branch_id: str | None = None
    parent_id: str | None = None
    branch_point_ms: int | None = None
    reaction_params: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    # 车辆针对各类感知刺激的响应延迟（毫秒）。感知事实按发生时刻记录，
    # 实际动作在 due_ms = 感知时刻 + delay 时才执行。
    reaction_delays: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    # 方案注入的计划感知；覆盖同一 (vehicle, stimulus, source) 的自动感知。
    planned_perceptions: Tuple[PlannedPerception, ...] = ()

    def ordered_actors(self) -> List[Actor]:
        actors: List[Actor] = []
        actors.extend(self.vehicles)
        actors.extend(self.pedestrians)
        return actors

    def ordered_events(self) -> List[ScriptedEvent]:
        """事件全序：时刻 -> seq -> event_id。"""
        return sorted(self.events, key=lambda e: (e.tick_ms, e.seq, e.event_id))
