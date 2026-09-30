"""场景时钟驱动的确定性回放引擎。

核心约定：
* 时间以整数毫秒（tick）推进，每 tick 先按固定顺序推进参与者，
  再按 (tick, seq, event_id) 全序触发事件；
* 事件在一次运行中只触发一次。seek 回过去通过从初始场景静默
  重放实现，已处理事件被记录但不会再次写入日志/重复产生副作用；
* 逐步 advance 与直接 seek/run_to 经过完全相同的状态转移，
  到达同一时刻时状态一致；
* pause 动作可中断场景推进。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .scenario import Actor, Scenario, ScriptedEvent

Vec2 = Tuple[float, float]


@dataclass
class EngineState:
    tick_ms: int = 0
    distances: Dict[str, float] = field(default_factory=dict)
    speeds: Dict[str, float] = field(default_factory=dict)
    # 运行时路线（分支改线）；未覆盖时使用场景静态 path
    paths: Dict[str, Tuple[Vec2, ...]] = field(default_factory=dict)
    signal_overrides: Dict[str, str] = field(default_factory=dict)
    # 分支替换的信号灯相位计划：绝对时刻 -> 相位
    signal_plans: Dict[str, Tuple[Tuple[int, str], ...]] = field(default_factory=dict)
    interrupted: bool = False

    def clone(self) -> "EngineState":
        return EngineState(
            tick_ms=self.tick_ms,
            distances=dict(self.distances),
            speeds=dict(self.speeds),
            paths={k: tuple(v) for k, v in self.paths.items()},
            signal_plans={k: tuple(v) for k, v in self.signal_plans.items()},
            signal_overrides=dict(self.signal_overrides),
            interrupted=self.interrupted,
        )


@dataclass(frozen=True)
class EventRecord:
    tick_ms: int
    event_id: str
    order_index: int  # 全局触发序号，同 tick 内反映固定顺序
    action: dict


@dataclass(frozen=True)
class CollisionRecord:
    tick_ms: int
    vehicle_id: str
    pedestrian_id: str
    vehicle_position: Vec2
    pedestrian_position: Vec2
    distance: float

    @property
    def reason(self) -> str:
        return (
            f"vehicle {self.vehicle_id} collided with pedestrian "
            f"{self.pedestrian_id} at t={self.tick_ms}ms"
        )


class ReplayEngine:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.collision_radius = 2.0
        self.state = self._initial_state(scenario)
        self.journal: List[EventRecord] = []
        self.collisions: List[CollisionRecord] = []
        self._fired: set[str] = set()
        self._collided_pairs: set[Tuple[str, str]] = set()
        self.paused = False

    # ---- 基础查询 ----

    @property
    def now_ms(self) -> int:
        return self.state.tick_ms

    def now_seconds(self) -> float:
        return self.state.tick_ms / 1000.0

    def position_of(self, actor_id: str) -> Vec2:
        actor = self._actors_by_id[actor_id]
        path = self.state.paths.get(actor_id, actor.path)
        return self._position_on_path(path, self.state.distances[actor_id])

    def positions(self) -> Dict[str, Vec2]:
        return {aid: self.position_of(aid) for aid in self._actors_by_id}

    def signal_phase(self, signal_id: str) -> str:
        if signal_id in self.state.signal_overrides:
            return self.state.signal_overrides[signal_id]
        if signal_id in self.state.signal_plans:
            return self._phase_from_plan(self.state.signal_plans[signal_id])
        sig = self._signals_by_id[signal_id]
        return sig.phase_at(self.state.tick_ms)

    def signal_phases(self) -> Dict[str, str]:
        return {sid: self.signal_phase(sid) for sid in self._signals_by_id}

    def fired_event_ids(self) -> Tuple[str, ...]:
        return tuple(r.event_id for r in self.journal)

    def collision_at(self, tick_ms: int) -> Tuple[CollisionRecord, ...]:
        return tuple(c for c in self.collisions if c.tick_ms == tick_ms)

    def triggered_reasons(self, tick_ms: int) -> Tuple[str, ...]:
        """返回某一时刻的事件原因与碰撞原因，顺序固定。"""
        reasons = [
            f"event {record.event_id}: {record.action.get('type')}"
            for record in self.journal
            if record.tick_ms == tick_ms
        ]
        reasons.extend(record.reason for record in self.collision_at(tick_ms))
        return tuple(reasons)

    # ---- 时钟控制 ----

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def advance(self, steps: int = 1) -> List[EventRecord]:
        """逐步推进；处于暂停或被中断时不推进。返回新记录的事件。"""
        new_records: List[EventRecord] = []
        for _ in range(steps):
            if self.paused or self.state.interrupted:
                break
            self._step_once(self.state, self._fired, self.journal, new_records)
        return new_records

    def run_to(self, target_ms: int) -> List[EventRecord]:
        """运行到目标时刻（必须不早于当前时刻）；遇到暂停/中断即停。"""
        if target_ms < self.state.tick_ms:
            raise ValueError(
                f"target {target_ms} is before current tick {self.state.tick_ms}; "
                "use seek() to revisit the past"
            )
        new_records: List[EventRecord] = []
        while self.state.tick_ms < target_ms:
            if self.paused or self.state.interrupted:
                break
            self._step_once(self.state, self._fired, self.journal, new_records)
        return new_records

    def seek(self, target_ms: int) -> None:
        """定位到任意时刻。

        从初始场景静默重放到 target_ms：*所有*经过的事件都会执行其
        副作用以重建状态，但已处理事件不会重复记入日志。之后继续运行，
        已触发事件不会再次触发。
        """
        if target_ms < 0:
            raise ValueError("target_ms must be non-negative")
        if target_ms % self.scenario.tick_ms != 0:
            raise ValueError("target_ms must align with scenario tick_ms")

        rebuilt = self._initial_state(self.scenario)
        shadow_fired = set(self._fired)  # 旧事件：重建副作用但不再登记
        shadow_journal = list(self.journal)
        shadow_collisions = list(self.collisions)
        shadow_pairs = set(self._collided_pairs)
        # 静默重放到目标时刻：旧事件/旧碰撞的副作用照常应用以重建状态，
        # 但只有此前未处理过的事件才追加进日志；碰撞不会重复登记
        while rebuilt.tick_ms < target_ms:
            self._step_once(
                rebuilt,
                shadow_fired,
                shadow_journal,
                None,
                shadow_collisions,
                shadow_pairs,
            )

        self.state = rebuilt
        self.journal = shadow_journal
        self.collisions = shadow_collisions
        self._fired = shadow_fired
        self._collided_pairs = shadow_pairs
        self.paused = False

    # ---- 内部实现 ----

    def _step_once(
        self,
        state: EngineState,
        fired: set[str],
        journal: List[EventRecord],
        new_records: Optional[List[EventRecord]],
        collisions: Optional[List[CollisionRecord]] = None,
        collided_pairs: Optional[set[Tuple[str, str]]] = None,
    ) -> None:
        if collisions is None:
            collisions = self.collisions
        if collided_pairs is None:
            collided_pairs = self._collided_pairs
        state.tick_ms += self.scenario.tick_ms
        dt_seconds = self.scenario.tick_ms / 1000.0

        # 1. 固定顺序推进所有参与者（车辆在前、行人在后）
        for actor in self.scenario.ordered_actors():
            speed = state.speeds[actor.actor_id]
            state.distances[actor.actor_id] += speed * dt_seconds

        # 1.5 参与者移动完成后做碰撞检测：车-行人距离进入阈值即记录一次
        self._detect_collisions(state, collisions, collided_pairs)

        # 2. 固定全序触发本 tick 的事件
        for event in self._events_at.get(state.tick_ms, ()):
            already_handled = event.event_id in fired
            # 副作用始终按时间轴应用：这样从过去继续推进时，
            # 已处理事件对状态的影响仍被确定性地重建。
            self._apply_action(state, event)
            if already_handled:
                continue  # 跳转后继续运行，不重复触发/登记
            fired.add(event.event_id)
            record = EventRecord(
                tick_ms=state.tick_ms,
                event_id=event.event_id,
                order_index=len(journal),
                action=dict(event.action),
            )
            journal.append(record)
            if new_records is not None:
                new_records.append(record)

    def _apply_action(self, state: EngineState, event: ScriptedEvent) -> None:
        action = event.action
        kind = action.get("type")
        if kind == "set_speed":
            if action["actor_id"] not in state.speeds:
                raise KeyError(f"unknown actor: {action['actor_id']}")
            state.speeds[action["actor_id"]] = float(action["speed"])
        elif kind == "stop_actor":
            if action["actor_id"] not in state.speeds:
                raise KeyError(f"unknown actor: {action['actor_id']}")
            state.speeds[action["actor_id"]] = 0.0
        elif kind == "set_signal":
            state.signal_overrides[action["signal_id"]] = action["phase"]
        elif kind == "set_path":
            actor_id = action["actor_id"]
            if actor_id not in state.speeds:
                raise KeyError(f"unknown actor: {actor_id}")
            state.paths[actor_id] = tuple(
                (float(x), float(y)) for x, y in action["path"]
            )
            # 从新折线起点继续，分支点以前的已行驶轨迹不受影响
            state.distances[actor_id] = 0.0
        elif kind == "set_signal_plan":
            signal_id = action["signal_id"]
            if signal_id not in self._signals_by_id:
                raise KeyError(f"unknown signal: {signal_id}")
            state.signal_plans[signal_id] = tuple(
                (int(tick), str(phase)) for tick, phase in action["phases"]
            )
            state.signal_overrides.pop(signal_id, None)
        elif kind == "pause":
            # 场景中断：事件副作用先生效，随后推进停止
            state.interrupted = True
        elif kind == "resume":
            state.interrupted = False
        else:
            raise ValueError(f"unknown event action type: {kind!r}")

    @staticmethod
    def _initial_state(scenario: Scenario) -> EngineState:
        state = EngineState()
        for actor in scenario.ordered_actors():
            state.distances[actor.actor_id] = 0.0
            state.speeds[actor.actor_id] = actor.speed
        return state

    def _detect_collisions(
        self,
        state: EngineState,
        collisions: List[CollisionRecord],
        collided_pairs: set[Tuple[str, str]],
    ) -> None:
        vehicles = self.scenario.vehicles
        pedestrians = self.scenario.pedestrians
        positions = {
            actor.actor_id: self._state_position(state, actor)
            for actor in self.scenario.ordered_actors()
        }
        for vehicle in vehicles:
            vpos = positions[vehicle.actor_id]
            for pedestrian in pedestrians:
                ppos = positions[pedestrian.actor_id]
                distance = (
                    (vpos[0] - ppos[0]) ** 2 + (vpos[1] - ppos[1]) ** 2
                ) ** 0.5
                pair = (vehicle.actor_id, pedestrian.actor_id)
                if distance <= self.collision_radius and pair not in collided_pairs:
                    collided_pairs.add(pair)
                    collisions.append(
                        CollisionRecord(
                            tick_ms=state.tick_ms,
                            vehicle_id=vehicle.actor_id,
                            pedestrian_id=pedestrian.actor_id,
                            vehicle_position=vpos,
                            pedestrian_position=ppos,
                            distance=distance,
                        )
                    )

    def _state_position(self, state: EngineState, actor: Actor) -> Vec2:
        path = state.paths.get(actor.actor_id, actor.path)
        return self._position_on_path(path, state.distances[actor.actor_id])

    @staticmethod
    def _position_on_path(path: Tuple[Vec2, ...], distance: float) -> Vec2:
        remaining = distance
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            seg_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
            if remaining <= seg_len:
                if seg_len == 0:
                    return path[i]
                ratio = remaining / seg_len
                return (x1 + (x2 - x1) * ratio, y1 + (y2 - y1) * ratio)
            remaining -= seg_len
        return path[-1]

    def _phase_from_plan(self, phases: Tuple[Tuple[int, str], ...]) -> str:
        current = phases[0][1]
        for start, phase in phases:
            if self.state.tick_ms >= start:
                current = phase
            else:
                break
        return current

    @property
    def _actors_by_id(self) -> Dict[str, Actor]:
        return {a.actor_id: a for a in self.scenario.ordered_actors()}

    @property
    def _signals_by_id(self) -> Dict[str, str]:
        return {s.signal_id: s for s in self.scenario.signals}

    @property
    def _events_at(self) -> Dict[int, Tuple[ScriptedEvent, ...]]:
        grouped: Dict[int, List[ScriptedEvent]] = {}
        for event in self.scenario.ordered_events():
            grouped.setdefault(event.tick_ms, []).append(event)
        return {tick: tuple(events) for tick, events in grouped.items()}
