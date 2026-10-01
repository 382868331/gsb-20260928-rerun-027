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

from dataclasses import dataclass, field, replace
import math
from typing import Dict, List, Optional, Tuple

from .scenario import Actor, Scenario, ScriptedEvent

Vec2 = Tuple[float, float]


@dataclass
class EngineState:
    tick_ms: int = 0
    distances: Dict[str, float] = field(default_factory=dict)
    speeds: Dict[str, float] = field(default_factory=dict)
    speed_targets: Dict[str, float] = field(default_factory=dict)
    signal_overrides: Dict[str, str] = field(default_factory=dict)
    reaction_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    interrupted: bool = False
    # 感知事实与待响应动作：与事件日志分离记录，随状态一起
    # 克隆/保存/重建，因此 seek 静默重放能确定性地还原。
    perceptions: List["PerceptionRecord"] = field(default_factory=list)
    responses: List["PendingResponse"] = field(default_factory=list)
    perceived_phases: Dict[str, Dict[str, str]] = field(default_factory=dict)
    exited: set = field(default_factory=set)

    def clone(self) -> "EngineState":
        return EngineState(
            tick_ms=self.tick_ms,
            distances=dict(self.distances),
            speeds=dict(self.speeds),
            speed_targets=dict(self.speed_targets),
            signal_overrides=dict(self.signal_overrides),
            reaction_params={k: dict(v) for k, v in self.reaction_params.items()},
            interrupted=self.interrupted,
            perceptions=list(self.perceptions),
            responses=[replace(r) for r in self.responses],
            perceived_phases={k: dict(v) for k, v in self.perceived_phases.items()},
            exited=set(self.exited),
        )


@dataclass(frozen=True)
class PerceptionRecord:
    """车辆看到的刺激事实；与实际响应动作分开记录。"""

    tick_ms: int
    vehicle_id: str
    stimulus: str  # "signal_phase" / "lead_brake"
    detail: str
    order_index: int  # 全局感知序号，同 tick 内反映固定处理顺序


@dataclass
class PendingResponse:
    """已感知、尚未到原定响应时刻的动作。

    status: pending（等待响应）/ superseded（被同车后续决定替代，不再执行）
    / executed / cancelled（车辆退出，不再执行）。
    """

    response_id: str
    vehicle_id: str
    perceive_tick_ms: int
    respond_tick_ms: int
    delay_ms: int
    action: dict
    stimulus: str
    detail: str
    status: str = "pending"


@dataclass(frozen=True)
class EventRecord:
    tick_ms: int
    event_id: str
    order_index: int  # 全局触发序号，同 tick 内反映固定顺序
    action: dict
    reason: Optional[str] = None


@dataclass(frozen=True)
class CollisionRecord:
    tick_ms: int
    actor_a: str
    actor_b: str
    distance: float
    position_a: Vec2
    position_b: Vec2
    reason: str

    @property
    def pair(self) -> Tuple[str, str]:
        return tuple(sorted((self.actor_a, self.actor_b)))


class ReplayEngine:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.state = self._initial_state(scenario)
        self.journal: List[EventRecord] = []
        self.collisions: List[CollisionRecord] = []
        self._fired: set[str] = set()
        self._collision_pairs: set[Tuple[str, str]] = set()
        self.paused = False

    # ---- 基础查询 ----

    @property
    def now_ms(self) -> int:
        return self.state.tick_ms

    def now_seconds(self) -> float:
        return self.state.tick_ms / 1000.0

    def position_of(self, actor_id: str) -> Vec2:
        actor = self._actors_by_id[actor_id]
        return actor.position_at(self.state.distances[actor_id])

    def positions(self) -> Dict[str, Vec2]:
        return {aid: self.position_of(aid) for aid in self._actors_by_id}

    def signal_phase(self, signal_id: str) -> str:
        return self._phase_at(self.state, signal_id)

    def signal_phases(self) -> Dict[str, str]:
        return {sid: self.signal_phase(sid) for sid in self._signals_by_id}

    def fired_event_ids(self) -> Tuple[str, ...]:
        return tuple(r.event_id for r in self.journal)

    def fired_reasons(self) -> Dict[str, Optional[str]]:
        """事件 ID -> 触发原因；同一事件只保留首次触发。"""
        return {r.event_id: r.reason for r in reversed(self.journal)}

    @property
    def perceptions(self) -> Tuple[PerceptionRecord, ...]:
        """已发生的感知事实（与实际响应动作分离记录）。"""
        return tuple(self.state.perceptions)

    @property
    def pending_responses(self) -> Tuple[PendingResponse, ...]:
        """仍在等待原定时刻响应的动作。"""
        return tuple(r for r in self.state.responses if r.status == "pending")

    @property
    def superseded_responses(self) -> Tuple[PendingResponse, ...]:
        """被同车后续决定替代、不再执行的待响应动作（保留记录）。"""
        return tuple(r for r in self.state.responses if r.status == "superseded")

    def collision_at_or_before(self, tick_ms: int) -> Tuple[CollisionRecord, ...]:
        return tuple(c for c in self.collisions if c.tick_ms <= tick_ms)

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
            self._step_once(
                self.state,
                self._fired,
                self.journal,
                new_records,
                self.collisions,
                self._collision_pairs,
            )
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
            self._step_once(
                self.state,
                self._fired,
                self.journal,
                new_records,
                self.collisions,
                self._collision_pairs,
            )
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
        shadow_collision_pairs = set(self._collision_pairs)
        start_tick_ms = self.state.tick_ms
        # 静默重放到目标时刻：旧事件的副作用照常应用以重建状态，
        # 但只有此前未处理过的事件才追加进日志
        while rebuilt.tick_ms < target_ms:
            self._step_once(
                rebuilt,
                shadow_fired,
                shadow_journal,
                None,
                shadow_collisions,
                shadow_collision_pairs,
            )
            if rebuilt.interrupted and start_tick_ms < rebuilt.tick_ms:
                break

        self.state = rebuilt
        self.journal = shadow_journal
        self.collisions = shadow_collisions
        self._fired = shadow_fired
        self._collision_pairs = shadow_collision_pairs
        self.paused = False

    # ---- 内部实现 ----

    def _step_once(
        self,
        state: EngineState,
        fired: set[str],
        journal: List[EventRecord],
        new_records: Optional[List[EventRecord]],
        collisions: Optional[List[CollisionRecord]] = None,
        collision_pairs: Optional[set[Tuple[str, str]]] = None,
    ) -> None:
        if collisions is None:
            collisions = self.collisions
        if collision_pairs is None:
            collision_pairs = self._collision_pairs
        state.tick_ms += self.scenario.tick_ms
        dt_seconds = self.scenario.tick_ms / 1000.0
        speeds_before = dict(state.speeds)

        # 1. 固定顺序推进所有参与者（车辆在前、行人在后）
        for actor in self.scenario.ordered_actors():
            if actor.actor_id in state.exited:
                continue  # 已退出场景的车辆不再推进
            target = state.speed_targets.get(actor.actor_id)
            if target is not None:
                self._approach_target(state, actor.actor_id, target)
            state.distances[actor.actor_id] += state.speeds[actor.actor_id] * dt_seconds

        self._detect_collisions(state, collisions, collision_pairs)

        # 2. 固定全序触发本 tick 的事件
        for event in self._events_at.get(state.tick_ms, ()):
            already_handled = event.event_id in fired
            # 副作用始终按时间轴应用：这样从过去继续推进时，
            # 已处理事件对状态的影响仍被确定性地重建。
            self._apply_action(state, event)
            if already_handled:
                continue  # 跳转后继续运行，不重复触发/登记
            fired.add(event.event_id)
            reason = event.action.get("reason")
            record = EventRecord(
                tick_ms=state.tick_ms,
                event_id=event.event_id,
                order_index=len(journal),
                action=dict(event.action),
                reason=reason,
            )
            journal.append(record)
            if new_records is not None:
                new_records.append(record)

        # 3. 先执行本 tick 到期的待响应动作：感知时刻+delay 的原定时刻
        #    优先于同 tick 的新感知（新决定只替代尚未到期的待响应）。
        self._execute_due_responses(state, fired, journal, new_records)

        # 4. 感知：车辆看到信号变化/前车制动，记录事实并安排延迟响应
        self._process_perceptions(state, speeds_before)

        # 5. 零延迟的响应（感知与响应同 tick）在此执行
        self._execute_due_responses(state, fired, journal, new_records)

    def _process_perceptions(self, state: EngineState, speeds_before: Dict[str, float]) -> None:
        """按固定顺序检测刺激并登记感知。

        顺序约定：按 scenario.reaction_schemes 的定义顺序逐车处理；
        同一车辆同刻的多个刺激先信号变化、后前车制动。新感知产生的
        决定会替代该车此前尚未执行的待响应动作（标记 superseded，
        记录保留但不再执行）。
        """
        for scheme in self.scenario.reaction_schemes:
            vehicle_id = scheme.vehicle_id
            if vehicle_id in state.exited:
                continue
            stimuli: List[Tuple[str, str]] = []
            if scheme.signal_id is not None:
                phase = self._phase_at(state, scheme.signal_id)
                seen = state.perceived_phases.setdefault(vehicle_id, {})
                last = seen.get(scheme.signal_id, phase)
                if phase != last:
                    stimuli.append(
                        ("signal_phase", f"信号 {scheme.signal_id} 由 {last} 变为 {phase}")
                    )
                seen[scheme.signal_id] = phase
            if scheme.lead_id is not None and scheme.lead_id not in state.exited:
                lead = scheme.lead_id
                if state.speeds.get(lead, 0.0) < speeds_before.get(lead, 0.0) - 1e-12:
                    stimuli.append(("lead_brake", f"前车 {lead} 制动"))
            for stimulus, detail in stimuli:
                self._record_perception(state, scheme, stimulus, detail)

    def _record_perception(
        self, state: EngineState, scheme, stimulus: str, detail: str
    ) -> None:
        vehicle_id = scheme.vehicle_id
        state.perceptions.append(
            PerceptionRecord(
                tick_ms=state.tick_ms,
                vehicle_id=vehicle_id,
                stimulus=stimulus,
                detail=detail,
                order_index=len(state.perceptions),
            )
        )
        # 同车此前尚未执行的待响应被后续决定替代
        for response in state.responses:
            if response.vehicle_id == vehicle_id and response.status == "pending":
                response.status = "superseded"
        response_id = f"react:{vehicle_id}:{state.tick_ms}:{len(state.responses)}"
        action = dict(scheme.action)
        action.setdefault("actor_id", vehicle_id)
        state.responses.append(
            PendingResponse(
                response_id=response_id,
                vehicle_id=vehicle_id,
                perceive_tick_ms=state.tick_ms,
                respond_tick_ms=state.tick_ms + int(scheme.delay_ms),
                delay_ms=int(scheme.delay_ms),
                action=action,
                stimulus=stimulus,
                detail=detail,
            )
        )

    def _execute_due_responses(
        self,
        state: EngineState,
        fired: set[str],
        journal: List[EventRecord],
        new_records: Optional[List[EventRecord]],
    ) -> None:
        for response in state.responses:
            if response.status != "pending" or response.respond_tick_ms != state.tick_ms:
                continue
            if response.vehicle_id in state.exited:
                response.status = "cancelled"  # 车辆已退出：不再执行
                continue
            response.status = "executed"
            action = dict(response.action)
            action.setdefault(
                "reason",
                f"{response.detail}，延迟 {response.delay_ms}ms 后响应",
            )
            pseudo = ScriptedEvent(response.response_id, state.tick_ms, action)
            # 与脚本事件一致：重放时副作用始终应用，但不重复登记
            self._apply_action(state, pseudo)
            if response.response_id in fired:
                continue
            fired.add(response.response_id)
            record = EventRecord(
                tick_ms=state.tick_ms,
                event_id=response.response_id,
                order_index=len(journal),
                action=action,
                reason=action.get("reason"),
            )
            journal.append(record)
            if new_records is not None:
                new_records.append(record)

    def _detect_collisions(
        self,
        state: EngineState,
        collisions: List[CollisionRecord],
        collision_pairs: set[Tuple[str, str]],
    ) -> None:
        actors = self.scenario.ordered_actors()
        actors = [a for a in actors if a.actor_id not in state.exited]
        positions = {a.actor_id: a.position_at(state.distances[a.actor_id]) for a in actors}
        for index, actor_a in enumerate(actors):
            for actor_b in actors[index + 1 :]:
                pair = tuple(sorted((actor_a.actor_id, actor_b.actor_id)))
                if pair in collision_pairs:
                    continue
                point_a = positions[actor_a.actor_id]
                point_b = positions[actor_b.actor_id]
                distance = math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])
                if distance <= actor_a.radius + actor_b.radius:
                    pair = tuple(pair)
                    collision_pairs.add(pair)
                    collisions.append(
                        CollisionRecord(
                            tick_ms=state.tick_ms,
                            actor_a=pair[0],
                            actor_b=pair[1],
                            distance=distance,
                            position_a=positions[pair[0]],
                            position_b=positions[pair[1]],
                            reason=f"{pair[0]} 与 {pair[1]} 距离小于碰撞半径之和",
                        )
                    )

    def _apply_action(self, state: EngineState, event: ScriptedEvent) -> None:
        action = event.action
        kind = action.get("type")
        if kind == "set_speed":
            if action["actor_id"] not in state.speeds:
                raise KeyError(f"unknown actor: {action['actor_id']}")
            self._change_speed(
                state,
                action["actor_id"],
                float(action["speed"]),
                rate=action.get("rate"),
            )
        elif kind == "stop_actor":
            if action["actor_id"] not in state.speeds:
                raise KeyError(f"unknown actor: {action['actor_id']}")
            self._change_speed(state, action["actor_id"], 0.0, rate=action.get("rate"))
        elif kind == "set_signal":
            state.signal_overrides[action["signal_id"]] = action["phase"]
        elif kind == "set_reaction_params":
            actor_id = action["actor_id"]
            if actor_id not in state.reaction_params:
                state.reaction_params[actor_id] = {}
            state.reaction_params[actor_id].update(
                {str(k): float(v) for k, v in action.get("params", {}).items()}
            )
        elif kind == "pause":
            # 场景中断：事件副作用先生效，随后推进停止
            state.interrupted = True
        elif kind == "resume":
            state.interrupted = False
        elif kind == "exit_actor":
            # 车辆退出场景：不再推进/参与碰撞，待响应动作取消且不再执行
            actor_id = action["actor_id"]
            if actor_id not in state.speeds:
                raise KeyError(f"unknown actor: {actor_id}")
            state.exited.add(actor_id)
            state.speed_targets.pop(actor_id, None)
            for response in state.responses:
                if response.vehicle_id == actor_id and response.status == "pending":
                    response.status = "cancelled"
        else:
            raise ValueError(f"unknown event action type: {kind!r}")

    def _change_speed(
        self,
        state: EngineState,
        actor_id: str,
        target_speed: float,
        rate: Optional[float] = None,
    ) -> None:
        """按车辆反应参数限速变化；未配置参数时保持原有瞬时反应。"""
        current = state.speeds[actor_id]
        if rate is None:
            params = state.reaction_params.get(actor_id, {})
            if target_speed < current and "brake_rate" in params:
                state.speed_targets[actor_id] = target_speed
                self._approach_target(state, actor_id, target_speed)
                return
            elif target_speed > current and "acceleration" in params:
                state.speed_targets[actor_id] = target_speed
                self._approach_target(state, actor_id, target_speed)
                return
        state.speed_targets.pop(actor_id, None)
        state.speeds[actor_id] = float(target_speed)

    def _approach_target(self, state: EngineState, actor_id: str, target_speed: float) -> None:
        current = state.speeds[actor_id]
        params = state.reaction_params.get(actor_id, {})
        rate_key = "brake_rate" if target_speed < current else "acceleration"
        rate = params.get(rate_key)
        dt_seconds = self.scenario.tick_ms / 1000.0
        if rate is None:
            state.speeds[actor_id] = float(target_speed)
            return
        max_delta = float(rate) * dt_seconds
        delta = target_speed - current
        state.speeds[actor_id] = float(
            target_speed if abs(delta) <= max_delta else current + math.copysign(max_delta, delta)
        )

    @staticmethod
    def _initial_state(scenario: Scenario) -> EngineState:
        state = EngineState()
        for actor in scenario.ordered_actors():
            state.distances[actor.actor_id] = 0.0
            state.speeds[actor.actor_id] = actor.speed
        for actor_id, params in getattr(scenario, "reaction_params", {}).items():
            state.reaction_params[actor_id] = {k: float(v) for k, v in params.items()}
        signals = {s.signal_id: s for s in scenario.signals}
        for scheme in getattr(scenario, "reaction_schemes", ()):
            if scheme.signal_id is not None:
                seen = state.perceived_phases.setdefault(scheme.vehicle_id, {})
                seen[scheme.signal_id] = signals[scheme.signal_id].phase_at(0)
        return state

    def _phase_at(self, state: EngineState, signal_id: str) -> str:
        if signal_id in state.signal_overrides:
            return state.signal_overrides[signal_id]
        return self._signals_by_id[signal_id].phase_at(state.tick_ms)

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
