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
import math
from typing import Dict, List, Optional, Tuple

from .scenario import Actor, Scenario, ScriptedEvent, Vehicle

Vec2 = Tuple[float, float]


@dataclass
class EngineState:
    tick_ms: int = 0
    distances: Dict[str, float] = field(default_factory=dict)
    speeds: Dict[str, float] = field(default_factory=dict)
    speed_targets: Dict[str, float] = field(default_factory=dict)
    signal_overrides: Dict[str, str] = field(default_factory=dict)
    reaction_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    reaction_delays: Dict[str, Dict[str, float]] = field(default_factory=dict)
    observed_signal_phases: Dict[str, str] = field(default_factory=dict)
    observed_speeds: Dict[str, float] = field(default_factory=dict)
    pending_responses: List["ResponseRecord"] = field(default_factory=list)
    active_actors: Dict[str, bool] = field(default_factory=dict)
    interrupted: bool = False

    def clone(self) -> "EngineState":
        return EngineState(
            tick_ms=self.tick_ms,
            distances=dict(self.distances),
            speeds=dict(self.speeds),
            speed_targets=dict(self.speed_targets),
            signal_overrides=dict(self.signal_overrides),
            reaction_params={k: dict(v) for k, v in self.reaction_params.items()},
            reaction_delays={k: dict(v) for k, v in self.reaction_delays.items()},
            observed_signal_phases=dict(self.observed_signal_phases),
            observed_speeds=dict(self.observed_speeds),
            pending_responses=[r.clone() for r in self.pending_responses],
            active_actors=dict(self.active_actors),
            interrupted=self.interrupted,
        )


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


@dataclass
class PerceptionRecord:
    """车辆在某一 tick 确认看到的外部事实；事实不等同于已经做出动作。"""

    perception_id: str
    tick_ms: int
    vehicle_id: str
    stimulus: str
    source_id: str
    observed: dict
    response_id: str
    order_index: int
    seq: int = 0
    supersedes_response_id: Optional[str] = None


@dataclass
class ResponseRecord:
    """由感知事实派生出的实际车辆动作。"""

    response_id: str
    vehicle_id: str
    stimulus: str
    source_id: str
    trigger_tick_ms: int
    due_tick_ms: int
    action: dict
    status: str = "scheduled"  # scheduled / executed / superseded / cancelled
    order_index: int = 0
    perception_id: Optional[str] = None
    superseded_by: Optional[str] = None
    reason: Optional[str] = None

    def clone(self) -> "ResponseRecord":
        return ResponseRecord(
            response_id=self.response_id,
            vehicle_id=self.vehicle_id,
            stimulus=self.stimulus,
            source_id=self.source_id,
            trigger_tick_ms=self.trigger_tick_ms,
            due_tick_ms=self.due_tick_ms,
            action=dict(self.action),
            status=self.status,
            order_index=self.order_index,
            perception_id=self.perception_id,
            superseded_by=self.superseded_by,
            reason=self.reason,
        )


class ReplayEngine:
    def __init__(self, scenario: Scenario):
        self.scenario = scenario
        self.state = self._initial_state(scenario)
        self.journal: List[EventRecord] = []
        self.collisions: List[CollisionRecord] = []
        self.perceptions: List[PerceptionRecord] = []
        self.responses: List[ResponseRecord] = []
        self._fired: set[str] = set()
        self._handled_perceptions: set[str] = set()
        self._handled_responses: set[str] = set()
        self._collision_pairs: set[Tuple[str, str]] = set()
        self.paused = False
        # 已保存但尚未由用户确认重派生的输入修订（旧版单候选接口）。
        # 这里只保存不透明数据；重派生前不得改变场景事件或反应参数。
        self.pending_revision: Optional[dict] = None
        # 同一交通方案的多份未确认感知修订候选：candidate_id -> 不透明修订数据。
        # 多候选互不覆盖；派生其中一份不删除其他候选，也不改写原方案事实。
        self.pending_revisions: Dict[str, dict] = {}
        # 当前选择（已派生确认）的候选；撤销一次选择会回到上一个选择。
        self.selected_revision_id: Optional[str] = None
        self.revision_selection_history: List[Optional[str]] = []
        # 已派生确认的候选修订数据：撤销选择时据此把候选退回未决集合，
        # 原方案事实与其他候选不受影响。
        self.adopted_revisions: Dict[str, dict] = {}
        # 多候选组成的联合解释。联合提案在确认前不得派生交通方案。
        self.pending_joint_proposals: Dict[str, dict] = {}
        self.adopted_joint_proposals: Dict[str, dict] = {}
        self.selected_joint_proposal_id: Optional[str] = None
        self.joint_selection_history: List[Optional[str]] = []

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

    def signal_phase(self, signal_id: str, state: Optional[EngineState] = None) -> str:
        state = self.state if state is None else state
        if signal_id in state.signal_overrides:
            return state.signal_overrides[signal_id]
        sig = self._signals_by_id[signal_id]
        return sig.phase_at(state.tick_ms)

    def signal_phases(self) -> Dict[str, str]:
        return {sid: self.signal_phase(sid) for sid in self._signals_by_id}

    def fired_event_ids(self) -> Tuple[str, ...]:
        return tuple(r.event_id for r in self.journal)

    def fired_reasons(self) -> Dict[str, Optional[str]]:
        """事件 ID -> 触发原因；同一事件只保留首次触发。"""
        return {r.event_id: r.reason for r in reversed(self.journal)}

    def perception_records(self) -> Tuple[PerceptionRecord, ...]:
        return tuple(record for record in self.perceptions if record.tick_ms <= self.now_ms)

    def response_records(self) -> Tuple[ResponseRecord, ...]:
        return tuple(record for record in self.responses if record.trigger_tick_ms <= self.now_ms)

    def pending_response_records(self) -> Tuple[ResponseRecord, ...]:
        return tuple(r for r in self.state.pending_responses if r.status == "scheduled")

    def first_response(
        self, vehicle_id: Optional[str] = None, stimulus: Optional[str] = None
    ) -> Optional[ResponseRecord]:
        for response in self.response_records():
            if vehicle_id is not None and response.vehicle_id != vehicle_id:
                continue
            if stimulus is not None and response.stimulus != stimulus:
                continue
            return response
        return None

    def minimum_gap(
        self, actor_a: Optional[str] = None, actor_b: Optional[str] = None
    ) -> Optional[float]:
        """返回当前 tick 两个尚在场景内参与者的边缘间距；缺省取最小车辆间距。"""
        actors = [a for a in self.scenario.ordered_actors() if self.state.active_actors.get(a.actor_id, True)]
        if actor_a is not None or actor_b is not None:
            if actor_a is None or actor_b is None:
                raise ValueError("actor_a and actor_b must be provided together")
            point_a = self.position_of(actor_a)
            point_b = self.position_of(actor_b)
            center_gap = math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])
            actor_a_record = self._actors_by_id[actor_a]
            actor_b_record = self._actors_by_id[actor_b]
            return max(0.0, center_gap - actor_a_record.radius - actor_b_record.radius)
        vehicles = [a for a in actors if isinstance(a, Vehicle)]
        gaps = [
            max(
                0.0,
                math.dist(self.position_of(a.actor_id), self.position_of(b.actor_id))
                - a.radius
                - b.radius,
            )
            for index, a in enumerate(vehicles)
            for b in vehicles[index + 1 :]
        ]
        return min(gaps) if gaps else None

    def minimum_gap_between(self, until_ms: int, actor_a: str, actor_b: str) -> Optional[float]:
        """重放至 until_ms，返回期间记录到的最小边缘间距。"""
        tick = self.scenario.tick_ms
        gaps = []
        for current in range(0, until_ms + 1, tick):
            self.seek(current)
            gaps.append(self.minimum_gap(actor_a, actor_b))
        return min(gaps) if gaps else None

    def conflict_event_chain(
        self, conflict: Optional[CollisionRecord] = None, max_steps: int = 20
    ) -> Tuple[object, ...]:
        """返回首次冲突发生前（含冲突 tick）的感知、响应与脚本事件链。"""
        if conflict is None:
            if not self.collisions:
                return ()
            conflict = self.collisions[0]
        actors = {conflict.actor_a, conflict.actor_b}
        chain: List[object] = []
        for record in self.perception_records():
            if record.tick_ms <= conflict.tick_ms and record.vehicle_id in actors:
                chain.append(record)
        for record in self.response_records():
            if record.due_tick_ms <= conflict.tick_ms and record.vehicle_id in actors:
                chain.append(record)
        for record in self.journal:
            if record.tick_ms <= conflict.tick_ms:
                action_actor = record.action.get("actor_id")
                if action_actor in actors or record.action.get("type") in {"pause", "resume"}:
                    chain.append(record)
        chain.sort(key=_event_chain_order)
        return tuple(chain[-max_steps:])

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
        shadow_perceptions = list(self.perceptions)
        shadow_responses = list(self.responses)
        shadow_handled_perceptions = set(self._handled_perceptions)
        shadow_handled_responses = set(self._handled_responses)
        rebuilt_perceptions: List[PerceptionRecord] = []
        rebuilt_responses: List[ResponseRecord] = []
        rebuilt_handled_perceptions = set(self._handled_perceptions)
        rebuilt_handled_responses = set(self._handled_responses)
        rebuilt_journal: List[EventRecord] = []
        known_perceptions = set(self._handled_perceptions)
        known_responses = set(self._handled_responses)
        known_events = set(self._fired)
        start_tick_ms = self.state.tick_ms
        # 静默重放到目标时刻：旧事件的副作用照常应用以重建状态，
        # 但只有此前未处理过的事件才追加进日志
        while rebuilt.tick_ms < target_ms:
            self._step_once(
                rebuilt,
                shadow_fired,
                rebuilt_journal,
                None,
                shadow_collisions,
                shadow_collision_pairs,
                rebuilt_perceptions,
                rebuilt_responses,
                rebuilt_handled_perceptions,
                rebuilt_handled_responses,
            )
            if rebuilt.interrupted and start_tick_ms < rebuilt.tick_ms:
                break

        self.state = rebuilt
        new_journal = [record for record in rebuilt_journal if record.event_id not in known_events]
        self.journal = shadow_journal + new_journal
        self.journal.sort(key=lambda record: (record.tick_ms, record.order_index, record.event_id))
        self.journal = [
            EventRecord(
                tick_ms=record.tick_ms,
                event_id=record.event_id,
                order_index=index,
                action=record.action,
                reason=record.reason,
            )
            for index, record in enumerate(self.journal)
        ]
        self.collisions = shadow_collisions
        self.perceptions = shadow_perceptions + [
            record
            for record in rebuilt_perceptions
            if record.perception_id not in known_perceptions
        ]
        self.responses = shadow_responses + [
            record
            for record in rebuilt_responses
            if record.response_id not in known_responses
        ]
        self._fired = shadow_fired
        self._handled_perceptions = shadow_handled_perceptions | {
            record.perception_id for record in rebuilt_perceptions
        }
        self._handled_responses = shadow_handled_responses | {
            record.response_id for record in rebuilt_responses
        }
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
        perceptions: Optional[List[PerceptionRecord]] = None,
        responses: Optional[List[ResponseRecord]] = None,
        handled_perceptions: Optional[set[str]] = None,
        handled_responses: Optional[set[str]] = None,
    ) -> None:
        if collisions is None:
            collisions = self.collisions
        if collision_pairs is None:
            collision_pairs = self._collision_pairs
        if perceptions is None:
            perceptions = self.perceptions
        if responses is None:
            responses = self.responses
        if handled_perceptions is None:
            handled_perceptions = self._handled_perceptions
        if handled_responses is None:
            handled_responses = self._handled_responses
        state.tick_ms += self.scenario.tick_ms
        dt_seconds = self.scenario.tick_ms / 1000.0

        self._execute_due_responses(
            state, responses, handled_responses, new_records, journal
        )

        # 1. 固定顺序推进所有参与者（车辆在前、行人在后）
        for actor in self.scenario.ordered_actors():
            if not state.active_actors.get(actor.actor_id, True):
                state.speeds[actor.actor_id] = 0.0
                state.speed_targets.pop(actor.actor_id, None)
                continue
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
            if event.action.get("type") == "inject_perception":
                self._apply_injected_perception(
                    state,
                    event,
                    perceptions,
                    responses,
                    handled_perceptions,
                    handled_responses,
                )
            else:
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

        self._perceive(state, perceptions, responses, handled_perceptions, handled_responses)
        self._record_cancelled_responses(state, responses, handled_responses)
        state.pending_responses = [
            response for response in state.pending_responses if response.status == "scheduled"
        ]

    @staticmethod
    def _record_cancelled_responses(
        state: EngineState,
        responses: List[ResponseRecord],
        handled_responses: set[str],
    ) -> None:
        for response in state.pending_responses:
            if response.status != "cancelled":
                continue
            for existing in responses:
                if existing.response_id == response.response_id:
                    existing.status = "cancelled"
                    existing.reason = response.reason
                    handled_responses.add(response.response_id)
                    break
            else:
                handled_responses.add(response.response_id)
                responses.append(response.clone())

    def _detect_collisions(
        self,
        state: EngineState,
        collisions: List[CollisionRecord],
        collision_pairs: set[Tuple[str, str]],
    ) -> None:
        actors = [
            actor
            for actor in self.scenario.ordered_actors()
            if state.active_actors.get(actor.actor_id, True)
        ]
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

    def _execute_due_responses(
        self,
        state: EngineState,
        responses: List[ResponseRecord],
        handled_responses: set[str],
        new_records: Optional[List[EventRecord]],
        journal: List[EventRecord],
    ) -> None:
        due = [
            response
            for response in sorted(
                state.pending_responses,
                key=lambda r: (r.due_tick_ms, r.order_index, r.response_id),
            )
            if response.status == "scheduled"
            and response.due_tick_ms <= state.tick_ms + self.scenario.tick_ms
        ]
        for response in due:
            response.status = "executed"
            self._apply_response_action(state, response)
            if response.response_id not in handled_responses:
                handled_responses.add(response.response_id)
                stored = response.clone()
                stored.status = "executed"
                responses.append(stored)
                record = EventRecord(
                    tick_ms=state.tick_ms,
                    event_id=response.response_id,
                    order_index=len(journal),
                    action=dict(response.action),
                    reason=response.reason,
                )
                journal.append(record)
                if new_records is not None:
                    new_records.append(record)
        state.pending_responses = [
            response for response in state.pending_responses if response.status == "scheduled"
        ]

    def _apply_response_action(self, state: EngineState, response: ResponseRecord) -> None:
        if not state.active_actors.get(response.vehicle_id, True):
            response.status = "cancelled"
            response.reason = f"vehicle exited; {response.reason or ''}".strip("; ")
            return
        action = response.action
        kind = action.get("type")
        if kind in {"set_speed", "stop_actor"}:
            target_speed = float(action.get("speed", 0.0))
            self._change_speed(
                state,
                response.vehicle_id,
                target_speed,
                rate=action.get("rate"),
                immediate=False,
            )
        elif kind == "exit_actor":
            state.active_actors[response.vehicle_id] = False
            state.speeds[response.vehicle_id] = 0.0
            self._cancel_pending_for_actor(
                state, response.vehicle_id, "vehicle exited"
            )
        else:
            raise ValueError(f"unknown response action type: {kind!r}")

    def _apply_injected_perception(
        self,
        state: EngineState,
        event: ScriptedEvent,
        perceptions: List[PerceptionRecord],
        responses: List[ResponseRecord],
        handled_perceptions: set[str],
        handled_responses: set[str],
    ) -> None:
        """派生方案注入的感知事实：在事件 tick 采用该感知时刻与延迟。

        与自动感知走同一套 ResponseRecord 调度，因此跳转重放、保存恢复
        与再次分支的行为完全一致；记录通过 handled 集合去重。
        """
        action = event.action
        vehicle_id = action["vehicle_id"]
        stimulus = action["stimulus"]
        source_id = action.get("source_id", "")
        delay = action.get("delay_ms")
        perception_id = f"injected-perception:{event.event_id}"
        response_id = f"injected-response:{event.event_id}"
        observed = dict(action.get("observed", {}))
        observed["injected"] = True
        observed["delay_ms"] = delay
        superseded = None
        if delay is None or not state.active_actors.get(vehicle_id, True):
            reason = action.get("reason")
            if not state.active_actors.get(vehicle_id, True):
                reason = f"vehicle exited; {reason or ''}".strip("; ")
            self._record_ignored_response(
                responses=responses,
                handled_responses=handled_responses,
                response_id=response_id,
                vehicle_id=vehicle_id,
                stimulus=stimulus,
                source_id=source_id,
                trigger_tick=state.tick_ms,
                perception_id=perception_id,
                reason=reason,
            )
        else:
            superseded = self._schedule_response(
                state=state,
                responses=responses,
                handled_responses=handled_responses,
                vehicle_id=vehicle_id,
                stimulus=stimulus,
                source_id=source_id,
                trigger_tick=state.tick_ms,
                delay_ms=int(round(delay)),
                response_id=response_id,
                perception_id=perception_id,
                action_override=action.get("response_action"),
                reason_override=action.get("reason"),
            )
        if perception_id not in handled_perceptions:
            handled_perceptions.add(perception_id)
            perceptions.append(
                PerceptionRecord(
                    perception_id=perception_id,
                    tick_ms=state.tick_ms,
                    vehicle_id=vehicle_id,
                    stimulus=stimulus,
                    source_id=source_id,
                    observed=observed,
                    response_id=response_id,
                    order_index=len(perceptions),
                    seq=0,
                    supersedes_response_id=superseded,
                )
            )

    def _perceive(
        self,
        state: EngineState,
        perceptions: List[PerceptionRecord],
        responses: List[ResponseRecord],
        handled_perceptions: set[str],
        handled_responses: set[str],
    ) -> None:
        stimuli: List[tuple] = []
        for signal in self.scenario.signals:
            phase = self.signal_phase(signal.signal_id, state)
            previous = state.observed_signal_phases.get(signal.signal_id)
            if previous is not None and phase != previous:
                for vehicle in self.scenario.vehicles:
                    if self._vehicle_watching_signal(state, vehicle.actor_id, signal.signal_id):
                        stimuli.append(
                            ("signal", signal.signal_id, previous, phase, vehicle.actor_id)
                        )

        current_speeds = dict(state.speeds)
        for follower in self.scenario.vehicles:
            if not state.active_actors.get(follower.actor_id, True):
                continue
            leader = self._leader_for(state, follower.actor_id)
            if leader is None:
                continue
            current = current_speeds[leader.actor_id]
            previous = state.observed_speeds.get(leader.actor_id)
            if previous is not None and current < previous - 1e-9:
                if self._reaction_delay(state, follower.actor_id, "lead_brake") is not None:
                    stimuli.append(
                        (
                            "lead_brake",
                            leader.actor_id,
                            previous,
                            current,
                            follower.actor_id,
                        )
                    )

        stimuli.sort(key=lambda item: (0 if item[0] == "signal" else 1, item[1], item[4], str(item)))
        order = len(perceptions)
        sequence = 0
        for stimulus in stimuli:
            kind, source_id = stimulus[0], stimulus[1]
            if kind == "signal":
                vehicle_id = stimulus[4]
                stimulus_key = self._signal_stimulus_key(stimulus[3])
            else:
                vehicle_id = stimulus[4]
                stimulus_key = "lead_brake"
            if vehicle_id is None:
                continue
            delay = self._reaction_delay(state, vehicle_id, stimulus_key)
            perception_id = (
                f"perception:{state.tick_ms}:{vehicle_id}:{kind}:{source_id}:{sequence}"
            )
            response_id = (
                f"response:{state.tick_ms}:{vehicle_id}:{kind}:{source_id}:{sequence}"
            )
            sequence += 1
            observed = self._observed_stimulus(stimulus)
            superseded = None
            if delay is None:
                self._record_ignored_response(
                    responses=responses,
                    handled_responses=handled_responses,
                    response_id=response_id,
                    vehicle_id=vehicle_id,
                    stimulus=stimulus_key,
                    source_id=source_id,
                    trigger_tick=state.tick_ms,
                    perception_id=perception_id,
                )
            else:
                superseded = self._schedule_response(
                    state=state,
                    responses=responses,
                    handled_responses=handled_responses,
                    vehicle_id=vehicle_id,
                    stimulus=stimulus_key,
                    source_id=source_id,
                    trigger_tick=state.tick_ms,
                    delay_ms=int(round(delay)),
                    response_id=response_id,
                    perception_id=perception_id,
                )
            if perception_id not in handled_perceptions:
                handled_perceptions.add(perception_id)
                perceptions.append(
                    PerceptionRecord(
                        perception_id=perception_id,
                        tick_ms=state.tick_ms,
                        vehicle_id=vehicle_id,
                        stimulus=stimulus_key,
                        source_id=source_id,
                        observed=observed,
                        response_id=response_id,
                        order_index=order,
                        seq=sequence - 1,
                        supersedes_response_id=superseded,
                    )
                )
                order += 1
        state.observed_speeds.update(current_speeds)
        for signal in self.scenario.signals:
            state.observed_signal_phases[signal.signal_id] = self.signal_phase(
                signal.signal_id, state
            )

    def _vehicle_watching_signal(
        self, state: EngineState, vehicle_id: str, signal_id: str
    ) -> bool:
        configured = state.reaction_delays.get(vehicle_id, {})
        return any(
            key in configured
            for key in (
                "signal_change",
                "signal_red",
                "signal_yellow",
                "signal_green",
                f"signal_{signal_id}",
                f"signal_{signal_id}_red",
                f"signal_{signal_id}_yellow",
                f"signal_{signal_id}_green",
            )
        )

    @staticmethod
    def _signal_stimulus_key(phase: str) -> str:
        return f"signal_{phase}"

    @staticmethod
    def _reaction_delay(state: EngineState, vehicle_id: str, stimulus: str) -> Optional[float]:
        delays = state.reaction_delays.get(vehicle_id, {})
        if stimulus in delays:
            return delays[stimulus]
        if stimulus.startswith("signal_") and "signal_change" in delays:
            return delays["signal_change"]
        return None

    def _schedule_response(
        self,
        state: EngineState,
        responses: List[ResponseRecord],
        handled_responses: set[str],
        vehicle_id: str,
        stimulus: str,
        source_id: str,
        trigger_tick: int,
        delay_ms: int,
        response_id: str,
        perception_id: str,
        action_override: Optional[dict] = None,
        reason_override: Optional[str] = None,
    ) -> Optional[str]:
        if delay_ms % self.scenario.tick_ms != 0:
            raise ValueError("reaction delay must align with scenario tick_ms")
        if delay_ms < 0:
            raise ValueError("reaction delay must be non-negative")
        if action_override is None:
            action = {
                "type": "stop_actor",
                "actor_id": vehicle_id,
            }
        else:
            action = dict(action_override)
            action.setdefault("actor_id", vehicle_id)
        reason = reason_override or (
            f"{vehicle_id} 看到 {source_id} 的 {stimulus} 后延迟 {delay_ms}ms 制动"
        )
        superseded_id = None
        for pending in state.pending_responses:
            if (
                pending.status == "scheduled"
                and pending.vehicle_id == vehicle_id
                and pending.action.get("type") in {"set_speed", "stop_actor"}
            ):
                pending.status = "superseded"
                pending.superseded_by = response_id
                superseded_id = pending.response_id
                self._mark_stored_response(
                    responses, handled_responses, pending
                )
        response = ResponseRecord(
            response_id=response_id,
            vehicle_id=vehicle_id,
            stimulus=stimulus,
            source_id=source_id,
            trigger_tick_ms=trigger_tick,
            due_tick_ms=trigger_tick + delay_ms,
            action=action,
            order_index=len(responses),
            perception_id=perception_id,
            reason=reason,
        )
        state.pending_responses.append(response)
        return superseded_id

    @staticmethod
    def _record_ignored_response(
        responses: List[ResponseRecord],
        handled_responses: set[str],
        response_id: str,
        vehicle_id: str,
        stimulus: str,
        source_id: str,
        trigger_tick: int,
        perception_id: str,
        reason: Optional[str] = None,
    ) -> None:
        if response_id in handled_responses:
            return
        handled_responses.add(response_id)
        responses.append(
            ResponseRecord(
                response_id=response_id,
                vehicle_id=vehicle_id,
                stimulus=stimulus,
                source_id=source_id,
                trigger_tick_ms=trigger_tick,
                due_tick_ms=trigger_tick,
                action={"type": "none"},
                status="ignored",
                order_index=len(responses),
                perception_id=perception_id,
                reason=reason
                or f"{vehicle_id} 看到 {source_id} 的 {stimulus}，该方案未配置响应延迟",
            )
        )

    @staticmethod
    def _mark_stored_response(
        responses: List[ResponseRecord],
        handled_responses: set[str],
        pending: ResponseRecord,
    ) -> None:
        for existing in responses:
            if existing.response_id == pending.response_id:
                existing.status = "superseded"
                existing.superseded_by = pending.superseded_by
                return
        stored = pending.clone()
        stored.status = "superseded"
        responses.append(stored)

    @staticmethod
    def _observed_stimulus(stimulus: tuple) -> dict:
        if stimulus[0] == "signal":
            return {"previous_phase": stimulus[2], "phase": stimulus[3]}
        return {"previous_speed": stimulus[2], "speed": stimulus[3]}

    def _leader_for(self, state: EngineState, follower_id: str):
        follower = self._actors_by_id[follower_id]
        follower_pos = follower.position_at(state.distances[follower_id])
        best = None
        best_distance = None
        for leader in self.scenario.vehicles:
            if (
                leader.actor_id == follower_id
                or not state.active_actors.get(leader.actor_id, True)
            ):
                continue
            leader_pos = leader.position_at(state.distances[leader.actor_id])
            longitudinal = leader_pos[0] - follower_pos[0]
            lateral = abs(leader_pos[1] - follower_pos[1])
            if longitudinal > 1e-9 and lateral <= follower.radius + leader.radius + 1.0:
                gap = math.hypot(longitudinal, lateral)
                if best_distance is None or gap < best_distance:
                    best = leader
                    best_distance = gap
        return best

    @staticmethod
    def _cancel_pending_for_actor(
        state: EngineState, actor_id: str, reason: str
    ) -> None:
        for response in state.pending_responses:
            if response.vehicle_id == actor_id and response.status == "scheduled":
                response.status = "cancelled"
                response.reason = f"{reason}; {response.reason or ''}".strip("; ")

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
        elif kind == "exit_actor":
            state.active_actors[action["actor_id"]] = False
            state.speeds[action["actor_id"]] = 0.0
            state.speed_targets.pop(action["actor_id"], None)
            self._cancel_pending_for_actor(state, action["actor_id"], "vehicle exited")
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
        else:
            raise ValueError(f"unknown event action type: {kind!r}")

    def _change_speed(
        self,
        state: EngineState,
        actor_id: str,
        target_speed: float,
        rate: Optional[float] = None,
        immediate: bool = True,
    ) -> None:
        """按车辆反应参数限速变化；未配置参数时保持原有瞬时反应。"""
        current = state.speeds[actor_id]
        if rate is None:
            params = state.reaction_params.get(actor_id, {})
            if target_speed < current and "brake_rate" in params:
                state.speed_targets[actor_id] = target_speed
                if immediate:
                    self._approach_target(state, actor_id, target_speed)
                return
            elif target_speed > current and "acceleration" in params:
                state.speed_targets[actor_id] = target_speed
                if immediate:
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
        for actor_id, delays in getattr(scenario, "reaction_delays", {}).items():
            state.reaction_delays[actor_id] = {k: float(v) for k, v in delays.items()}
        for actor in scenario.ordered_actors():
            state.active_actors[actor.actor_id] = True
        state.observed_signal_phases.update(
            {signal.signal_id: signal.phase_at(0) for signal in scenario.signals}
        )
        state.observed_speeds.update(
            {actor.actor_id: actor.speed for actor in scenario.ordered_actors()}
        )
        return state

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


def _event_chain_order(record) -> tuple:
    if isinstance(record, PerceptionRecord):
        return (record.tick_ms, 0, record.order_index, record.perception_id)
    if isinstance(record, ResponseRecord):
        return (record.due_tick_ms, 2, record.order_index, record.response_id)
    return (record.tick_ms, 1, record.order_index, record.event_id)
