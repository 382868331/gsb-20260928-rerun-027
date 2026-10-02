"""不确定感知时刻的范围派生与离散对照。

场景记录有时只能确定"驾驶员在 [earliest_ms, latest_ms] 内看到了刺激"，
无法给出唯一感知时刻。本模块把这种不确定感知作为分支输入：

* PerceptionWindow 描述一个刺激的可能感知范围与该方案采用的反应延迟；
* WindowConstraint 描述同一车辆多个感知范围之间的约束
  （"after"：后一个观察不早于前一个；"same"：两条记录来自同一次观察）；
* enumerate_assignments 枚举所有满足约束的感知时刻组合，
  derive_schemes 为每个组合派生一个自包含分支方案，
  每个方案的感知事实通过 inject_perception 事件写入，确定可重放；
* evaluate_schemes 把结果整理为 UncertaintyReport：已发现冲突的方案
  与范围内未见冲突的方案分开列出，未考察的组合单独计数——
  单一方案无碰撞不代表整个范围安全，离散对照也不是连续安全证明；
* attribute_difference 说明两个方案的差异来自感知选择、反应延迟
  还是后续信号安排，并给出首次使结果分开的交通事件。

修改不确定范围只会派生新方案，不回写旧方案已记录的感知事实。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .branching import BranchChanges, branch_from
from .comparison import Divergence, first_divergence
from .engine import CollisionRecord, ReplayEngine
from .scenario import ScriptedEvent


@dataclass(frozen=True)
class PerceptionWindow:
    """一个刺激的不确定感知范围；派生方案在范围内采用某一具体时刻。"""

    window_id: str
    vehicle_id: str
    stimulus: str  # 如 "lead_brake" / "signal_red"
    source_id: str  # 刺激来源（前车或信号灯）
    earliest_ms: int
    latest_ms: int
    delay_ms: int  # 该方案采用的反应延迟（毫秒）
    observed: Mapping[str, object] = field(default_factory=dict)
    response_action: Optional[Mapping[str, object]] = None  # 缺省 stop_actor


@dataclass(frozen=True)
class WindowConstraint:
    """两个感知范围之间的关系；kind 为 "after" 或 "same"。"""

    kind: str
    first_id: str
    second_id: str


@dataclass(frozen=True)
class AdoptedPerception:
    """方案实际采用的感知时刻与反应延迟。"""

    vehicle_id: str
    stimulus: str
    source_id: str
    tick_ms: int
    delay_ms: Optional[int]
    window_id: Optional[str]


@dataclass
class DerivedScheme:
    scheme_id: str
    assignment: Dict[str, int]  # window_id -> 采用的感知时刻
    engine: ReplayEngine


@dataclass(frozen=True)
class SchemeEvaluation:
    scheme_id: str
    assignment: Tuple[Tuple[str, int], ...]
    collisions: Tuple[CollisionRecord, ...]
    minimum_gap: Optional[float]
    adopted: Tuple[AdoptedPerception, ...]

    @property
    def clear(self) -> bool:
        return not self.collisions


@dataclass(frozen=True)
class UncertaintyReport:
    """离散对照结果；未考察部分与已发现冲突分开显示。"""

    evaluations: Tuple[SchemeEvaluation, ...]
    horizon_ms: int
    tick_ms: int
    total_assignments: Optional[int]

    @property
    def conflicts(self) -> Tuple[SchemeEvaluation, ...]:
        return tuple(evaluation for evaluation in self.evaluations if not evaluation.clear)

    @property
    def clears(self) -> Tuple[SchemeEvaluation, ...]:
        return tuple(evaluation for evaluation in self.evaluations if evaluation.clear)

    @property
    def explored_count(self) -> int:
        return len(self.evaluations)

    @property
    def unexplored_count(self) -> Optional[int]:
        if self.total_assignments is None:
            return None
        return max(0, self.total_assignments - self.explored_count)

    @property
    def covers_full_range(self) -> bool:
        return self.total_assignments is not None and self.unexplored_count == 0

    def conclusion(self) -> str:
        """人工可核对的结论；离散结果不得显示成连续安全结论。"""
        parts = [
            f"已考察 {self.explored_count} 个感知组合"
            + (
                f"（共 {self.total_assignments} 个）"
                if self.total_assignments is not None
                else ""
            ),
            f"{len(self.conflicts)} 个方案在 {self.horizon_ms}ms 内发现冲突",
            f"{len(self.clears)} 个方案在 {self.horizon_ms}ms 内未见冲突",
        ]
        unexplored = self.unexplored_count
        if unexplored is None:
            parts.append("组合总数未知，结果仅为抽样，不能据此判定范围安全")
        elif unexplored > 0:
            parts.append(
                f"另有 {unexplored} 个组合尚未考察，不能据此判定范围安全"
            )
        elif not self.conflicts:
            parts.append(f"全部组合在 {self.horizon_ms}ms 内无碰撞")
        parts.append(
            f"以上仅为 tick={self.tick_ms}ms 的离散采样与 "
            f"{self.horizon_ms}ms 有限时域内的对照，不构成连续时间安全证明"
        )
        return "；".join(parts) + "。"


@dataclass(frozen=True)
class DifferenceAttribution:
    """两个方案结果差异的归因与首次分开的交通事件。"""

    causes: Tuple[str, ...]  # perception_choice / reaction_delay / signal_plan
    divergence: Optional[Divergence]
    first_events_a: Tuple[object, ...]
    first_events_b: Tuple[object, ...]


def window_ticks(window: PerceptionWindow, tick_ms: int) -> Tuple[int, ...]:
    if window.earliest_ms > window.latest_ms:
        raise ValueError(f"window {window.window_id}: earliest after latest")
    for bound in (window.earliest_ms, window.latest_ms):
        if bound % tick_ms != 0:
            raise ValueError(
                f"window {window.window_id}: bound {bound} not aligned to tick {tick_ms}"
            )
    if window.delay_ms < 0 or window.delay_ms % tick_ms != 0:
        raise ValueError(
            f"window {window.window_id}: delay {window.delay_ms} must be "
            f"non-negative and aligned to tick {tick_ms}"
        )
    return tuple(range(window.earliest_ms, window.latest_ms + 1, tick_ms))


def enumerate_assignments(
    windows: Sequence[PerceptionWindow],
    constraints: Sequence[WindowConstraint] = (),
    tick_ms: int = 100,
) -> Tuple[Dict[str, int], ...]:
    """枚举所有满足约束的感知时刻组合（按窗口顺序字典序）。"""
    windows = tuple(windows)
    _validate_windows(windows)
    choices = [window_ticks(window, tick_ms) for window in windows]
    result: List[Dict[str, int]] = []
    for combo in product(*choices):
        assignment = {
            window.window_id: chosen for window, chosen in zip(windows, combo)
        }
        if _satisfies(assignment, constraints):
            result.append(assignment)
    return tuple(result)


def check_assignment(
    windows: Sequence[PerceptionWindow],
    constraints: Sequence[WindowConstraint],
    assignment: Mapping[str, int],
    tick_ms: int,
) -> None:
    """校验具体组合：必须在各自范围内且满足全部约束。"""
    windows = tuple(windows)
    _validate_windows(windows)
    by_id = {window.window_id: window for window in windows}
    if set(assignment) != set(by_id):
        raise ValueError(
            f"assignment windows {sorted(assignment)} != {sorted(by_id)}"
        )
    for window in windows:
        chosen = int(assignment[window.window_id])
        if chosen not in window_ticks(window, tick_ms):
            raise ValueError(
                f"window {window.window_id}: {chosen}ms 不在范围 "
                f"[{window.earliest_ms}, {window.latest_ms}] 内"
            )
    if not _satisfies(assignment, constraints):
        raise ValueError("assignment violates window constraints")


def derive_scheme(
    engine: ReplayEngine,
    scheme_id: str,
    windows: Sequence[PerceptionWindow],
    assignment: Mapping[str, int],
    constraints: Sequence[WindowConstraint] = (),
    at_ms: Optional[int] = None,
    base_changes: Optional[BranchChanges] = None,
) -> DerivedScheme:
    """按指定感知时刻组合派生一个具体方案（可在已有冲突附近追加）。"""
    windows = tuple(windows)
    tick_ms = engine.scenario.tick_ms
    check_assignment(windows, constraints, assignment, tick_ms)
    for window in windows:
        configured = engine.state.reaction_delays.get(window.vehicle_id, {})
        if window.stimulus in configured:
            raise ValueError(
                f"{window.vehicle_id} 已对 {window.stimulus} 配置自动响应延迟；"
                "不确定感知方案要求该刺激由 inject_perception 注入，"
                "请先在基础方案中移除对应 reaction_delays"
            )
    branch_point = min(assignment.values()) if at_ms is None else int(at_ms)
    if branch_point > min(assignment.values()):
        raise ValueError("branch point must not be after the earliest adopted perception")
    events = []
    for seq, window in enumerate(windows):
        chosen = int(assignment[window.window_id])
        events.append(
            ScriptedEvent(
                event_id=f"{scheme_id}:perceive:{window.window_id}",
                tick_ms=chosen,
                action={
                    "type": "inject_perception",
                    "vehicle_id": window.vehicle_id,
                    "stimulus": window.stimulus,
                    "source_id": window.source_id,
                    "delay_ms": window.delay_ms,
                    "observed": {**window.observed, "window_id": window.window_id},
                    "response_action": (
                        dict(window.response_action)
                        if window.response_action is not None
                        else None
                    ),
                    "reason": (
                        f"方案 {scheme_id} 采用 {window.vehicle_id} 在 {chosen}ms "
                        f"感知 {window.source_id} 的 {window.stimulus}，"
                        f"延迟 {window.delay_ms}ms"
                    ),
                },
                seq=seq,
            )
        )
    changes = _merge_changes(base_changes, events)
    branched = branch_from(engine, scheme_id, changes, at_ms=branch_point)
    return DerivedScheme(
        scheme_id=scheme_id,
        assignment={key: int(value) for key, value in assignment.items()},
        engine=branched,
    )


def derive_schemes(
    engine: ReplayEngine,
    scheme_prefix: str,
    windows: Sequence[PerceptionWindow],
    constraints: Sequence[WindowConstraint] = (),
    at_ms: Optional[int] = None,
    base_changes: Optional[BranchChanges] = None,
) -> Tuple[DerivedScheme, ...]:
    """枚举范围内全部合法组合，为每个组合派生一个方案。"""
    tick_ms = engine.scenario.tick_ms
    assignments = enumerate_assignments(windows, constraints, tick_ms)
    return tuple(
        derive_scheme(
            engine,
            f"{scheme_prefix}-{index}",
            windows,
            assignment,
            constraints=constraints,
            at_ms=at_ms,
            base_changes=base_changes,
        )
        for index, assignment in enumerate(assignments)
    )


def adopted_perception_facts(engine: ReplayEngine) -> Tuple[AdoptedPerception, ...]:
    """方案实际采用的感知时刻与反应延迟（注入感知，含已跳转抹除之外的记录）。"""
    facts: List[AdoptedPerception] = []
    for record in engine.perceptions:
        if not record.observed.get("injected"):
            continue
        facts.append(
            AdoptedPerception(
                vehicle_id=record.vehicle_id,
                stimulus=record.stimulus,
                source_id=record.source_id,
                tick_ms=record.tick_ms,
                delay_ms=record.observed.get("delay_ms"),
                window_id=record.observed.get("window_id"),
            )
        )
    return tuple(facts)


def evaluate_schemes(
    schemes: Sequence[DerivedScheme],
    until_ms: int,
    total_assignments: Optional[int] = None,
) -> UncertaintyReport:
    """逐个重放方案并汇总；未见冲突与已发现冲突分开列出。"""
    evaluations: List[SchemeEvaluation] = []
    tick_ms = None
    for scheme in schemes:
        engine = scheme.engine
        tick_ms = engine.scenario.tick_ms
        engine.seek(until_ms)
        collisions = tuple(
            collision
            for collision in engine.collisions
            if collision.tick_ms <= until_ms
        )
        gap = _minimum_gap_over(engine, until_ms)
        evaluations.append(
            SchemeEvaluation(
                scheme_id=scheme.scheme_id,
                assignment=tuple(sorted(scheme.assignment.items())),
                collisions=collisions,
                minimum_gap=gap,
                adopted=adopted_perception_facts(engine),
            )
        )
    return UncertaintyReport(
        evaluations=tuple(evaluations),
        horizon_ms=until_ms,
        tick_ms=tick_ms if tick_ms is not None else 0,
        total_assignments=total_assignments,
    )


def attribute_difference(
    engine_a: ReplayEngine,
    engine_b: ReplayEngine,
    until_ms: int,
) -> DifferenceAttribution:
    """说明结果差异来自感知选择、反应延迟还是后续信号安排。

    同时返回首次使两个方案分开的交通事件（first_divergence 时刻的
    感知、响应与脚本事件记录），便于人工回到分歧点核对。
    """
    facts_a = {
        (fact.vehicle_id, fact.stimulus, fact.source_id): fact
        for fact in adopted_perception_facts(engine_a)
    }
    facts_b = {
        (fact.vehicle_id, fact.stimulus, fact.source_id): fact
        for fact in adopted_perception_facts(engine_b)
    }
    causes: List[str] = []
    shared_keys = set(facts_a) & set(facts_b)
    if set(facts_a) != set(facts_b) or any(
        facts_a[key].tick_ms != facts_b[key].tick_ms for key in shared_keys
    ):
        causes.append("perception_choice")
    if any(
        facts_a[key].delay_ms != facts_b[key].delay_ms for key in shared_keys
    ):
        causes.append("reaction_delay")
    plans_a = {
        signal.signal_id: signal.phases for signal in engine_a.scenario.signals
    }
    plans_b = {
        signal.signal_id: signal.phases for signal in engine_b.scenario.signals
    }
    if plans_a != plans_b:
        causes.append("signal_plan")
    divergence = first_divergence(engine_a, engine_b, until_ms)
    first_events_a: Tuple[object, ...] = ()
    first_events_b: Tuple[object, ...] = ()
    if divergence is not None:
        first_events_a = _records_at(engine_a, divergence.tick_ms)
        first_events_b = _records_at(engine_b, divergence.tick_ms)
    return DifferenceAttribution(
        causes=tuple(causes),
        divergence=divergence,
        first_events_a=first_events_a,
        first_events_b=first_events_b,
    )


def _validate_windows(windows: Tuple[PerceptionWindow, ...]) -> None:
    ids = [window.window_id for window in windows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate window ids: {ids}")


def _satisfies(
    assignment: Mapping[str, int], constraints: Sequence[WindowConstraint]
) -> bool:
    for constraint in constraints:
        first = assignment.get(constraint.first_id)
        second = assignment.get(constraint.second_id)
        if first is None or second is None:
            raise ValueError(
                f"constraint references unknown window: {constraint}"
            )
        if constraint.kind == "after" and not second >= first:
            return False
        if constraint.kind == "same" and not second == first:
            return False
        if constraint.kind not in {"after", "same"}:
            raise ValueError(f"unknown constraint kind: {constraint.kind!r}")
    return True


def _merge_changes(
    base: Optional[BranchChanges], events: List[ScriptedEvent]
) -> BranchChanges:
    if base is None:
        return BranchChanges(extra_events=tuple(events))
    return BranchChanges(
        vehicle_routes=base.vehicle_routes,
        pedestrian_routes=base.pedestrian_routes,
        pedestrian_speeds=base.pedestrian_speeds,
        signal_plans=base.signal_plans,
        vehicle_reaction_params=base.vehicle_reaction_params,
        vehicle_reaction_delays=base.vehicle_reaction_delays,
        extra_events=tuple(base.extra_events) + tuple(events),
        drop_event_ids=base.drop_event_ids,
    )


def _minimum_gap_over(engine: ReplayEngine, until_ms: int) -> Optional[float]:
    tick = engine.scenario.tick_ms
    gaps: List[float] = []
    for current in range(0, until_ms + 1, tick):
        engine.seek(current)
        gap = engine.minimum_gap()
        if gap is not None:
            gaps.append(gap)
    engine.seek(until_ms)
    return min(gaps) if gaps else None


def _records_at(engine: ReplayEngine, tick_ms: int) -> Tuple[object, ...]:
    records: List[object] = []
    records.extend(
        record for record in engine.perceptions if record.tick_ms == tick_ms
    )
    records.extend(
        record
        for record in engine.responses
        if record.trigger_tick_ms == tick_ms or record.due_tick_ms == tick_ms
    )
    records.extend(record for record in engine.journal if record.tick_ms == tick_ms)
    return tuple(records)

