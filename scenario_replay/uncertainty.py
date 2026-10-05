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

import json
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .branching import BranchChanges, branch_from
from .comparison import Divergence, first_divergence
from .engine import CollisionRecord, ReplayEngine
from .scenario import Scenario, ScriptedEvent
from .storage import scenario_from_dict, scenario_to_dict


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
class ConstraintViolation:
    """离散时刻组合不满足刺激先后/同刻约束；它不是交通碰撞。"""

    assignment: Tuple[Tuple[str, int], ...]
    violated_constraints: Tuple[WindowConstraint, ...]
    reason: str


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
    constraint_violations: Tuple[ConstraintViolation, ...] = ()

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
        return max(
            0,
            self.total_assignments - self.explored_count - len(self.constraint_violations),
        )

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
        if self.constraint_violations:
            parts.append(
                f"{len(self.constraint_violations)} 个离散时刻组合不满足刺激约束（未派生、非碰撞）"
            )
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
    fact_divergence: Optional[Divergence] = None


@dataclass(frozen=True)
class PerceptionRevision:
    """待用户确认的感知范围、刺激约束和反应延迟修订。"""

    windows: Tuple[PerceptionWindow, ...]
    constraints: Tuple[WindowConstraint, ...] = ()
    base_scenario: Optional[Mapping[str, Any]] = None
    base_changes: Optional[Mapping[str, Any]] = None
    at_ms: Optional[int] = None
    scheme_prefix: str = "revised"
    horizon_ms: Optional[int] = None
    created_at_ms: Optional[int] = None
    reason: Optional[str] = None
    # 多候选身份：candidate_id 唯一标识同一份交通方案下的未决修订；
    # based_on 记录它从哪个候选（或原方案 None）继续编辑，形成可追溯链。
    candidate_id: Optional[str] = None
    based_on: Optional[str] = None
    note: Optional[str] = None


@dataclass(frozen=True)
class RevisionResult:
    revision: PerceptionRevision
    schemes: Tuple[DerivedScheme, ...]
    constraint_violations: Tuple[ConstraintViolation, ...]
    report: Optional[UncertaintyReport]


@dataclass(frozen=True)
class WindowImpact:
    """一份候选对单个观测范围的影响：观测区间、可选时刻数与反应延迟。"""

    window_id: str
    vehicle_id: str
    stimulus: str
    source_id: str
    earliest_ms: int
    latest_ms: int
    tick_choices: int
    delay_ms: int


@dataclass(frozen=True)
class CandidateComparison:
    """一份未决候选相对原方案的影响汇总，供多候选并排比较。"""

    candidate_id: str
    based_on: Optional[str]
    note: Optional[str]
    windows: Tuple[WindowImpact, ...]
    stimulus_order: Tuple[WindowConstraint, ...]
    reaction_delays_ms: Tuple[Tuple[str, int], ...]
    explored_assignments: Tuple[Tuple[Tuple[str, int], ...], ...]
    explored_count: int
    total_assignments: int
    constraint_excluded_count: int
    unexplored_count: int
    collisions_by_scheme: Tuple[Tuple[str, Tuple[CollisionRecord, ...]], ...]
    horizon_ms: int
    first_divergence_from_base: Optional[Divergence]
    first_divergence_scheme_id: Optional[str]
    fact_divergence_from_base: Optional[Divergence]
    fact_divergence_scheme_id: Optional[str]


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
    return tuple(
        assignment
        for assignment, violation in _all_assignments_and_violations(
            windows, constraints, tick_ms
        )
        if violation is None
    )


def constraint_violations(
    windows: Sequence[PerceptionWindow],
    constraints: Sequence[WindowConstraint],
    tick_ms: int = 100,
) -> Tuple[ConstraintViolation, ...]:
    """返回范围内所有不满足刺激约束的离散时刻组合。"""
    windows = tuple(windows)
    _validate_windows(windows)
    return tuple(
        violation
        for _, violation in _all_assignments_and_violations(
            windows, constraints, tick_ms
        )
        if violation is not None
    )


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


def stage_perception_revision(
    engine: ReplayEngine,
    windows: Sequence[PerceptionWindow],
    constraints: Sequence[WindowConstraint] = (),
    *,
    at_ms: Optional[int] = None,
    base_changes: Optional[BranchChanges] = None,
    scheme_prefix: str = "revised",
    horizon_ms: Optional[int] = None,
    reason: Optional[str] = None,
) -> PerceptionRevision:
    """暂存修订；保存/重开后仍需用户显式调用重派生。"""
    windows = tuple(windows)
    constraints = tuple(constraints)
    tick_ms = engine.scenario.tick_ms
    for window in windows:
        window_ticks(window, tick_ms)
    for constraint in constraints:
        ids = {window.window_id for window in windows}
        if constraint.first_id not in ids or constraint.second_id not in ids:
            raise ValueError(f"constraint references unknown window: {constraint}")
        if constraint.kind not in {"after", "same"}:
            raise ValueError(f"unknown constraint kind: {constraint.kind!r}")

    revision = PerceptionRevision(
        windows=windows,
        constraints=constraints,
        base_scenario=scenario_to_dict(_revision_base_scenario(engine)),
        base_changes=branch_changes_to_dict(base_changes) if base_changes else None,
        at_ms=at_ms,
        scheme_prefix=scheme_prefix,
        horizon_ms=horizon_ms,
        created_at_ms=engine.now_ms,
        reason=reason,
    )
    engine.pending_revision = revision_to_dict(revision)
    return revision


def stage_revision_candidate(
    engine: ReplayEngine,
    candidate_id: str,
    windows: Sequence[PerceptionWindow],
    constraints: Sequence[WindowConstraint] = (),
    *,
    at_ms: Optional[int] = None,
    base_changes: Optional[BranchChanges] = None,
    horizon_ms: Optional[int] = None,
    reason: Optional[str] = None,
    note: Optional[str] = None,
    based_on: Optional[str] = None,
) -> PerceptionRevision:
    """暂存一份命名候选；同一份交通方案可同时保留任意多份未决候选。

    多次编辑同一 candidate_id 是幂等更新（不产生重复候选）；不同
    candidate_id 互不覆盖，最后编辑的一份不会抹掉先前候选。
    based_on 可为另一候选 id，表示该候选由它继续编辑，形成
    “首次分歧”可追溯链；为 None 表示相对原交通方案修订。
    """
    if not candidate_id:
        raise ValueError("candidate_id must be a non-empty string")
    windows = tuple(windows)
    constraints = tuple(constraints)
    tick_ms = engine.scenario.tick_ms
    for window in windows:
        window_ticks(window, tick_ms)
    for constraint in constraints:
        ids = {window.window_id for window in windows}
        if constraint.first_id not in ids or constraint.second_id not in ids:
            raise ValueError(f"constraint references unknown window: {constraint}")
        if constraint.kind not in {"after", "same"}:
            raise ValueError(f"unknown constraint kind: {constraint.kind!r}")
    if based_on is not None and based_on not in engine.pending_revisions:
        raise ValueError(f"based_on candidate {based_on!r} is not staged")

    revision = PerceptionRevision(
        windows=windows,
        constraints=constraints,
        base_scenario=scenario_to_dict(_revision_base_scenario(engine)),
        base_changes=branch_changes_to_dict(base_changes) if base_changes else None,
        at_ms=at_ms,
        scheme_prefix=candidate_id,
        horizon_ms=horizon_ms,
        created_at_ms=engine.now_ms,
        reason=reason,
        candidate_id=candidate_id,
        based_on=based_on,
        note=note,
    )
    engine.pending_revisions[candidate_id] = revision_to_dict(revision)
    return revision


def load_revision_candidate(
    engine: ReplayEngine, candidate_id: str
) -> Optional[PerceptionRevision]:
    data = engine.pending_revisions.get(candidate_id)
    return revision_from_dict(data) if data else None


def list_revision_candidates(engine: ReplayEngine) -> Tuple[PerceptionRevision, ...]:
    """按暂存顺序返回全部未决候选；候选集合持久化并重开后仍在。"""
    return tuple(
        revision_from_dict(engine.pending_revisions[candidate_id])
        for candidate_id in engine.pending_revisions
    )


def discard_revision_candidate(engine: ReplayEngine, candidate_id: str) -> None:
    """丢弃一份候选；当前选择或历史链指向它时一并清理引用。"""
    engine.pending_revisions.pop(candidate_id, None)
    engine.adopted_revisions.pop(candidate_id, None)
    engine.revision_selection_history = [
        item for item in engine.revision_selection_history if item != candidate_id
    ]
    if engine.selected_revision_id == candidate_id:
        engine.selected_revision_id = (
        engine.revision_selection_history[-1]
        if engine.revision_selection_history
        else None
    )


def derive_revision_candidate(
    engine: ReplayEngine,
    candidate_id: str,
    *,
    adopt: bool = True,
    evaluate_until_ms: Optional[int] = None,
) -> RevisionResult:
    """显式派生某一份未决候选；其他候选与原方案事实一律保留。

    采用（adopt=True）只登记选择并把该候选从“未决”集合移出，
    不覆盖任何其他候选，也不改写引擎内原方案的场景与事件事实。
    重复确认同一候选是幂等操作，历史中不重复登记。
    """
    data = engine.pending_revisions.get(candidate_id)
    if data is None:
        already_selected = engine.selected_revision_id == candidate_id
        if already_selected and candidate_id in engine.adopted_revisions:
            data = engine.adopted_revisions[candidate_id]  # 重复确认：幂等重放
        else:
            raise ValueError(f"no staged revision candidate {candidate_id!r}")
    else:
        already_selected = engine.selected_revision_id == candidate_id
    result = _rederive_revision(
        revision_from_dict(data),
        evaluate_until_ms=evaluate_until_ms,
    )
    if adopt:
        engine.adopted_revisions[candidate_id] = dict(data)
        if not already_selected:
            engine.pending_revisions.pop(candidate_id, None)
        engine.pending_revision = None
        if not already_selected:
            engine.revision_selection_history.append(engine.selected_revision_id)
            engine.selected_revision_id = candidate_id
    return result


def undo_revision_selection(engine: ReplayEngine) -> Optional[str]:
    """撤销最近一次选择并改回上一个选择（默认原方案 None）。

    撤销不删除任何候选：被撤销的候选重新回到未决集合（若其修订
    数据仍可从上次派生结果外保留——见说明），从而允许改用另一份。
    为支持反复撤销/重选，撤销时会把当前候选重新暂存。
    """
    if not engine.revision_selection_history:
        return engine.selected_revision_id
    current = engine.selected_revision_id
    previous = engine.revision_selection_history.pop()
    if current is not None:
        adopted = engine.adopted_revisions.pop(current, None)
        if adopted is not None:
            engine.pending_revisions[current] = adopted
    engine.selected_revision_id = previous
    return previous


def _rederive_revision(
    stored: PerceptionRevision,
    *,
    evaluate_until_ms: Optional[int] = None,
) -> RevisionResult:
    if stored.base_scenario is None:
        raise ValueError("pending revision does not contain a replayable base scenario")
    base = ReplayEngine(scenario_from_dict(dict(stored.base_scenario)))
    base_changes = (
        branch_changes_from_dict(dict(stored.base_changes))
        if stored.base_changes is not None
        else None
    )
    tick_ms = base.scenario.tick_ms
    assignments = enumerate_assignments(stored.windows, stored.constraints, tick_ms)
    schemes = tuple(
        derive_scheme(
            base,
            f"{stored.scheme_prefix}-{index}",
            stored.windows,
            assignment,
            constraints=stored.constraints,
            at_ms=stored.at_ms,
            base_changes=base_changes,
        )
        for index, assignment in enumerate(assignments)
    )
    violations = constraint_violations(
        stored.windows, stored.constraints, tick_ms
    )
    total = len(assignments) + len(violations)
    horizon = evaluate_until_ms if evaluate_until_ms is not None else stored.horizon_ms
    report = (
        evaluate_schemes(
            schemes,
            horizon,
            total_assignments=total,
            constraint_violations_found=violations,
        )
        if horizon is not None
        else None
    )
    return RevisionResult(
        revision=stored,
        schemes=schemes,
        constraint_violations=violations,
        report=report,
    )


def load_pending_revision(engine: ReplayEngine) -> Optional[PerceptionRevision]:
    if not engine.pending_revision:
        return None
    return revision_from_dict(engine.pending_revision)


def rederive_pending_revision(
    engine: ReplayEngine,
    *,
    adopt: bool = True,
    evaluate_until_ms: Optional[int] = None,
) -> RevisionResult:
    """用户确认后才采用暂存修订；原引擎和旧方案事实不被原地改写。"""
    stored = load_pending_revision(engine)
    if stored is None:
        raise ValueError("no pending perception revision")
    result = _rederive_revision(stored, evaluate_until_ms=evaluate_until_ms)
    if adopt:
        engine.pending_revision = None
    return result


def compare_revision_candidates(
    candidates: Sequence["RevisionResult"],
    until_ms: Optional[int] = None,
    base_engine: Optional[ReplayEngine] = None,
) -> Tuple[CandidateComparison, ...]:
    """并排比较多份候选影响的观测范围、刺激顺序与反应延迟。

    每份候选分别给出：约束排除的离散组合数、实际重放出的碰撞、
    已探索的离散组合、未验证（未考察）范围，以及相对原方案的
    物理轨迹首处分歧与采用事实首处分歧。约束排除与实际碰撞始终
    分开计数，互不混入。
    """
    return tuple(
        _compare_one(result, until_ms=until_ms, base_engine=base_engine)
        for result in candidates
    )


def _compare_one(
    result: "RevisionResult",
    *,
    until_ms: Optional[int],
    base_engine: Optional[ReplayEngine],
) -> CandidateComparison:
    stored = result.revision
    tick_ms = (
        result.schemes[0].engine.scenario.tick_ms
        if result.schemes
        else (base_engine.scenario.tick_ms if base_engine is not None else 0)
    )
    horizon = until_ms if until_ms is not None else stored.horizon_ms
    windows = tuple(
        WindowImpact(
            window_id=window.window_id,
            vehicle_id=window.vehicle_id,
            stimulus=window.stimulus,
            source_id=window.source_id,
            earliest_ms=window.earliest_ms,
            latest_ms=window.latest_ms,
            tick_choices=len(window_ticks(window, tick_ms)) if tick_ms else 0,
            delay_ms=window.delay_ms,
        )
        for window in stored.windows
    )
    explored_assignments = tuple(
        tuple(sorted(scheme.assignment.items())) for scheme in result.schemes
    )
    collisions_by_scheme: List[Tuple[str, Tuple[CollisionRecord, ...]]] = []
    total = len(result.schemes) + len(result.constraint_violations)
    if horizon is not None:
        for scheme in result.schemes:
            scheme.engine.seek(horizon)
            collisions_by_scheme.append(
                (
                    scheme.scheme_id,
                    tuple(
                        collision
                        for collision in scheme.engine.collisions
                        if collision.tick_ms <= horizon
                    ),
                )
            )
    unexplored = max(
        0, total - len(result.schemes) - len(result.constraint_violations)
    )
    first_divergence_from_base = None
    first_divergence_scheme_id = None
    fact_divergence_from_base = None
    fact_divergence_scheme_id = None
    if base_engine is not None and horizon is not None and result.schemes:
        for scheme in result.schemes:
            divergence = first_divergence(base_engine, scheme.engine, horizon)
            if divergence is not None:
                first_divergence_from_base = divergence
                first_divergence_scheme_id = scheme.scheme_id
                break
        for scheme in result.schemes:
            attribution = attribute_difference(base_engine, scheme.engine, horizon)
            if attribution.fact_divergence is not None:
                fact_divergence_from_base = attribution.fact_divergence
                fact_divergence_scheme_id = scheme.scheme_id
                break
    return CandidateComparison(
        candidate_id=stored.candidate_id or stored.scheme_prefix,
        based_on=stored.based_on,
        note=stored.note,
        windows=windows,
        stimulus_order=stored.constraints,
        reaction_delays_ms=tuple(
            (window.window_id, window.delay_ms) for window in stored.windows
        ),
        explored_assignments=explored_assignments,
        explored_count=len(result.schemes),
        total_assignments=total,
        constraint_excluded_count=len(result.constraint_violations),
        unexplored_count=unexplored,
        collisions_by_scheme=tuple(collisions_by_scheme),
        horizon_ms=horizon if horizon is not None else 0,
        first_divergence_from_base=first_divergence_from_base,
        first_divergence_scheme_id=first_divergence_scheme_id,
        fact_divergence_from_base=fact_divergence_from_base,
        fact_divergence_scheme_id=fact_divergence_scheme_id,
    )


def _divergence_to_dict(divergence: Optional[Divergence]) -> Optional[Dict[str, Any]]:
    if divergence is None:
        return None
    return {
        "tick_ms": divergence.tick_ms,
        "aspects": list(divergence.aspects),
        "actors": list(divergence.actors),
        "signals": list(divergence.signals),
    }


def candidate_comparisons_to_dict(
    comparisons: Sequence[CandidateComparison],
) -> Dict[str, Any]:
    """导出 JSON 可序列化的比较结果；约束排除与实际碰撞分别说明。"""
    return {
        "candidates": [
            {
                "candidate_id": comparison.candidate_id,
                "based_on": comparison.based_on,
                "note": comparison.note,
                "observation_ranges": [
                    {
                        "window_id": impact.window_id,
                        "vehicle_id": impact.vehicle_id,
                        "stimulus": impact.stimulus,
                        "source_id": impact.source_id,
                        "earliest_ms": impact.earliest_ms,
                        "latest_ms": impact.latest_ms,
                        "tick_choices": impact.tick_choices,
                        "reaction_delay_ms": impact.delay_ms,
                    }
                    for impact in comparison.windows
                ],
                "stimulus_order": [
                    {
                        "kind": constraint.kind,
                        "first_id": constraint.first_id,
                        "second_id": constraint.second_id,
                    }
                    for constraint in comparison.stimulus_order
                ],
                "reaction_delays_ms": [
                    {"window_id": window_id, "delay_ms": delay}
                    for window_id, delay in comparison.reaction_delays_ms
                ],
                "explored_combinations": [
                    {window_id: tick for window_id, tick in assignment}
                    for assignment in comparison.explored_assignments
                ],
                "explored_count": comparison.explored_count,
                "total_combinations": comparison.total_assignments,
                "constraint_excluded_count": comparison.constraint_excluded_count,
                "unverified_count": comparison.unexplored_count,
                "actual_collisions": [
                    {
                        "scheme_id": scheme_id,
                        "collisions": [
                            {
                                "tick_ms": collision.tick_ms,
                                "actor_a": collision.actor_a,
                                "actor_b": collision.actor_b,
                                "distance": collision.distance,
                                "reason": collision.reason,
                            }
                            for collision in collisions
                        ],
                    }
                    for scheme_id, collisions in comparison.collisions_by_scheme
                    if collisions
                ],
                "horizon_ms": comparison.horizon_ms,
                "first_divergence_from_base": _divergence_to_dict(
                    comparison.first_divergence_from_base
                ),
                "first_divergence_scheme_id": comparison.first_divergence_scheme_id,
                "first_fact_divergence_from_base": _divergence_to_dict(
                    comparison.fact_divergence_from_base
                ),
                "first_fact_divergence_scheme_id": comparison.fact_divergence_scheme_id,
            }
            for comparison in comparisons
        ],
    }


def export_revision_comparison(
    candidates: Sequence["RevisionResult"],
    path: Optional[str] = None,
    *,
    until_ms: Optional[int] = None,
    base_engine: Optional[ReplayEngine] = None,
) -> Dict[str, Any]:
    """导出多候选比较；给 path 时同时写 JSON 文件，始终返回可序列化字典。"""
    payload = candidate_comparisons_to_dict(
        compare_revision_candidates(
            candidates, until_ms=until_ms, base_engine=base_engine
        )
    )
    if path is not None:
        from pathlib import Path

        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return payload


def _revision_base_scenario(engine: ReplayEngine) -> Scenario:
    """移除当前待修订方案自身注入的未来/待响应事实，防止新方案借用。"""
    current_branch = engine.scenario.branch_id
    if current_branch is None:
        return engine.scenario
    prefix = f"{current_branch}:perceive:"
    kept_events = tuple(
        event
        for event in engine.scenario.events
        if not (
            event.action.get("type") == "inject_perception"
            and event.event_id.startswith(prefix)
        )
    )
    if len(kept_events) == len(engine.scenario.events):
        return engine.scenario
    return Scenario(
        name=engine.scenario.name,
        version=engine.scenario.version,
        seed=engine.scenario.seed,
        vehicles=engine.scenario.vehicles,
        pedestrians=engine.scenario.pedestrians,
        signals=engine.scenario.signals,
        events=kept_events,
        tick_ms=engine.scenario.tick_ms,
        branch_id=engine.scenario.branch_id,
        parent_id=engine.scenario.parent_id,
        branch_point_ms=engine.scenario.branch_point_ms,
        reaction_params=engine.scenario.reaction_params,
        reaction_delays=engine.scenario.reaction_delays,
    )


def branch_changes_to_dict(changes: BranchChanges) -> Dict[str, Any]:
    return {
        "vehicle_routes": {k: [list(p) for p in v] for k, v in changes.vehicle_routes.items()},
        "pedestrian_routes": {k: [list(p) for p in v] for k, v in changes.pedestrian_routes.items()},
        "pedestrian_speeds": dict(changes.pedestrian_speeds),
        "signal_plans": {
            k: [[t, phase] for t, phase in v]
            for k, v in changes.signal_plans.items()
        },
        "vehicle_reaction_params": {
            k: dict(v) for k, v in changes.vehicle_reaction_params.items()
        },
        "vehicle_reaction_delays": {
            k: dict(v) for k, v in changes.vehicle_reaction_delays.items()
        },
        "extra_events": [
            {
                "event_id": event.event_id,
                "tick_ms": event.tick_ms,
                "action": dict(event.action),
                "seq": event.seq,
            }
            for event in changes.extra_events
        ],
        "drop_event_ids": list(changes.drop_event_ids),
    }


def branch_changes_from_dict(data: Mapping[str, Any]) -> BranchChanges:
    return BranchChanges(
        vehicle_routes={
            k: tuple((float(p[0]), float(p[1])) for p in v)
            for k, v in data.get("vehicle_routes", {}).items()
        },
        pedestrian_routes={
            k: tuple((float(p[0]), float(p[1])) for p in v)
            for k, v in data.get("pedestrian_routes", {}).items()
        },
        pedestrian_speeds={k: float(v) for k, v in data.get("pedestrian_speeds", {}).items()},
        signal_plans={
            k: tuple((int(t), phase) for t, phase in v)
            for k, v in data.get("signal_plans", {}).items()
        },
        vehicle_reaction_params={
            k: {stimulus: float(value) for stimulus, value in params.items()}
            for k, params in data.get("vehicle_reaction_params", {}).items()
        },
        vehicle_reaction_delays={
            k: {stimulus: float(value) for stimulus, value in params.items()}
            for k, params in data.get("vehicle_reaction_delays", {}).items()
        },
        extra_events=tuple(
            ScriptedEvent(
                event_id=event["event_id"],
                tick_ms=int(event["tick_ms"]),
                action=dict(event["action"]),
                seq=int(event.get("seq", 0)),
            )
            for event in data.get("extra_events", ())
        ),
        drop_event_ids=tuple(data.get("drop_event_ids", ())),
    )


def revision_to_dict(revision: PerceptionRevision) -> Dict[str, Any]:
    return {
        "windows": [
            {
                "window_id": window.window_id,
                "vehicle_id": window.vehicle_id,
                "stimulus": window.stimulus,
                "source_id": window.source_id,
                "earliest_ms": window.earliest_ms,
                "latest_ms": window.latest_ms,
                "delay_ms": window.delay_ms,
                "observed": dict(window.observed),
                "response_action": (
                    dict(window.response_action)
                    if window.response_action is not None
                    else None
                ),
            }
            for window in revision.windows
        ],
        "constraints": [
            {
                "kind": constraint.kind,
                "first_id": constraint.first_id,
                "second_id": constraint.second_id,
            }
            for constraint in revision.constraints
        ],
        "base_scenario": dict(revision.base_scenario) if revision.base_scenario else None,
        "base_changes": dict(revision.base_changes) if revision.base_changes else None,
        "at_ms": revision.at_ms,
        "scheme_prefix": revision.scheme_prefix,
        "horizon_ms": revision.horizon_ms,
        "created_at_ms": revision.created_at_ms,
        "reason": revision.reason,
        "candidate_id": revision.candidate_id,
        "based_on": revision.based_on,
        "note": revision.note,
    }


def revision_from_dict(data: Mapping[str, Any]) -> PerceptionRevision:
    return PerceptionRevision(
        windows=tuple(
            PerceptionWindow(
                window_id=window["window_id"],
                vehicle_id=window["vehicle_id"],
                stimulus=window["stimulus"],
                source_id=window["source_id"],
                earliest_ms=int(window["earliest_ms"]),
                latest_ms=int(window["latest_ms"]),
                delay_ms=int(window["delay_ms"]),
                observed=dict(window.get("observed", {})),
                response_action=dict(window["response_action"])
                if window.get("response_action") is not None
                else None,
            )
            for window in data["windows"]
        ),
        constraints=tuple(
            WindowConstraint(
                kind=constraint["kind"],
                first_id=constraint["first_id"],
                second_id=constraint["second_id"],
            )
            for constraint in data.get("constraints", ())
        ),
        base_scenario=dict(data["base_scenario"]) if data.get("base_scenario") else None,
        base_changes=dict(data["base_changes"]) if data.get("base_changes") else None,
        at_ms=int(data["at_ms"]) if data.get("at_ms") is not None else None,
        scheme_prefix=data.get("scheme_prefix", "revised"),
        horizon_ms=int(data["horizon_ms"]) if data.get("horizon_ms") is not None else None,
        created_at_ms=int(data["created_at_ms"]) if data.get("created_at_ms") is not None else None,
        reason=data.get("reason"),
        candidate_id=data.get("candidate_id"),
        based_on=data.get("based_on"),
        note=data.get("note"),
    )


def evaluate_schemes(
    schemes: Sequence[DerivedScheme],
    until_ms: int,
    total_assignments: Optional[int] = None,
    constraint_violations_found: Sequence[ConstraintViolation] = (),
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
        constraint_violations=tuple(constraint_violations_found),
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
    fact_divergence = _first_fact_divergence(engine_a, engine_b, until_ms, causes)
    if fact_divergence is not None and (not first_events_a or not first_events_b):
        first_events_a = _records_at(engine_a, fact_divergence.tick_ms)
        first_events_b = _records_at(engine_b, fact_divergence.tick_ms)
    return DifferenceAttribution(
        causes=tuple(causes),
        divergence=divergence,
        first_events_a=first_events_a,
        first_events_b=first_events_b,
        fact_divergence=fact_divergence,
    )


def _first_fact_divergence(
    engine_a: ReplayEngine,
    engine_b: ReplayEngine,
    until_ms: int,
    causes: Sequence[str],
) -> Optional[Divergence]:
    if not causes:
        return None
    candidates: List[int] = []
    perceptions_a = _fact_tuples(engine_a.perceptions, "perception")
    perceptions_b = _fact_tuples(engine_b.perceptions, "perception")
    responses_a = _fact_tuples(engine_a.responses, "response")
    responses_b = _fact_tuples(engine_b.responses, "response")
    events_a = _fact_tuples(engine_a.journal, "event")
    events_b = _fact_tuples(engine_b.journal, "event")
    if perceptions_a != perceptions_b:
        candidates.append(_first_different_time(perceptions_a, perceptions_b, until_ms))
    if responses_a != responses_b:
        candidates.append(_first_different_time(responses_a, responses_b, until_ms))
    if events_a != events_b:
        candidates.append(_first_different_time(events_a, events_b, until_ms))
    return Divergence(tick_ms=min(candidates), aspects=("adopted_facts",)) if candidates else None


def _fact_tuples(records: Sequence[object], kind: str) -> Tuple[Tuple[int, tuple], ...]:
    normalized: List[Tuple[int, tuple]] = []
    for index, record in enumerate(records):
        if kind == "perception":
            if not getattr(record, "observed", {}).get("injected"):
                continue
            when = record.tick_ms
            identity = (
                record.vehicle_id,
                record.stimulus,
                record.source_id,
                _stable(_observed_without_scheme_fields(record.observed)),
            )
        elif kind == "response":
            when = record.trigger_tick_ms
            identity = (
                record.vehicle_id,
                record.stimulus,
                record.source_id,
                record.due_tick_ms,
                _stable(record.action),
            )
        else:
            when = record.tick_ms
            if record.action.get("type") != "inject_perception":
                continue
            identity = (
                record.action.get("vehicle_id"),
                record.action.get("stimulus"),
                record.action.get("source_id"),
                record.action.get("delay_ms"),
                _stable(_observed_without_scheme_fields(record.action.get("observed", {}))),
            )
        if when <= max(0, 10**12):
            normalized.append((when, identity))
    return tuple(normalized)


def _first_different_time(
    records_a: Sequence[Tuple[int, tuple]],
    records_b: Sequence[Tuple[int, tuple]],
    until_ms: int,
) -> int:
    set_a = set(records_a)
    set_b = set(records_b)
    times = [
        when
        for when, _ in (*(set_a - set_b), *(set_b - set_a))
        if when <= until_ms
    ]
    return min(times) if times else until_ms


def _validate_windows(windows: Tuple[PerceptionWindow, ...]) -> None:
    ids = [window.window_id for window in windows]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate window ids: {ids}")


def _all_assignments_and_violations(
    windows: Tuple[PerceptionWindow, ...],
    constraints: Sequence[WindowConstraint],
    tick_ms: int,
) -> Tuple[Tuple[Dict[str, int], Optional[ConstraintViolation]], ...]:
    choices = [window_ticks(window, tick_ms) for window in windows]
    result: List[Tuple[Dict[str, int], Optional[ConstraintViolation]]] = []
    for combo in product(*choices):
        assignment = {
            window.window_id: chosen for window, chosen in zip(windows, combo)
        }
        violated = _violated_constraints(assignment, constraints)
        if violated:
            result.append(
                (
                    assignment,
                    ConstraintViolation(
                        assignment=tuple(sorted(assignment.items())),
                        violated_constraints=tuple(violated),
                        reason="感知组合不满足刺激先后/同刻约束，未派生为交通方案；这是约束不满足，非碰撞",
                    ),
                )
            )
        else:
            result.append((assignment, None))
    return tuple(result)


def _violated_constraints(
    assignment: Mapping[str, int], constraints: Sequence[WindowConstraint]
) -> List[WindowConstraint]:
    violated: List[WindowConstraint] = []
    for constraint in constraints:
        first = assignment.get(constraint.first_id)
        second = assignment.get(constraint.second_id)
        if first is None or second is None:
            raise ValueError(
                f"constraint references unknown window: {constraint}"
            )
        if constraint.kind not in {"after", "same"}:
            raise ValueError(f"unknown constraint kind: {constraint.kind!r}")
        if constraint.kind == "after" and second < first:
            violated.append(constraint)
        if constraint.kind == "same" and second != first:
            violated.append(constraint)
    return violated


def _satisfies(
    assignment: Mapping[str, int], constraints: Sequence[WindowConstraint]
) -> bool:
    return not _violated_constraints(assignment, constraints)


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


def _stable(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _observed_without_scheme_fields(observed: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in observed.items()
        if key not in {"window_id", "injected", "delay_ms"}
    }
