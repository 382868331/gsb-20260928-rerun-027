"""不确定感知时间范围的分支对照。

场景记录有时只能给出"驾驶员在 [earliest_ms, latest_ms] 内看到了前车
制动"这类区间事实，无法定位唯一感知时刻。本模块把区间作为分支输入：
枚举区间内（按 tick 对齐）的候选感知时刻，结合感知间约束（先后顺序 /
同一次观察）生成具体方案，每个方案作为独立分支运行，并保留该方案实际
采用的感知时刻与反应延迟。

结果是离散对照：只覆盖被考察的 tick 组合。某个方案没有碰撞不代表整个
范围安全；未被考察的区间在报告中单独列出，不得把离散结论展示为未经
验证的连续安全结论。
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Mapping, Optional, Sequence, Tuple

from .branching import BranchChanges, branch_from
from .comparison import Divergence, first_divergence
from .engine import (
    CollisionRecord,
    PerceptionRecord,
    ReplayEngine,
    ResponseRecord,
)
from .scenario import PlannedPerception


@dataclass(frozen=True)
class UncertainPerception:
    """一次无法定位唯一时刻的感知：落在 [earliest_ms, latest_ms] 内。"""

    key: str
    vehicle_id: str
    stimulus: str
    source_id: str
    earliest_ms: int
    latest_ms: int
    # 感知后的反应延迟（毫秒）；None 表示只记录感知、不调度动作。
    delay_ms: Optional[int] = None


@dataclass(frozen=True)
class PerceptionConstraint:
    """感知范围之间的关系。

    kind="order"：second 的感知时刻不得早于 first；
    kind="same_observation"：两条记录来自同一次观察，时刻必须相同。
    """

    kind: str
    first: str
    second: str


@dataclass(frozen=True)
class PerceptionScheme:
    """一个具体方案：每条不确定感知取定的感知时刻与反应延迟。"""

    scheme_id: str
    assignments: Tuple[Tuple[str, int], ...]  # (uncertainty key, 感知时刻)
    delays: Tuple[Tuple[str, Optional[int]], ...]  # (uncertainty key, 反应延迟)

    def assignment_of(self, key: str) -> int:
        return dict(self.assignments)[key]

    def delay_of(self, key: str) -> Optional[int]:
        return dict(self.delays)[key]


@dataclass(frozen=True)
class UnexploredInterval:
    """某条不确定感知范围内未被任何方案考察的闭区间。"""

    key: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True)
class SchemeResult:
    """一个方案的运行结果；engine 保留下来可继续派生嵌套分支。"""

    scheme: PerceptionScheme
    branch_id: str
    engine: ReplayEngine
    collisions: Tuple[CollisionRecord, ...]  # 分支点之后新发生的碰撞
    perceptions: Tuple[PerceptionRecord, ...]  # 该方案实际采用的感知事实
    responses: Tuple[ResponseRecord, ...]


@dataclass(frozen=True)
class DifferenceExplanation:
    """两个方案结果差异的归因与首次分开的位置。"""

    causes: Tuple[str, ...]  # perception_choice / reaction_delay / signal_schedule
    first_divergence: Optional[Divergence]
    separating_events: Tuple[object, ...]  # 首次分开 tick 上的感知/响应/事件


@dataclass(frozen=True)
class UncertaintyReport:
    """一次不确定感知对照的结果集（离散采样，非连续证明）。"""

    results: Tuple[SchemeResult, ...]
    unexplored: Tuple[UnexploredInterval, ...]
    coverage: str = "discrete"

    @property
    def conflicts(self) -> Tuple[SchemeResult, ...]:
        """已发现冲突的方案。"""
        return tuple(result for result in self.results if result.collisions)

    @property
    def clean(self) -> Tuple[SchemeResult, ...]:
        """已考察且未发现碰撞的方案；不代表其邻近未考察区间安全。"""
        return tuple(result for result in self.results if not result.collisions)

    @property
    def is_continuous_safety_verified(self) -> bool:
        """离散对照永远不能当作连续安全结论，恒为 False。"""
        return False


def enumerate_schemes(
    uncertainties: Sequence[UncertainPerception],
    constraints: Sequence[PerceptionConstraint] = (),
    tick_ms: int = 100,
    samples: Optional[Mapping[str, Sequence[int]]] = None,
    scheme_prefix: str = "scheme",
) -> Tuple[PerceptionScheme, ...]:
    """枚举满足约束的感知时刻组合。

    samples 可指定某条不确定感知只考察哪些 tick（如只考察边界），
    缺省考察范围内全部 tick。组合必须满足全部约束，不会把各条记录
    各自最有利的时刻任意拼接。
    """
    uncertainties = tuple(uncertainties)
    keys = [item.key for item in uncertainties]
    if len(keys) != len(set(keys)):
        raise ValueError("uncertain perception keys must be unique")
    known = set(keys)
    for constraint in constraints:
        if constraint.kind not in {"order", "same_observation"}:
            raise ValueError(f"unknown constraint kind: {constraint.kind!r}")
        if constraint.first not in known or constraint.second not in known:
            raise KeyError(
                f"constraint references unknown key: "
                f"{constraint.first!r}, {constraint.second!r}"
            )
    candidates = {}
    for item in uncertainties:
        _validate_range(item, tick_ms)
        ticks = None if samples is None else samples.get(item.key)
        if ticks is None:
            ticks = tuple(range(item.earliest_ms, item.latest_ms + 1, tick_ms))
        else:
            ticks = tuple(int(tick) for tick in ticks)
            for tick in ticks:
                if tick % tick_ms != 0:
                    raise ValueError("sample tick must align with tick_ms")
                if not item.earliest_ms <= tick <= item.latest_ms:
                    raise ValueError(
                        f"sample {tick} outside range of {item.key!r}"
                    )
        if not ticks:
            raise ValueError(f"no candidate ticks for {item.key!r}")
        candidates[item.key] = ticks
    schemes = []
    for combo in itertools.product(*(candidates[key] for key in keys)):
        assignment = dict(zip(keys, combo))
        if not _satisfies(constraints, assignment):
            continue
        schemes.append(
            PerceptionScheme(
                scheme_id=f"{scheme_prefix}-{len(schemes)}",
                assignments=tuple((key, assignment[key]) for key in keys),
                delays=tuple((item.key, item.delay_ms) for item in uncertainties),
            )
        )
    return tuple(schemes)


def make_scheme(
    assignments: Mapping[str, int],
    uncertainties: Sequence[UncertainPerception],
    delays: Optional[Mapping[str, Optional[int]]] = None,
    scheme_id: str = "custom",
) -> PerceptionScheme:
    """在已有冲突附近追加一个具体方案（手工指定感知时刻/延迟）。"""
    uncertainties = tuple(uncertainties)
    keys = [item.key for item in uncertainties]
    if set(assignments) != set(keys):
        raise ValueError("assignments must cover every uncertainty key exactly")
    delay_map = {item.key: item.delay_ms for item in uncertainties}
    if delays:
        unknown = set(delays) - set(keys)
        if unknown:
            raise KeyError(f"delays reference unknown keys: {sorted(unknown)}")
        delay_map.update(delays)
    return PerceptionScheme(
        scheme_id=scheme_id,
        assignments=tuple((key, int(assignments[key])) for key in keys),
        delays=tuple((key, delay_map[key]) for key in keys),
    )


def evaluate_uncertainty(
    engine: ReplayEngine,
    uncertainties: Sequence[UncertainPerception],
    constraints: Sequence[PerceptionConstraint] = (),
    until_ms: int = 0,
    at_ms: Optional[int] = None,
    samples: Optional[Mapping[str, Sequence[int]]] = None,
    changes: Optional[BranchChanges] = None,
    scheme_prefix: str = "scheme",
) -> UncertaintyReport:
    """枚举方案并逐个作为分支运行，返回离散对照报告。"""
    uncertainties = tuple(uncertainties)
    schemes = enumerate_schemes(
        uncertainties,
        constraints,
        tick_ms=engine.scenario.tick_ms,
        samples=samples,
        scheme_prefix=scheme_prefix,
    )
    results = tuple(
        _run_scheme(engine, uncertainties, scheme, at_ms, until_ms, changes)
        for scheme in schemes
    )
    return UncertaintyReport(
        results=results,
        unexplored=_unexplored(uncertainties, results, engine.scenario.tick_ms),
    )


def append_scheme(
    report: UncertaintyReport,
    engine: ReplayEngine,
    uncertainties: Sequence[UncertainPerception],
    scheme: PerceptionScheme,
    until_ms: int,
    at_ms: Optional[int] = None,
    changes: Optional[BranchChanges] = None,
) -> UncertaintyReport:
    """在已有报告上追加一个具体方案；旧方案的感知事实原样保留。"""
    uncertainties = tuple(uncertainties)
    result = _run_scheme(engine, uncertainties, scheme, at_ms, until_ms, changes)
    results = (*report.results, result)
    return UncertaintyReport(
        results=results,
        unexplored=_unexplored(uncertainties, results, engine.scenario.tick_ms),
        coverage=report.coverage,
    )


def explain_difference(
    result_a: SchemeResult,
    result_b: SchemeResult,
    until_ms: int,
) -> DifferenceExplanation:
    """说明两个方案的结果差异来自哪里，并定位首次分开的交通事件。"""
    causes = []
    if dict(result_a.scheme.assignments) != dict(result_b.scheme.assignments):
        causes.append("perception_choice")
    if dict(result_a.scheme.delays) != dict(result_b.scheme.delays):
        causes.append("reaction_delay")
    if _signal_plans(result_a.engine) != _signal_plans(result_b.engine):
        causes.append("signal_schedule")
    divergence = first_divergence(result_a.engine, result_b.engine, until_ms)
    separating: Tuple[object, ...] = ()
    if divergence is not None:
        separating = _events_at_tick(
            (result_a.engine, result_b.engine), divergence.tick_ms
        )
    return DifferenceExplanation(
        causes=tuple(causes),
        first_divergence=divergence,
        separating_events=separating,
    )


def _run_scheme(
    engine: ReplayEngine,
    uncertainties: Tuple[UncertainPerception, ...],
    scheme: PerceptionScheme,
    at_ms: Optional[int],
    until_ms: int,
    changes: Optional[BranchChanges],
) -> SchemeResult:
    plans = []
    for item in uncertainties:
        plans.append(
            PlannedPerception(
                plan_id=f"{scheme.scheme_id}:{item.key}",
                vehicle_id=item.vehicle_id,
                stimulus=item.stimulus,
                source_id=item.source_id,
                tick_ms=scheme.assignment_of(item.key),
                delay_ms=scheme.delay_of(item.key),
                observed={
                    "scheme_id": scheme.scheme_id,
                    "uncertain_range_ms": [item.earliest_ms, item.latest_ms],
                },
            )
        )
    base = changes if changes is not None else BranchChanges()
    merged = BranchChanges(
        vehicle_routes=base.vehicle_routes,
        pedestrian_routes=base.pedestrian_routes,
        pedestrian_speeds=base.pedestrian_speeds,
        signal_plans=base.signal_plans,
        vehicle_reaction_params=base.vehicle_reaction_params,
        vehicle_reaction_delays=base.vehicle_reaction_delays,
        planned_perceptions=(*base.planned_perceptions, *plans),
        extra_events=base.extra_events,
        drop_event_ids=base.drop_event_ids,
    )
    branch = branch_from(engine, scheme.scheme_id, merged, at_ms=at_ms)
    branch.run_to(until_ms)
    branch_point = branch.scenario.branch_point_ms or 0
    return SchemeResult(
        scheme=scheme,
        branch_id=scheme.scheme_id,
        engine=branch,
        collisions=tuple(
            record for record in branch.collisions if record.tick_ms > branch_point
        ),
        perceptions=tuple(
            record
            for record in branch.perception_records()
            if record.perception_id.startswith("planned:")
        ),
        responses=tuple(
            record
            for record in branch.response_records()
            if record.response_id.startswith("planned-response:")
        ),
    )


def _unexplored(
    uncertainties: Tuple[UncertainPerception, ...],
    results: Tuple[SchemeResult, ...],
    tick_ms: int,
) -> Tuple[UnexploredInterval, ...]:
    examined = {item.key: set() for item in uncertainties}
    for result in results:
        for key, tick in result.scheme.assignments:
            if key in examined:
                examined[key].add(tick)
    intervals = []
    for item in uncertainties:
        missing = [
            tick
            for tick in range(item.earliest_ms, item.latest_ms + 1, tick_ms)
            if tick not in examined[item.key]
        ]
        intervals.extend(
            UnexploredInterval(key=item.key, start_ms=start, end_ms=end)
            for start, end in _group_intervals(missing, tick_ms)
        )
    return tuple(intervals)


def _group_intervals(ticks, tick_ms: int):
    if not ticks:
        return ()
    groups = []
    start = prev = ticks[0]
    for tick in ticks[1:]:
        if tick != prev + tick_ms:
            groups.append((start, prev))
            start = tick
        prev = tick
    groups.append((start, prev))
    return tuple(groups)


def _validate_range(item: UncertainPerception, tick_ms: int) -> None:
    if item.earliest_ms > item.latest_ms:
        raise ValueError(f"empty range for {item.key!r}")
    for bound in (item.earliest_ms, item.latest_ms):
        if bound % tick_ms != 0:
            raise ValueError("uncertainty bounds must align with tick_ms")
    if item.delay_ms is not None and item.delay_ms % tick_ms != 0:
        raise ValueError("reaction delay must align with tick_ms")


def _satisfies(constraints, assignment) -> bool:
    for constraint in constraints:
        first = assignment[constraint.first]
        second = assignment[constraint.second]
        if constraint.kind == "order" and second < first:
            return False
        if constraint.kind == "same_observation" and second != first:
            return False
    return True


def _signal_plans(engine: ReplayEngine):
    return tuple(
        (signal.signal_id, tuple(signal.phases)) for signal in engine.scenario.signals
    )


def _events_at_tick(engines, tick_ms: int) -> Tuple[object, ...]:
    seen = []
    for engine in engines:
        engine.seek(tick_ms)
        seen.extend(
            record
            for record in engine.perception_records()
            if record.tick_ms == tick_ms
        )
        seen.extend(
            record
            for record in engine.response_records()
            if record.due_tick_ms == tick_ms
        )
        seen.extend(record for record in engine.journal if record.tick_ms == tick_ms)
    return tuple(seen)

