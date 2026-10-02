"""不确定感知时间范围的分支对照测试（可人工核对的小例子）。"""
import pytest

from scenario_replay import (
    BranchChanges,
    PlannedPerception,
    ReplayEngine,
    Scenario,
    ScriptedEvent,
    Signal,
    Vehicle,
    append_scheme,
    branch_from,
    enumerate_schemes,
    evaluate_uncertainty,
    explain_difference,
    load_run,
    make_scheme,
    PerceptionConstraint,
    UncertainPerception,
    save_run,
)


def lead_brake_scenario() -> Scenario:
    # 前车 500ms 时急停；后车 10 m/s（每 tick 前进 1m）。碰撞半径各 2m。
    # 人工核对：500ms 时后车在 5m、前车停在 17m；间距 <=4m 即碰撞，
    # 故 1300ms 后车到达 13m 时碰撞。制动减速度 20m/s^2（每 tick 2m/s），
    # 10m/s 刹停需 2m：感知时刻 t 的方案在 t/100+2 米处停住，
    # t>=1000ms 时停不住（>=12m 处已入碰撞范围）。
    return Scenario(
        name="uncertain-lead-brake",
        version="1.0.0",
        seed=3,
        vehicles=(
            Vehicle("follower", ((0.0, 0.0), (200.0, 0.0)), 10.0),
            Vehicle("leader", ((12.0, 0.0), (200.0, 0.0)), 10.0),
        ),
        signals=(Signal("j1", ((0, "green"),)),),
        events=(
            ScriptedEvent(
                "leader-brakes", 500, {"type": "stop_actor", "actor_id": "leader"}
            ),
        ),
        tick_ms=100,
    )


def saw_brake(delay_ms=200, latest=1300) -> UncertainPerception:
    return UncertainPerception(
        key="saw-lead-brake",
        vehicle_id="follower",
        stimulus="lead_brake",
        source_id="leader",
        earliest_ms=500,
        latest_ms=latest,
        delay_ms=delay_ms,
    )


def brake_changes() -> BranchChanges:
    return BranchChanges(vehicle_reaction_params={"follower": {"brake_rate": 20.0}})


def evaluate_lead_brake(**kwargs):
    engine = ReplayEngine(lead_brake_scenario())
    return evaluate_uncertainty(
        engine,
        [saw_brake(**{k: v for k, v in kwargs.items() if k in {"delay_ms", "latest"}})],
        until_ms=2000,
        at_ms=500,
        changes=brake_changes(),
        samples=kwargs.get("samples"),
    )


def scheme_ticks(report):
    return {
        result.scheme.assignment_of("saw-lead-brake"): result
        for result in report.results
    }


def test_range_schemes_keep_actual_perception_tick_and_delay():
    report = evaluate_lead_brake()
    assert report.coverage == "discrete"
    assert not report.is_continuous_safety_verified
    by_tick = scheme_ticks(report)
    assert set(by_tick) == set(range(500, 1301, 100))
    for tick, result in by_tick.items():
        (perception,) = result.perceptions
        assert perception.tick_ms == tick
        assert perception.observed["uncertain_range_ms"] == [500, 1300]
        (response,) = result.responses
        assert response.trigger_tick_ms == tick
        assert response.due_tick_ms == tick + 200


def test_clean_scheme_does_not_imply_range_safe():
    report = evaluate_lead_brake()
    by_tick = scheme_ticks(report)
    # 人工核对：感知 500ms 的方案在 7m 处停住（间距 10m），无碰撞；
    # 感知 1300ms 的方案在 1400ms 前已撞上。
    assert not by_tick[500].collisions
    assert by_tick[1300].collisions
    assert by_tick[1300].collisions[0].pair == ("follower", "leader")
    conflict_ticks = {
        r.scheme.assignment_of("saw-lead-brake") for r in report.conflicts
    }
    clean_ticks = {r.scheme.assignment_of("saw-lead-brake") for r in report.clean}
    assert conflict_ticks == {1100, 1200, 1300}
    assert clean_ticks == {500, 600, 700, 800, 900, 1000}
    assert conflict_ticks.isdisjoint(clean_ticks)
    assert report.unexplored == ()  # 全量枚举时无未考察区间


def test_boundary_sampling_lists_unexplored_intervals():
    report = evaluate_lead_brake(samples={"saw-lead-brake": (500, 1300)})
    assert set(scheme_ticks(report)) == {500, 1300}
    # 600~1200 未考察，单独列出，不得被 500 的无碰撞结论覆盖
    assert len(report.unexplored) == 1
    assert report.unexplored[0].start_ms == 600
    assert report.unexplored[0].end_ms == 1200
    assert not report.is_continuous_safety_verified


def test_order_and_same_observation_constraints():
    first = UncertainPerception("see-signal", "follower", "signal_red", "j1", 400, 600, 100)
    second = UncertainPerception("see-brake", "follower", "lead_brake", "leader", 500, 700, 100)
    ordered = enumerate_schemes(
        [first, second], [PerceptionConstraint("order", "see-signal", "see-brake")]
    )
    pairs = {tuple(sorted(s.assignments)) for s in ordered}
    assert len(pairs) == 8  # 9 种组合排除 (600, 500)
    assert (("see-brake", 500), ("see-signal", 600)) not in pairs
    same = enumerate_schemes(
        [first, second],
        [PerceptionConstraint("same_observation", "see-signal", "see-brake")],
    )
    assert {tuple(sorted(s.assignments)) for s in same} == {
        (("see-brake", 500), ("see-signal", 500)),
        (("see-brake", 600), ("see-signal", 600)),
    }
    with pytest.raises(KeyError):
        enumerate_schemes([first], [PerceptionConstraint("order", "see-signal", "ghost")])


def test_same_tick_stimuli_both_fire():
    scenario = Scenario(
        name="same-tick",
        version="1.0.0",
        seed=1,
        vehicles=(
            Vehicle("follower", ((0.0, 0.0), (200.0, 0.0)), 10.0),
            Vehicle("leader", ((30.0, 0.0), (200.0, 0.0)), 10.0),
        ),
        signals=(Signal("j1", ((0, "green"), (500, "red"))),),
        events=(
            ScriptedEvent(
                "leader-brakes", 500, {"type": "stop_actor", "actor_id": "leader"}
            ),
        ),
        tick_ms=100,
    )
    uncertainties = [
        UncertainPerception("see-signal", "follower", "signal_red", "j1", 500, 500, 100),
        UncertainPerception("see-brake", "follower", "lead_brake", "leader", 500, 500, 100),
    ]
    report = evaluate_uncertainty(
        ReplayEngine(scenario),
        uncertainties,
        until_ms=900,
        at_ms=400,
        changes=brake_changes(),
    )
    (result,) = report.results
    ticks = sorted(p.tick_ms for p in result.perceptions)
    assert ticks == [500, 500]  # 同刻刺激都记录
    # 同车同刻两个制动响应：第二个取代第一个
    statuses = sorted(r.status for r in result.responses)
    assert statuses == ["executed", "superseded"]


def test_planned_perception_suppresses_auto_perception():
    scenario = Scenario(
        name="suppress",
        version="1.0.0",
        seed=1,
        vehicles=(Vehicle("car", ((0.0, 0.0), (100.0, 0.0)), 5.0),),
        signals=(Signal("j1", ((0, "green"), (300, "red"))),),
        reaction_delays={"car": {"signal_red": 100}},
        planned_perceptions=(
            PlannedPerception("p1", "car", "signal_red", "j1", 400, 100),
        ),
        tick_ms=100,
    )
    engine = ReplayEngine(scenario)
    engine.run_to(600)
    records = [p for p in engine.perception_records() if p.stimulus == "signal_red"]
    assert len(records) == 1
    assert records[0].perception_id == "planned:p1"
    assert records[0].tick_ms == 400


def test_explain_difference_attributes_causes():
    report = evaluate_lead_brake()
    by_tick = scheme_ticks(report)
    explanation = explain_difference(by_tick[500], by_tick[1300], 2000)
    assert explanation.causes == ("perception_choice",)
    assert explanation.first_divergence is not None
    assert explanation.first_divergence.tick_ms == 600  # 500 方案先开始制动
    assert explanation.separating_events

    slower = make_scheme(
        {"saw-lead-brake": 500},
        [saw_brake()],
        delays={"saw-lead-brake": 400},
        scheme_id="slower",
    )
    extended = append_scheme(
        report, ReplayEngine(lead_brake_scenario()), [saw_brake()], slower,
        until_ms=2000, at_ms=500, changes=brake_changes(),
    )
    delay_explanation = explain_difference(
        by_tick[500], extended.results[-1], 1500
    )
    assert delay_explanation.causes == ("reaction_delay",)


def test_explain_difference_signal_schedule_cause():
    engine = ReplayEngine(lead_brake_scenario())
    report_a = evaluate_uncertainty(
        engine,
        [UncertainPerception("see-signal", "follower", "signal_red", "j1", 500, 500, 100)],
        until_ms=900,
        at_ms=400,
        changes=brake_changes(),
    )
    # 同一感知选择、同一延迟：信号表一致时不误报差异来源
    alt = evaluate_uncertainty(
        ReplayEngine(lead_brake_scenario()),
        [UncertainPerception("see-signal", "follower", "signal_red", "j1", 500, 500, 100)],
        until_ms=900,
        at_ms=400,
        changes=BranchChanges(
            vehicle_reaction_params={"follower": {"brake_rate": 20.0}},
            extra_events=(
                ScriptedEvent(
                    "late-red", 700,
                    {"type": "set_signal", "signal_id": "jX", "phase": "red"},
                ),
            ),
        ),
    )
    explanation = explain_difference(report_a.results[0], alt.results[0], 900)
    assert explanation.causes == ()


def test_explain_difference_detects_signal_schedule_cause():
    uncertainties = [
        UncertainPerception("see-signal", "follower", "signal_red", "j1", 500, 500, 100)
    ]
    base = evaluate_uncertainty(
        ReplayEngine(lead_brake_scenario()), uncertainties,
        until_ms=900, at_ms=400, changes=brake_changes(),
    )
    rephased = evaluate_uncertainty(
        ReplayEngine(lead_brake_scenario()), uncertainties,
        until_ms=900, at_ms=400,
        changes=BranchChanges(
            vehicle_reaction_params={"follower": {"brake_rate": 20.0}},
            signal_plans={"j1": ((450, "red"),)},
        ),
    )
    explanation = explain_difference(base.results[0], rephased.results[0], 900)
    assert explanation.causes == ("signal_schedule",)


def test_append_scheme_preserves_old_results_and_updates_unexplored():
    uncertainties = [saw_brake()]
    report = evaluate_lead_brake(samples={"saw-lead-brake": (500, 1300)})
    old_perceptions = [
        (r.scheme.scheme_id, [p.tick_ms for p in r.perceptions])
        for r in report.results
    ]
    extra = make_scheme({"saw-lead-brake": 700}, uncertainties, scheme_id="near-conflict")
    updated = append_scheme(
        report,
        ReplayEngine(lead_brake_scenario()),
        uncertainties,
        extra,
        until_ms=1500,
        at_ms=500,
        changes=brake_changes(),
    )
    # 旧方案的感知事实不被回写
    assert [
        (r.scheme.scheme_id, [p.tick_ms for p in r.perceptions])
        for r in report.results
    ] == old_perceptions
    assert len(report.results) == 2
    assert len(updated.results) == 3
    assert updated.results[-1].scheme.scheme_id == "near-conflict"
    assert [(i.start_ms, i.end_ms) for i in updated.unexplored] == [
        (600, 600),
        (800, 1200),
    ]


def test_nested_branch_keeps_parent_perception_facts():
    report = evaluate_lead_brake()
    parent = scheme_ticks(report)[500]
    (parent_perception,) = parent.perceptions
    nested = evaluate_uncertainty(
        parent.engine,
        [UncertainPerception("see-again", "follower", "lead_brake", "leader", 900, 900, 100)],
        until_ms=1200,
        at_ms=800,
        changes=BranchChanges(),
    )
    (nested_result,) = nested.results
    parent_ticks = [
        p.tick_ms
        for p in nested_result.engine.perception_records()
        if p.perception_id == parent_perception.perception_id
    ]
    assert parent_ticks == [500]  # 父方案感知事实原样保留
    assert nested_result.engine.scenario.parent_id == parent.engine.scenario.branch_id


def test_exited_vehicle_pending_response_not_inherited():
    scenario = Scenario(
        name="exit-pending",
        version="1.0.0",
        seed=1,
        vehicles=(
            Vehicle("follower", ((0.0, 0.0), (200.0, 0.0)), 10.0),
            Vehicle("leader", ((30.0, 0.0), (200.0, 0.0)), 10.0),
        ),
        events=(
            ScriptedEvent(
                "leader-brakes", 400, {"type": "stop_actor", "actor_id": "leader"}
            ),
        ),
        reaction_delays={"follower": {"lead_brake": 300}},
        tick_ms=100,
    )
    engine = ReplayEngine(scenario)
    engine.run_to(400)  # 感知已发生（400ms），响应 700ms 才到期
    assert engine.pending_response_records()
    branch = branch_from(
        engine,
        "exit-branch",
        BranchChanges(
            extra_events=(
                ScriptedEvent(
                    "exit-now", 400, {"type": "exit_actor", "actor_id": "follower"}
                ),
            )
        ),
        at_ms=400,
    )
    branch.run_to(900)
    # 退出车辆的待响应事件不在派生方案中执行
    assert branch.state.pending_responses == []
    follower_responses = [
        r for r in branch.response_records() if r.vehicle_id == "follower"
    ]
    assert follower_responses
    assert all(r.status == "cancelled" for r in follower_responses)


def test_save_and_continue_between_perception_and_response(tmp_path):
    scenario = Scenario(
        name="pending-save",
        version="1.0.0",
        seed=1,
        vehicles=(
            Vehicle("follower", ((0.0, 0.0), (200.0, 0.0)), 10.0),
            Vehicle("leader", ((30.0, 0.0), (200.0, 0.0)), 10.0),
        ),
        events=(
            ScriptedEvent(
                "leader-brakes", 400, {"type": "stop_actor", "actor_id": "leader"}
            ),
        ),
        reaction_delays={"follower": {"lead_brake": 300}},
        tick_ms=100,
    )
    engine = ReplayEngine(scenario)
    engine.run_to(500)  # 感知已发生（400ms），响应 700ms 才到期
    assert engine.pending_response_records()
    path = save_run(engine, tmp_path / "run.json")
    restored = load_run(path)
    restored.run_to(900)
    (response,) = [
        r for r in restored.response_records() if r.status == "executed"
    ]
    assert response.trigger_tick_ms == 400
    assert response.due_tick_ms == 700


def test_planned_perceptions_survive_save_load(tmp_path):
    scenario = Scenario(
        name="planned-save",
        version="1.0.0",
        seed=1,
        vehicles=(Vehicle("car", ((0.0, 0.0), (100.0, 0.0)), 5.0),),
        planned_perceptions=(
            PlannedPerception("p1", "car", "lead_brake", "leader", 300, 100),
        ),
        tick_ms=100,
    )
    engine = ReplayEngine(scenario)
    engine.seek(100)
    path = save_run(engine, tmp_path / "run.json")
    restored = load_run(path)
    assert restored.scenario.planned_perceptions == scenario.planned_perceptions
    restored.run_to(500)
    ids = [p.perception_id for p in restored.perception_records()]
    assert ids == ["planned:p1"]
