import pytest

from scenario_replay import (
    BranchChanges,
    PerceptionWindow,
    ReplayEngine,
    Scenario,
    ScriptedEvent,
    Signal,
    Vehicle,
    WindowConstraint,
    adopted_perception_facts,
    attribute_difference,
    constraint_violations,
    check_assignment,
    derive_scheme,
    derive_schemes,
    enumerate_assignments,
    evaluate_schemes,
    compare_revision_candidates,
    confirm_revision_candidate,
    export_candidate_comparison,
    list_revision_candidates,
    load_revision_candidate,
    load_run,
    load_pending_revision,
    rederive_pending_revision,
    rederive_revision_candidate,
    save_run,
    stage_perception_revision,
    stage_perception_candidate,
    undo_revision_selection,
    selected_revision_candidate,
)


def uncertain_lead_brake_scenario() -> Scenario:
    """前车 400ms 制动停车；记录只能确定驾驶员在 400~800ms 内看到。

    人工可核对：leader 在 400ms 停于 x=16.0；follower 10m/s、
    brake_rate=20（每 tick 减速 2m/s，制动距离 2.0m），延迟 500ms
    时响应在 chosen+400ms 执行，最终停于 (chosen+300)/100 + 2.0。
    碰撞半径之和 4.0：chosen<=600 停于 9/10/11（间距>4 安全），
    chosen=700 恰好间距 4.0（边界，判碰撞），chosen=800 间距 3.6。
    """
    return Scenario(
        name="uncertain-lead-brake",
        version="1.0.0",
        seed=5,
        vehicles=(
            Vehicle("follower", ((0.0, 0.0), (200.0, 0.0)), 10.0),
            Vehicle("leader", ((12.0, 0.0), (200.0, 0.0)), 10.0),
        ),
        signals=(Signal("j1", ((0, "green"), (400, "red"))),),
        events=(
            ScriptedEvent(
                "leader-brakes",
                400,
                {"type": "stop_actor", "actor_id": "leader"},
            ),
        ),
        tick_ms=100,
    )


def brake_window(earliest=400, latest=800, delay=500) -> PerceptionWindow:
    return PerceptionWindow(
        "w-brake", "follower", "lead_brake", "leader",
        earliest, latest, delay,
        observed={"note": "记录只能给出范围"},
    )


def brake_base_changes() -> BranchChanges:
    return BranchChanges(vehicle_reaction_params={"follower": {"brake_rate": 20.0}})


def derive_all(latest=800):
    schemes = derive_schemes(
        ReplayEngine(uncertain_lead_brake_scenario()),
        "s",
        [brake_window(latest=latest)],
        base_changes=brake_base_changes(),
    )
    return schemes


def test_enumerate_assignments_respects_after_and_same_constraints():
    first = PerceptionWindow("w1", "follower", "lead_brake", "leader", 400, 500, 300)
    second = PerceptionWindow("w2", "follower", "signal_red", "j1", 400, 500, 300)

    all_assignments = enumerate_assignments([first, second], (), 100)
    assert len(all_assignments) == 4

    after = enumerate_assignments(
        [first, second], [WindowConstraint("after", "w1", "w2")], 100
    )
    assert [tuple(sorted(a.items())) for a in after] == [
        (("w1", 400), ("w2", 400)),
        (("w1", 400), ("w2", 500)),
        (("w1", 500), ("w2", 500)),
    ]

    same = enumerate_assignments(
        [first, second], [WindowConstraint("same", "w1", "w2")], 100
    )
    assert [tuple(sorted(a.items())) for a in same] == [
        (("w1", 400), ("w2", 400)),
        (("w1", 500), ("w2", 500)),
    ]

    with pytest.raises(ValueError):
        check_assignment(
            [first, second],
            [WindowConstraint("after", "w1", "w2")],
            {"w1": 500, "w2": 400},
            100,
        )
    with pytest.raises(ValueError):
        check_assignment([first], (), {"w1": 600}, 100)
    with pytest.raises(ValueError):
        enumerate_assignments(
            [first, second], [WindowConstraint("after", "w1", "nope")], 100
        )


def test_derive_schemes_cover_range_and_report_separates_conflicts():
    schemes = derive_all()
    assert len(schemes) == 5
    assert [s.assignment["w-brake"] for s in schemes] == [400, 500, 600, 700, 800]

    report = evaluate_schemes(schemes, 2500, total_assignments=5)
    assert report.covers_full_range
    # 已发现冲突与未见冲突的方案分开列出
    assert [e.scheme_id for e in report.conflicts] == ["s-3", "s-4"]
    assert [e.scheme_id for e in report.clears] == ["s-0", "s-1", "s-2"]
    # 边界时刻：chosen=700 时最小间距恰好等于碰撞半径之和
    boundary = report.conflicts[0]
    assert boundary.assignment == (("w-brake", 700),)
    assert boundary.collisions[0].distance == pytest.approx(4.0)
    # 每个方案保留实际采用的感知时刻与反应延迟
    for evaluation in report.evaluations:
        assert len(evaluation.adopted) == 1
        adopted = evaluation.adopted[0]
        assert adopted.tick_ms == dict(evaluation.assignment)["w-brake"]
        assert adopted.delay_ms == 500
    # 离散对照不显示成连续安全结论
    conclusion = report.conclusion()
    assert "离散" in conclusion
    assert "不构成连续时间安全证明" in conclusion


def test_single_clear_scheme_does_not_imply_range_safe():
    schemes = derive_all()
    clears_only = [s for s in schemes if s.assignment["w-brake"] == 400]
    report = evaluate_schemes(clears_only, 2500, total_assignments=5)
    assert not report.covers_full_range
    assert report.unexplored_count == 4
    conclusion = report.conclusion()
    assert "尚未考察" in conclusion
    assert "不能据此判定范围安全" in conclusion
    # 只考察无冲突方案时，结论不得宣称范围安全
    assert "无碰撞" not in conclusion


def test_same_tick_stimuli_and_boundary_window_injection():
    scenario = uncertain_lead_brake_scenario()
    signal_window = PerceptionWindow(
        "w-signal", "follower", "signal_red", "j1", 400, 400, 200
    )
    brake = brake_window(latest=400)  # 边界：范围退化为单一时刻
    scheme = derive_scheme(
        ReplayEngine(scenario),
        "same-tick",
        [signal_window, brake],
        {"w-signal": 400, "w-brake": 400},
        base_changes=brake_base_changes(),
    )
    engine = scheme.engine
    engine.run_to(600)
    perceptions = [p for p in engine.perceptions if p.observed.get("injected")]
    # 同刻刺激按窗口顺序确定先后
    assert [p.stimulus for p in perceptions] == ["signal_red", "lead_brake"]
    assert all(p.tick_ms == 400 for p in perceptions)
    assert perceptions[0].supersedes_response_id is None
    assert perceptions[1].supersedes_response_id == perceptions[0].response_id
    pending = engine.pending_response_records()
    assert len(pending) == 1
    assert pending[0].stimulus == "lead_brake"
    assert pending[0].due_tick_ms == 900


def test_append_scheme_near_conflict_and_range_edit_keeps_old_facts():
    schemes = derive_all(latest=600)
    assert [s.assignment["w-brake"] for s in schemes] == [400, 500, 600]
    before = {
        s.scheme_id: adopted_perception_facts(s.engine) for s in schemes
    }

    # 在已有冲突附近追加具体方案：范围扩到 800 后单独派生 chosen=800
    wider = brake_window(latest=800)
    appended = derive_scheme(
        ReplayEngine(uncertain_lead_brake_scenario()),
        "s-extra",
        [wider],
        {"w-brake": 800},
        base_changes=brake_base_changes(),
    )
    appended.engine.run_to(2500)
    assert appended.engine.collisions

    # 修改不确定范围不回写旧方案的感知事实
    for scheme in schemes:
        assert adopted_perception_facts(scheme.engine) == before[scheme.scheme_id]
    assert adopted_perception_facts(appended.engine)[0].tick_ms == 800


def test_derive_scheme_rejects_constraint_violation_and_auto_delay_conflict():
    w1 = PerceptionWindow("w1", "follower", "lead_brake", "leader", 400, 500, 300)
    w2 = PerceptionWindow("w2", "follower", "signal_red", "j1", 400, 500, 300)
    constraints = [WindowConstraint("after", "w1", "w2")]
    with pytest.raises(ValueError):
        derive_scheme(
            ReplayEngine(uncertain_lead_brake_scenario()),
            "bad",
            [w1, w2],
            {"w1": 500, "w2": 400},
            constraints=constraints,
        )

    # 基础方案已对 lead_brake 配置自动响应延迟：不允许重复注入
    configured = ReplayEngine(uncertain_lead_brake_scenario())
    configured.state.reaction_delays["follower"] = {"lead_brake": 300}
    with pytest.raises(ValueError):
        derive_scheme(
            configured, "dup", [w1], {"w1": 400}
        )


def test_attribute_difference_distinguishes_choice_delay_and_signal_plan():
    schemes = derive_all()
    early, late = schemes[0].engine, schemes[-1].engine

    # 仅感知时刻不同
    attr = attribute_difference(early, late, 2500)
    assert attr.causes == ("perception_choice",)
    assert attr.divergence is not None
    assert attr.divergence.tick_ms == 800  # 早感知方案 800ms 开始制动
    assert "positions" in attr.divergence.aspects
    assert attr.first_events_a  # 可回到首次使结果分开的交通事件

    # 仅反应延迟不同（同一感知时刻）
    quick = derive_scheme(
        ReplayEngine(uncertain_lead_brake_scenario()),
        "quick",
        [brake_window(latest=400, delay=300)],
        {"w-brake": 400},
        base_changes=brake_base_changes(),
    )
    slow = derive_scheme(
        ReplayEngine(uncertain_lead_brake_scenario()),
        "slow",
        [brake_window(latest=400, delay=500)],
        {"w-brake": 400},
        base_changes=brake_base_changes(),
    )
    attr = attribute_difference(quick.engine, slow.engine, 2500)
    assert attr.causes == ("reaction_delay",)

    # 仅后续信号安排不同
    red_early = derive_scheme(
        ReplayEngine(uncertain_lead_brake_scenario()),
        "red-early",
        [brake_window(latest=400)],
        {"w-brake": 400},
        base_changes=BranchChanges(
            vehicle_reaction_params={"follower": {"brake_rate": 20.0}},
            signal_plans={"j1": ((700, "green"),)},
        ),
    )
    red_late = derive_scheme(
        ReplayEngine(uncertain_lead_brake_scenario()),
        "red-late",
        [brake_window(latest=400)],
        {"w-brake": 400},
        base_changes=BranchChanges(
            vehicle_reaction_params={"follower": {"brake_rate": 20.0}},
            signal_plans={"j1": ((900, "green"),)},
        ),
    )
    # 信号在 400ms 已转红；两个方案分别安排 700/900ms 恢复绿灯
    attr = attribute_difference(red_early.engine, red_late.engine, 2500)
    assert attr.causes == ("signal_plan",)
    assert attr.divergence.tick_ms == 700
    assert attr.divergence.signals == ("j1",)


def test_exited_vehicle_pending_response_not_carried_into_scheme():
    engine = ReplayEngine(uncertain_lead_brake_scenario())
    engine.state.reaction_delays["follower"] = {"lead_brake": 800}
    engine.run_to(400)
    assert engine.pending_response_records()
    # 车辆退出（等价于 exit_actor 后的状态）：待响应事件不进入派生方案
    engine.state.active_actors["follower"] = False
    scheme = derive_scheme(
        engine,
        "after-exit",
        [PerceptionWindow("w-signal", "leader", "signal_red", "j1", 400, 400, 200)],
        {"w-signal": 400},
    )
    assert all(
        response.vehicle_id != "follower"
        for response in scheme.engine.pending_response_records()
    )
    scheme.engine.run_to(1500)
    assert all(
        not (r.vehicle_id == "follower" and r.status == "executed")
        for r in scheme.engine.responses
    )


def test_save_and_reload_mid_pending_continues(tmp_path):
    # 感知已发生而动作尚未响应时保存；恢复后仍在原到期时刻执行
    scheme = derive_scheme(
        ReplayEngine(uncertain_lead_brake_scenario()),
        "pending",
        [brake_window(latest=400)],
        {"w-brake": 400},
        base_changes=brake_base_changes(),
    )
    engine = scheme.engine
    engine.run_to(600)
    assert len(engine.pending_response_records()) == 1
    path = save_run(engine, tmp_path / "pending.json")

    reopened = load_run(path)
    pending = reopened.pending_response_records()
    assert len(pending) == 1
    assert pending[0].due_tick_ms == 900
    reopened.run_to(1200)
    executed = [r for r in reopened.responses if r.status == "executed"]
    assert len(executed) == 1
    assert executed[0].due_tick_ms == 900
    # 回跳再继续：感知与响应不重复登记
    reopened.seek(400)
    reopened.run_to(1200)
    assert len([r for r in reopened.responses if r.status == "executed"]) == 1
    injected = [p for p in reopened.perceptions if p.observed.get("injected")]
    assert len(injected) == 1
    assert adopted_perception_facts(reopened)[0].tick_ms == 400


def test_nested_derivation_from_scheme_keeps_event_ownership():
    schemes = derive_all(latest=400)
    parent = schemes[0].engine
    parent.run_to(600)
    # 在派生方案内再次派生：新的不确定感知窗口
    nested_window = PerceptionWindow(
        "w-nested", "follower", "signal_red", "j1", 600, 700, 200
    )
    nested = derive_schemes(parent, "n", [nested_window])
    assert len(nested) == 2
    for scheme in nested:
        engine = scheme.engine
        assert engine.scenario.parent_id == "s-0"
        # 父方案在分支点之后的事件不带入嵌套方案
        assert all(
            record.tick_ms <= 600 or record.event_id.startswith(scheme.scheme_id)
            for record in engine.journal
        )
        engine.run_to(1000)
        # 分支点之前的注入感知是共享事实，不重复登记
        injected = [p for p in engine.perceptions if p.observed.get("injected")]
        brake_facts = [p for p in injected if p.stimulus == "lead_brake"]
        assert len(brake_facts) == 1
        assert brake_facts[0].tick_ms == 400
        nested_facts = [p for p in injected if p.stimulus == "signal_red"]
        assert len(nested_facts) == 1
        assert nested_facts[0].tick_ms == scheme.assignment["w-nested"]


def test_constraint_violations_are_separate_from_actual_collisions():
    stop_action = {"type": "set_speed", "speed": 0.0, "rate": 20.0}
    first = PerceptionWindow(
        "w1", "follower", "signal_red", "j1", 400, 500, 900,
        response_action=stop_action,
    )
    second = PerceptionWindow(
        "w2", "follower", "lead_brake", "leader", 400, 500, 900,
        response_action=stop_action,
    )
    constraints = [WindowConstraint("same", "w1", "w2")]

    violations = constraint_violations([first, second], constraints, 100)
    assert {dict(v.assignment)["w1"] for v in violations} == {400, 500}
    assert all(v.violated_constraints == tuple(constraints) for v in violations)
    assert "非碰撞" in violations[0].reason

    schemes = derive_schemes(
        ReplayEngine(uncertain_lead_brake_scenario()),
        "same",
        [first, second],
        constraints=constraints,
        base_changes=brake_base_changes(),
    )
    assert len(schemes) == 2
    report = evaluate_schemes(
        schemes,
        2500,
        total_assignments=4,
        constraint_violations_found=violations,
    )
    assert report.unexplored_count == 0
    assert len(report.constraint_violations) == 2
    assert all(evaluation.collisions for evaluation in report.conflicts)
    conclusion = report.conclusion()
    assert "不满足刺激约束（未派生、非碰撞）" in conclusion
    assert str(len(report.conflicts)) in conclusion


def test_revision_is_staged_until_reopened_and_user_rederives(tmp_path):
    original = derive_scheme(
        ReplayEngine(uncertain_lead_brake_scenario()),
        "old",
        [brake_window(400, 400, 500)],
        {"w-brake": 400},
        base_changes=brake_base_changes(),
    ).engine
    original.run_to(600)
    old_pending = original.pending_response_records()[0]
    assert old_pending.response_id == "injected-response:old:perceive:w-brake"

    stage_perception_revision(
        original,
        [brake_window(600, 800, 500)],
        base_changes=brake_base_changes(),
        scheme_prefix="new",
        horizon_ms=2500,
    )
    assert original.pending_response_records()[0].response_id == old_pending.response_id
    assert not any(event.event_id.startswith("new:") for event in original.scenario.events)
    path = save_run(original, tmp_path / "staged.json")

    reopened = load_run(path)
    staged = load_pending_revision(reopened)
    assert staged is not None
    assert staged.windows[0].earliest_ms == 600
    assert reopened.pending_response_records()[0].response_id == old_pending.response_id

    result = rederive_pending_revision(reopened, evaluate_until_ms=2500)
    assert len(result.schemes) == 3
    assert [scheme.assignment["w-brake"] for scheme in result.schemes] == [600, 700, 800]
    assert reopened.pending_revision is None
    assert not result.constraint_violations
    assert result.report is not None
    assert [evaluation.scheme_id for evaluation in result.report.conflicts] == [
        "new-1",
        "new-2",
    ]

    old_engine = reopened
    new_engine = result.schemes[0].engine
    old_engine.run_to(900)
    new_engine.seek(900)
    assert old_engine.pending_response_records() == ()
    assert [p.tick_ms for p in new_engine.perceptions if p.observed.get("injected")] == [600]
    assert [
        r.response_id for r in new_engine.responses if r.status == "executed"
    ] == ["injected-response:new-0:perceive:w-brake"]
    assert not any(
        response.response_id == old_pending.response_id
        for response in new_engine.responses
    )

    attr = attribute_difference(old_engine, new_engine, 2500)
    assert "perception_choice" in attr.causes
    assert attr.divergence is not None
    assert attr.divergence.tick_ms == 800
    assert attr.fact_divergence is not None
    assert attr.fact_divergence.tick_ms == 400
    assert attr.first_events_a and attr.first_events_b


def test_staged_revision_reports_invalid_and_collision_separately_after_reload(tmp_path):
    stop_action = {"type": "set_speed", "speed": 0.0, "rate": 20.0}
    first = PerceptionWindow(
        "w1", "follower", "signal_red", "j1", 400, 500, 900,
        response_action=stop_action,
    )
    second = PerceptionWindow(
        "w2", "follower", "lead_brake", "leader", 400, 500, 900,
        response_action=stop_action,
    )
    constraints = [WindowConstraint("same", "w1", "w2")]
    engine = ReplayEngine(uncertain_lead_brake_scenario())
    stage_perception_revision(
        engine,
        [first, second],
        constraints,
        base_changes=brake_base_changes(),
        scheme_prefix="revised",
    )
    path = save_run(engine, tmp_path / "invalid.json")

    result = rederive_pending_revision(load_run(path), evaluate_until_ms=2500)
    assert len(result.schemes) == 2
    assert len(result.constraint_violations) == 2
    assert result.report is not None
    assert result.report.total_assignments == 4
    assert len(result.report.constraint_violations) == 2
    assert {evaluation.scheme_id for evaluation in result.report.conflicts} == {"revised-1"}
    assert all(evaluation.collisions for evaluation in result.report.conflicts)
    assert "非碰撞" in result.report.conclusion()


def _two_shared_window_candidates(engine):
    """两份候选共享同一窗口，但采用不同刺激先后约束和反应延迟。"""
    stop_action = {"type": "set_speed", "speed": 0.0, "rate": 20.0}
    first = PerceptionWindow(
        "w1", "follower", "signal_red", "j1", 400, 500, 900,
        response_action=stop_action,
    )
    second = PerceptionWindow(
        "w2", "follower", "lead_brake", "leader", 400, 500, 900,
        response_action=stop_action,
    )
    late_second = PerceptionWindow(
        "w2", "follower", "lead_brake", "leader", 500, 700, 500,
        response_action=stop_action,
    )
    stage_perception_candidate(
        engine,
        "same-time",
        [first, second],
        [WindowConstraint("same", "w1", "w2")],
        base_changes=brake_base_changes(),
        horizon_ms=2500,
    )
    stage_perception_candidate(
        engine,
        "brake-after-signal",
        [first, late_second],
        [WindowConstraint("after", "w2", "w1")],
        base_changes=brake_base_changes(),
        horizon_ms=2500,
    )
    return first, second


def test_multiple_candidates_coexist_without_overwriting_each_other(tmp_path):
    engine = ReplayEngine(uncertain_lead_brake_scenario())
    first, second = _two_shared_window_candidates(engine)

    assert [candidate.candidate_id for candidate in list_revision_candidates(engine)] == [
        "brake-after-signal",
        "same-time",
    ]
    # 再暂存第三份候选不影响前两份事实。
    stage_perception_revision(
        engine,
        [brake_window(600, 800, 500)],
        candidate_id="pending",
        base_changes=brake_base_changes(),
        scheme_prefix="legacy",
        horizon_ms=2500,
    )
    same_time = load_revision_candidate(engine, "same-time")
    assert [window.delay_ms for window in same_time.windows] == [900, 900]
    assert [constraint.kind for constraint in same_time.constraints] == ["same"]

    comparison = compare_revision_candidates(
        [
            load_revision_candidate(engine, "same-time"),
            load_revision_candidate(engine, "brake-after-signal"),
        ]
    )
    assert comparison.shared_windows == ("w1", "w2")
    assert any("w2 反应延迟" in text for text in comparison.delay_differences)
    assert comparison.ordering_differences
    assert "同刻" in comparison.summary() or "独有" in comparison.summary()

    path = save_run(engine, tmp_path / "candidates.json")
    reopened = load_run(path)
    assert [candidate.candidate_id for candidate in list_revision_candidates(reopened)] == [
        "brake-after-signal",
        "pending",
        "same-time",
    ]
    assert reopened.pending_revision is not None
    assert load_pending_revision(reopened).candidate_id == "pending"


def test_select_one_candidate_then_undo_and_select_another_keeps_facts(tmp_path):
    engine = ReplayEngine(uncertain_lead_brake_scenario())
    _two_shared_window_candidates(engine)

    first_choice = confirm_revision_candidate(engine, "same-time", evaluate_until_ms=2500)
    assert selected_revision_candidate(engine) == "same-time"
    first_scheme_ids = [scheme.scheme_id for scheme in first_choice.schemes]
    first_event_ids = [
        record.event_id
        for scheme in first_choice.schemes
        for record in scheme.engine.journal
        if record.action.get("type") == "inject_perception"
    ]

    # 重复确认同一候选幂等：事件身份不变，两份其它候选与原方案仍保留。
    repeated = confirm_revision_candidate(engine, "same-time", evaluate_until_ms=2500)
    assert [scheme.scheme_id for scheme in repeated.schemes] == first_scheme_ids
    assert selected_revision_candidate(engine) == "same-time"
    assert {candidate.candidate_id for candidate in list_revision_candidates(engine)} == {
        "same-time",
        "brake-after-signal",
    }

    undone = undo_revision_selection(engine)
    assert undone == "same-time"
    assert selected_revision_candidate(engine) is None
    other = confirm_revision_candidate(
        engine, "brake-after-signal", evaluate_until_ms=2500
    )
    assert selected_revision_candidate(engine) == "brake-after-signal"
    other_event_ids = [
        record.event_id
        for scheme in other.schemes
        for record in scheme.engine.journal
        if record.action.get("type") == "inject_perception"
    ]
    # 两次选择的事件身份使用各自稳定前缀；改选不借用先选候选的事件。
    assert not set(first_event_ids) & set(other_event_ids)
    assert all(event_id.startswith("brake-after-signal") for event_id in other_event_ids)
    # 首次分歧仍可追溯：同刻候选只派生同刻组合，先后候选只派生合法先后组合。
    assert len(first_choice.constraint_violations) == 2
    assert len(other.constraint_violations) == 5
    assert len(other.schemes) == 1

    path = save_run(engine, tmp_path / "selected.json")
    reopened = load_run(path)
    assert selected_revision_candidate(reopened) == "brake-after-signal"
    assert undo_revision_selection(reopened) == "brake-after-signal"
    assert selected_revision_candidate(reopened) is None


def test_candidate_comparison_export_separates_exclusion_collision_explored_unverified():
    engine = ReplayEngine(uncertain_lead_brake_scenario())
    _two_shared_window_candidates(engine)

    # 不给定时域：可以比较输入，但碰撞/已探索/未验证范围明确标为尚未评估。
    pending = export_candidate_comparison(engine)
    pending_data = pending.export()
    assert pending_data["candidate_ids"] == ["brake-after-signal", "same-time"]
    assert all(
        candidate["actual_collisions"] == []
        and candidate["explored_discrete_assignments"] == 0
        and candidate["unexplored_assignments"] is None
        for candidate in pending_data["candidates"]
    )
    assert "尚未重放评估" in pending.narrative()

    report = export_candidate_comparison(engine, evaluate_until_ms=2500)
    data = report.export()
    by_id = {candidate["candidate_id"]: candidate for candidate in data["candidates"]}
    same_time = by_id["same-time"]
    brake_after = by_id["brake-after-signal"]
    # 约束排除不是实际碰撞：同刻候选排除 2 个组合，碰撞方案另行计数。
    assert len(same_time["constraint_excluded"]) == 2
    assert same_time["explored_discrete_assignments"] == 2
    assert same_time["unexplored_assignments"] == 0
    assert {
        collision["scheme_id"]
        for collision in same_time["actual_collisions"]
    } == {"same-time-1"}
    assert len(brake_after["constraint_excluded"]) == 5
    assert brake_after["explored_discrete_assignments"] == 1
    assert brake_after["unexplored_assignments"] == 0
    assert brake_after["actual_collisions"] == []
    narrative = report.narrative()
    assert "约束排除 2 个离散组合（未派生、非实际碰撞）" in narrative
    assert "实际碰撞方案" in narrative
    assert "未验证组合" in narrative
    # 导出不改变选择状态、不删除候选。
    assert selected_revision_candidate(engine) is None
    assert len(list_revision_candidates(engine)) == 2


def test_rederive_candidate_is_idempotent_and_does_not_touch_other_candidates():
    engine = ReplayEngine(uncertain_lead_brake_scenario())
    stage_perception_candidate(
        engine,
        "a",
        [brake_window(400, 500, 500)],
        base_changes=brake_base_changes(),
        horizon_ms=2500,
    )
    stage_perception_candidate(
        engine,
        "b",
        [brake_window(700, 800, 500)],
        base_changes=brake_base_changes(),
        horizon_ms=2500,
    )
    first = rederive_revision_candidate(engine, "a", evaluate_until_ms=2500)
    second = rederive_revision_candidate(engine, "a", evaluate_until_ms=2500)
    assert [scheme.scheme_id for scheme in first.schemes] == [
        scheme.scheme_id for scheme in second.schemes
    ]
    assert selected_revision_candidate(engine) is None
    candidate_b = load_revision_candidate(engine, "b")
    assert [window.earliest_ms for window in candidate_b.windows] == [700]
    assert candidate_b.candidate_id == "b"
