import pytest

from scenario_replay import (
    BranchChanges,
    PerceptionWindow,
    RevisionResult,
    ReplayEngine,
    Scenario,
    ScriptedEvent,
    Signal,
    Vehicle,
    WindowConstraint,
    adopted_perception_facts,
    attribute_difference,
    check_assignment,
    derive_scheme,
    derive_schemes,
    derive_revised_schemes,
    enumerate_assignments,
    evaluate_schemes,
    load_run,
    rederive_revision,
    request_revision,
    save_run,
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


def test_revision_saved_reopened_is_not_adopted_until_user_rederives(tmp_path):
    scheme = derive_scheme(
        ReplayEngine(uncertain_lead_brake_scenario()),
        "original",
        [brake_window(earliest=400, latest=400, delay=500)],
        {"w-brake": 400},
        base_changes=brake_base_changes(),
    ).engine
    scheme.run_to(600)
    original_pending_id = scheme.pending_response_records()[0].response_id
    original_due = scheme.pending_response_records()[0].due_tick_ms

    request_revision(
        scheme,
        [brake_window(earliest=700, latest=800, delay=500)],
        revision_id="later-range",
    )
    # 保存重开前后，修订都只是待确认输入：旧感知事实和待响应动作保持不变
    assert scheme.pending_revision.status == "pending"
    assert adopted_perception_facts(scheme)[0].tick_ms == 400
    assert scheme.pending_response_records()[0].response_id == original_pending_id
    path = save_run(scheme, tmp_path / "revision-pending.json")

    reopened = load_run(path)
    assert reopened.pending_revision.status == "pending"
    assert adopted_perception_facts(reopened)[0].tick_ms == 400
    assert reopened.pending_response_records()[0].response_id == original_pending_id
    assert reopened.pending_response_records()[0].due_tick_ms == original_due

    result = rederive_revision(reopened, scheme_prefix="revised")
    assert reopened.pending_revision.status == "adopted"
    # 原方案仍保留原采用事实；新方案不能重复或借用旧响应事实
    assert adopted_perception_facts(reopened)[0].tick_ms == 400
    assert reopened.pending_response_records()[0].response_id == original_pending_id
    assert [s.assignment["w-brake"] for s in result.schemes] == [700, 800]
    revised = result.schemes[0].engine
    revised.run_to(700)
    revised_facts = adopted_perception_facts(revised)
    assert len(revised_facts) == 1
    assert revised_facts[0].tick_ms == 700
    pending = revised.pending_response_records()
    assert len(pending) == 1
    assert pending[0].perception_id.startswith("injected-perception:revised-0")
    assert pending[0].due_tick_ms == 1200


def test_constraint_violations_are_reported_separately_from_collisions():
    signal_window = PerceptionWindow(
        "w-signal", "follower", "signal_red", "j1", 700, 800, 1000
    )
    brake_window = PerceptionWindow(
        "w-brake", "follower", "lead_brake", "leader", 600, 800, 100
    )
    constraints = [WindowConstraint("after", "w-signal", "w-brake")]

    result = derive_revised_schemes(
        ReplayEngine(uncertain_lead_brake_scenario()),
        "ordered",
        [signal_window, brake_window],
        constraints,
        base_changes=brake_base_changes(),
    )
    assert isinstance(result, RevisionResult)
    assert [s.assignment["w-signal"] for s in result.schemes] == [700, 700, 800]
    assert [s.assignment["w-brake"] for s in result.schemes] == [700, 800, 800]
    assert len(result.constraint_violations) == 3
    assert all(
        "after" in violation.reason for violation in result.constraint_violations
    )

    report = evaluate_schemes(
        result.schemes, 2500, total_assignments=6,
        constraint_violations=result.constraint_violations,
    )
    assert report.explored_count == 3
    assert report.examined_count == 6
    assert report.unexplored_count == 0
    assert len(report.constraint_violations) == 3
    conclusion = report.conclusion()
    assert "3 个感知组合不满足约束" in conclusion
    assert "约束失败，不等同于交通碰撞" in conclusion


def test_revision_can_change_reaction_delay_and_traces_first_divergence():
    original = derive_scheme(
        ReplayEngine(uncertain_lead_brake_scenario()),
        "quick-original",
        [brake_window(earliest=400, latest=400, delay=500)],
        {"w-brake": 400},
        base_changes=brake_base_changes(),
    ).engine
    original.run_to(600)
    request_revision(
        original,
        [brake_window(earliest=400, latest=400, delay=700)],
        revision_id="slower-response",
    )
    result = rederive_revision(original, scheme_prefix="slow-revised")
    revised = result.schemes[0].engine

    assert adopted_perception_facts(revised)[0].delay_ms == 700
    revised.run_to(900)
    pending = revised.pending_response_records()
    assert len(pending) == 1
    assert pending[0].due_tick_ms == 1100
    assert original.pending_response_records()[0].due_tick_ms == 900
    attr = attribute_difference(original, revised, 1200)
    assert attr.causes == ("reaction_delay",)
    assert attr.divergence.tick_ms == 800
    assert attr.first_events_a
