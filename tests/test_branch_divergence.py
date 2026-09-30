from scenario_replay import (
    BranchChanges,
    Pedestrian,
    ReplayEngine,
    Scenario,
    ScriptedEvent,
    Signal,
    Vehicle,
    branch_from,
    compare_at,
    first_divergence,
    load_run,
    save_run,
)


def timeline_scenario() -> Scenario:
    return Scenario(
        name="timeline",
        version="1.0.0",
        seed=3,
        vehicles=(Vehicle("car1", ((0.0, 0.0), (100.0, 0.0)), 10.0),),
        pedestrians=(Pedestrian("ped1", ((50.0, -10.0), (50.0, 10.0)), 1.0),),
        signals=(
            Signal("j1", ((0, "green"),)),
            Signal("j2", ((0, "green"),)),
        ),
        events=(
            ScriptedEvent("e100", 100, {"type": "set_speed", "actor_id": "car1", "speed": 5.0}),
            ScriptedEvent("e300", 300, {"type": "set_signal", "signal_id": "j1", "phase": "red"}),
            ScriptedEvent("e500", 500, {"type": "set_speed", "actor_id": "car1", "speed": 1.0}),
            ScriptedEvent("e600", 600, {"type": "set_signal", "signal_id": "j2", "phase": "yellow"}),
        ),
        tick_ms=100,
    )


def test_branch_does_not_inherit_parent_events_after_branch_point():
    parent = ReplayEngine(timeline_scenario())
    parent.run_to(800)  # 父场景已跑过分支点之后

    branch = branch_from(parent, "b1", BranchChanges(), at_ms=300)
    # 分支点以前的 e100/e300 保留；父场景之后的 e500 不带入分支历史
    assert branch.fired_event_ids() == ("e100", "e300")

    branch.run_to(800)
    # e500/e600 在分支时间线上首次触发，只出现一次
    assert branch.fired_event_ids().count("e500") == 1
    assert branch.fired_event_ids() == ("e100", "e300", "e500", "e600")


def test_branch_seek_replay_does_not_reaccumulate_events():
    parent = ReplayEngine(timeline_scenario())
    parent.run_to(800)
    branch = branch_from(parent, "b1", BranchChanges(), at_ms=300)
    branch.run_to(800)
    journal_before = list(branch.journal)

    # 向后跳转再重放：事件不重复累计，位置与首次一致
    branch.seek(100)
    branch.run_to(800)
    assert [(r.tick_ms, r.event_id) for r in branch.journal] == [
        (r.tick_ms, r.event_id) for r in journal_before
    ]
    assert branch.fired_event_ids().count("e100") == 1
    assert branch.fired_event_ids().count("e300") == 1
    assert branch.fired_event_ids().count("e500") == 1
    assert branch.fired_event_ids().count("e600") == 1

    fresh = ReplayEngine(branch.scenario)
    fresh.run_to(800)
    assert branch.positions() == fresh.positions()
    assert branch.fired_event_ids() == fresh.fired_event_ids()


def test_event_exactly_at_branch_point_belongs_to_shared_prefix():
    parent = ReplayEngine(timeline_scenario())
    parent.run_to(800)
    branch = branch_from(parent, "b1", BranchChanges(), at_ms=300)
    # 恰好发生在分支点的 e300 属于共同前缀：只记一次，seek 后也不重复
    assert branch.fired_event_ids().count("e300") == 1
    branch.seek(0)
    branch.run_to(800)
    assert branch.fired_event_ids().count("e300") == 1


def test_branch_within_branch_keeps_only_own_prefix(tmp_path):
    parent = ReplayEngine(timeline_scenario())
    parent.run_to(800)
    first = branch_from(parent, "b1", BranchChanges(
        extra_events=(ScriptedEvent(
            "b1-event", 400, {"type": "set_speed", "actor_id": "car1", "speed": 2.0}),),
    ), at_ms=300)
    first.run_to(700)
    assert first.fired_event_ids() == ("e100", "e300", "b1-event", "e500", "e600")

    second = branch_from(first, "b2", BranchChanges(), at_ms=500)
    # b2 的共同前缀包含 b1 在 500 之前（含）新增的事件，但不包含之后的
    assert second.fired_event_ids() == ("e100", "e300", "b1-event", "e500")
    second.run_to(700)
    assert second.fired_event_ids().count("e500") == 1
    assert second.fired_event_ids() == ("e100", "e300", "b1-event", "e500", "e600")

    # 保存重开后事件归属不变
    path = save_run(second, tmp_path / "b2.json")
    reopened = load_run(path)
    assert reopened.fired_event_ids() == second.fired_event_ids()
    reopened.seek(100)
    reopened.run_to(700)
    assert reopened.fired_event_ids() == second.fired_event_ids()
    assert reopened.scenario.parent_id == "b1"


def test_multiple_branches_same_point_first_divergence():
    parent = ReplayEngine(timeline_scenario())
    parent.run_to(300)

    # 分支 A：只替换车辆后续路线
    branch_a = branch_from(parent, "route", BranchChanges(
        vehicle_routes={"car1": ((2.0, 0.0), (2.0, 50.0))},
    ), at_ms=300)
    # 分支 B：只替换一组信号灯安排
    branch_b = branch_from(parent, "signals", BranchChanges(
        signal_plans={"j2": ((400, "yellow"), (600, "green"))},
    ), at_ms=300)

    for branch in (branch_a, branch_b):
        branch.run_to(800)

    # 分支点处三者一致
    assert compare_at(branch_a, branch_b, 300).positions_match
    # 车辆改道从分支点后第一个 tick 开始出现位置差异
    divergence = first_divergence(branch_a, branch_b, until_ms=800)
    assert divergence is not None
    assert divergence.tick_ms == 400
    assert "positions" in divergence.aspects
    assert "signals" in divergence.aspects

    # 信号分支在 400ms 起 j2 相位不同（分支点之后的 j2 计划事件被替换）
    branch_a.seek(400)
    branch_b.seek(400)
    assert branch_a.signal_phase("j2") == "green"
    assert branch_b.signal_phase("j2") == "yellow"
    assert branch_a.fired_event_ids().count("e600") == 1
    assert "e600" not in branch_b.fired_event_ids()


def test_vehicle_route_branch_save_reload(tmp_path):
    parent = ReplayEngine(timeline_scenario())
    parent.run_to(300)
    branch = branch_from(parent, "route", BranchChanges(
        vehicle_routes={"car1": ((2.0, 0.0), (2.0, 50.0))},
    ), at_ms=300)
    branch.run_to(600)
    path = save_run(branch, tmp_path / "route.json")

    reopened = load_run(path)
    assert reopened.fired_event_ids() == branch.fired_event_ids()
    reopened.run_to(800)
    branch.run_to(800)
    assert reopened.positions() == branch.positions()
    assert reopened.fired_event_ids() == branch.fired_event_ids()
    # 改道后车辆沿 y 轴行驶
    assert reopened.position_of("car1")[1] > 0.0
