import pytest

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
    first_response_comparison,
    load_run,
    save_run,
)


def delayed_lead_brake_scenario() -> Scenario:
    return Scenario(
        name="reaction-delay",
        version="3.0.0",
        seed=11,
        vehicles=(
            Vehicle("follower", ((0.0, 0.0), (100.0, 0.0)), 10.0),
            Vehicle("leader", ((8.0, 0.0), (100.0, 0.0)), 10.0),
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


def delay_branch(delay_ms: int, branch_id: str) -> ReplayEngine:
    source = ReplayEngine(delayed_lead_brake_scenario())
    return branch_from(
        source,
        branch_id,
        BranchChanges(
            vehicle_reaction_params={"follower": {"brake_rate": 20.0}},
            vehicle_reaction_delays={"follower": {"lead_brake": delay_ms}},
        ),
        at_ms=300,
    )


def crossing_scenario() -> Scenario:
    return Scenario(
        name="crossing-branch",
        version="2.0.0",
        seed=7,
        vehicles=(Vehicle("car1", ((0.0, 0.0), (20.0, 0.0)), 10.0, radius=2.3),),
        pedestrians=(
            Pedestrian("ped1", ((5.0, -4.0), (5.0, 4.0)), 10.0, radius=0.5),
        ),
        signals=(Signal("j1", ((0, "green"), (800, "red"))),),
        events=(
            ScriptedEvent(
                "light-scheduled",
                800,
                {"type": "set_signal", "signal_id": "j1", "phase": "red"},
            ),
        ),
        tick_ms=100,
    )


def make_brake_branch(engine: ReplayEngine) -> ReplayEngine:
    changes = BranchChanges(
        signal_plans={"j1": ((100, "red"), (700, "green"))},
        vehicle_reaction_params={"car1": {"brake_rate": 50.0}},
        extra_events=(
            ScriptedEvent(
                "branch-brake",
                100,
                {
                    "type": "stop_actor",
                    "actor_id": "car1",
                    "reason": "同刻变灯触发刹车",
                },
                seq=1,
            ),
            ScriptedEvent("branch-incident", 300, {"type": "pause"}),
            ScriptedEvent("branch-recover", 400, {"type": "resume"}),
        ),
    )
    return branch_from(engine, "brake-on-red", changes, at_ms=100)


def test_branch_preserves_prefix_and_compares_collision_reason():
    baseline = ReplayEngine(crossing_scenario())
    branched = make_brake_branch(ReplayEngine(crossing_scenario()))

    at_branch = compare_at(baseline, branched, 0)
    assert at_branch.equal
    branched.seek(100)
    assert branched.signal_phase("j1") == "red"

    result = compare_at(baseline, branched, 400)
    assert not result.equal
    assert result.collisions_a.pairs == (("car1", "ped1"),)
    assert result.collisions_b.pairs == ()
    assert "距离小于碰撞半径之和" in result.collisions_a.reasons[("car1", "ped1")]
    assert result.reasons_b["branch-brake"] == "同刻变灯触发刹车"
    assert branched.state.speeds["car1"] == 0.0


def test_branch_stepwise_equals_seek_after_interrupt_and_resume():
    stepped = make_brake_branch(ReplayEngine(crossing_scenario()))
    stepped.run_to(1000)
    assert stepped.now_ms == 300
    assert stepped.state.interrupted is True
    stepped.state.interrupted = False
    stepped.run_to(700)

    jumped = make_brake_branch(ReplayEngine(crossing_scenario()))
    jumped.seek(700)
    assert jumped.now_ms == 300
    jumped.state.interrupted = False
    jumped.seek(700)

    assert stepped.now_ms == jumped.now_ms == 700
    assert stepped.positions() == jumped.positions()
    assert stepped.signal_phases() == jumped.signal_phases()
    assert [(r.tick_ms, r.event_id) for r in stepped.journal] == [
        (r.tick_ms, r.event_id) for r in jumped.journal
    ]
    assert stepped.collisions == jumped.collisions

    stepped.seek(300)
    stepped.run_to(700)
    assert stepped.fired_event_ids().count("branch-brake") == 1
    assert stepped.fired_event_ids().count("branch-incident") == 1
    assert len(stepped.collisions) == len(jumped.collisions)


def test_pedestrian_route_branch_is_self_contained_after_save(tmp_path):
    source = ReplayEngine(crossing_scenario())
    changes = BranchChanges(
        pedestrian_routes={"ped1": ((5.0, -1.0), (-5.0, -1.0), (-5.0, 5.0))},
        pedestrian_speeds={"ped1": 5.0},
    )
    branched = branch_from(source, "ped-divert", changes, at_ms=300)
    assert branched.position_of("ped1") == (5.0, -1.0)
    branched.run_to(700)
    path = save_run(branched, tmp_path / "branch.json")

    reopened = load_run(path)
    assert reopened.scenario.branch_id == "ped-divert"
    assert reopened.scenario.parent_id == "crossing-branch"
    assert reopened.scenario.branch_point_ms == 300
    assert reopened.now_ms == 700

    rebuilt = ReplayEngine(reopened.scenario)
    rebuilt.seek(700)
    assert rebuilt.positions() == reopened.positions()
    assert rebuilt.fired_event_ids() == reopened.fired_event_ids()
    assert rebuilt.position_of("ped1")[0] < 5.0


def test_seek_backward_does_not_duplicate_collisions_or_events():
    engine = make_brake_branch(ReplayEngine(crossing_scenario()))
    engine.run_to(400)
    engine.state.interrupted = False
    engine.run_to(800)
    engine.seek(700)
    event_ids = engine.fired_event_ids()
    collision_count = len(engine.collisions)

    engine.seek(300)
    engine.state.interrupted = False
    engine.run_to(800)
    assert engine.fired_event_ids() == event_ids
    assert len(engine.collisions) == collision_count


def test_branch_drops_parent_future_events_and_never_recounts_prefix():
    # 父场景先跑到 800（light-scheduled 已触发），再在 100 处分支
    source = ReplayEngine(crossing_scenario())
    source.run_to(800)
    branched = branch_from(
        source,
        "no-future-leak",
        BranchChanges(signal_plans={"j1": ((100, "red"), (700, "green"))}),
        at_ms=100,
    )
    # 分支点之后的父场景事件（800ms 的 light-scheduled）不得带入分支
    assert all(record.tick_ms <= 100 for record in branched.journal)

    # 向后跳转再重放，分支点之前的事实不得重复累计
    branched.run_to(500)
    journal_before = list(branched.journal)
    branched.seek(0)
    branched.run_to(500)
    assert branched.fired_event_ids() == tuple(r.event_id for r in journal_before)

    # 分支点恰好在事件时刻：light-scheduled 在 800，分支点也取 800，
    # 该事件已在父场景触发，属于共享前缀，分支不会再次登记
    source2 = ReplayEngine(crossing_scenario())
    source2.run_to(800)
    at_event = branch_from(source2, "at-event", BranchChanges(), at_ms=800)
    assert at_event.fired_event_ids().count("light-scheduled") == 1
    at_event.seek(0)
    at_event.run_to(1000)
    assert at_event.fired_event_ids().count("light-scheduled") == 1


def test_sibling_branches_same_point_and_first_divergence():
    # 同一时刻分两个分支：一个只替换车辆后续路线，一个只改信号灯安排
    route_branch = branch_from(
        ReplayEngine(crossing_scenario()),
        "car-divert",
        BranchChanges(vehicle_routes={"car1": ((5.0, 5.0), (5.0, 10.0))}),
        at_ms=300,
    )
    signal_branch = branch_from(
        ReplayEngine(crossing_scenario()),
        "early-red",
        BranchChanges(signal_plans={"j1": ((500, "red"),)}),
        at_ms=500,
    )
    baseline = ReplayEngine(crossing_scenario())

    # 路线分支在分支点后的首个 tick 即按新路线推进
    assert compare_at(baseline, signal_branch, 500).positions_match

    # 第一处行为差异：车辆分支在 500ms 的运动已沿新路线发生
    divergence = first_divergence(baseline, route_branch, until_ms=1000, start_ms=500)
    assert divergence is not None
    assert divergence.tick_ms == 500
    assert divergence.aspects == ("positions",)
    assert divergence.actors == ("car1",)

    # 信号灯分支的第一处差异是 500ms 的相位
    divergence = first_divergence(baseline, signal_branch, until_ms=1000, start_ms=500)
    assert divergence is not None
    assert divergence.tick_ms == 500
    assert divergence.aspects == ("signals",)
    assert divergence.signals == ("j1",)

    # 两个分支之间也在 500ms 因信号灯安排不同而分叉
    divergence = first_divergence(route_branch, signal_branch, until_ms=1000, start_ms=500)
    assert divergence is not None
    assert divergence.tick_ms == 500
    assert "signals" in divergence.aspects


def test_branch_within_branch_event_ownership(tmp_path):
    # 第一级分支：200ms 起车辆改道，并加入分支事件
    level_one = branch_from(
        ReplayEngine(crossing_scenario()),
        "level-1",
        BranchChanges(
            vehicle_routes={"car1": ((2.0, 0.0), (2.0, 10.0))},
            extra_events=(
                ScriptedEvent(
                    "level-1-honk",
                    400,
                    {"type": "set_speed", "actor_id": "car1", "speed": 4.0},
                ),
            ),
        ),
        at_ms=200,
    )
    level_one.run_to(600)
    assert "level-1-honk" in level_one.fired_event_ids()

    # 分支内再分支：300ms 起只替换信号灯安排
    level_two = branch_from(
        level_one,
        "level-2",
        BranchChanges(
            signal_plans={"j1": ((300, "yellow"),)},
            drop_event_ids=("level-1-honk",),
        ),
        at_ms=300,
    )
    # 二级分支改道：不带入一级分支在 300ms 之后的事件（level-1-honk 在 400ms）
    assert "level-1-honk" not in level_two.fired_event_ids()
    assert level_two.scenario.parent_id == "level-1"
    assert level_two.scenario.branch_point_ms == 300
    level_two.run_to(700)
    assert "level-1-honk" not in level_two.fired_event_ids()

    # 保存重开：事件归属不变，父级后续事件不会复活
    path = save_run(level_two, tmp_path / "level2.json")
    reopened = load_run(path)
    assert "level-1-honk" not in reopened.fired_event_ids()
    reopened.seek(0)
    reopened.run_to(700)
    assert "level-1-honk" not in reopened.fired_event_ids()
    ids = reopened.fired_event_ids()
    assert len(ids) == len(set(ids))


def test_reaction_delay_branches_compare_perception_response_and_gap():
    quick = delay_branch(100, "quick")
    slow = delay_branch(500, "slow")

    quick.seek(400)
    slow.seek(400)
    assert quick.perceptions[0].tick_ms == 400
    assert slow.perceptions[0].tick_ms == 400
    assert quick.pending_response_records()[0].due_tick_ms == 500
    assert slow.pending_response_records()[0].due_tick_ms == 900
    assert quick.position_of("follower") == slow.position_of("follower")

    quick.seek(500)
    slow.seek(500)
    assert quick.state.speeds["follower"] == pytest.approx(8.0)
    assert slow.state.speeds["follower"] == 10.0

    quick_gap = quick.minimum_gap_between(1200, "follower", "leader")
    slow_gap = slow.minimum_gap_between(1200, "follower", "leader")
    assert quick_gap > slow_gap

    summary = first_response_comparison(quick, slow, 1200, "follower", "lead_brake")
    assert summary["a"]["trigger_tick_ms"] == 400
    assert summary["a"]["delay_ms"] == 100
    assert summary["b"]["delay_ms"] == 500
    slow.seek(1000)
    assert slow.collisions
    chain = slow.conflict_event_chain(slow.collisions[0])
    assert any(getattr(record, "stimulus", None) == "lead_brake" for record in chain)
    assert any(getattr(record, "event_id", None) == "leader-brakes" for record in chain)


def test_pending_response_survives_seek_and_save_without_duplication(tmp_path):
    engine = delay_branch(300, "saved-pending")
    engine.run_to(500)
    assert len(engine.pending_response_records()) == 1
    path = save_run(engine, tmp_path / "pending.json")

    reopened = load_run(path)
    assert reopened.pending_response_records()[0].due_tick_ms == 700
    reopened.seek(400)
    assert reopened.pending_response_records()[0].due_tick_ms == 700
    reopened.run_to(700)
    response_ids = [r.response_id for r in reopened.responses if r.status == "executed"]
    assert len(response_ids) == 1
    assert reopened.fired_event_ids().count(response_ids[0]) == 1


def test_same_tick_stimuli_use_signal_order_and_replace_pending_response():
    source = ReplayEngine(delayed_lead_brake_scenario())
    branched = branch_from(
        source,
        "same-tick",
        BranchChanges(
            signal_plans={"j1": ((400, "red"),)},
            vehicle_reaction_params={"follower": {"brake_rate": 20.0}},
            vehicle_reaction_delays={
                "follower": {"signal_red": 500, "lead_brake": 100}
            },
        ),
        at_ms=300,
    )
    branched.run_to(400)
    perceptions = branched.perceptions
    assert [p.stimulus for p in perceptions] == ["signal_red", "lead_brake"]
    statuses = {r.response_id: r.status for r in branched.responses}
    assert any(status == "superseded" for status in statuses.values())
    assert len(branched.pending_response_records()) == 1
    assert branched.pending_response_records()[0].stimulus == "lead_brake"
    assert perceptions[0].supersedes_response_id is None
    assert perceptions[1].supersedes_response_id == perceptions[0].response_id


def test_future_signal_change_does_not_rewrite_seen_fact():
    source = ReplayEngine(delayed_lead_brake_scenario())
    seen_branch = branch_from(
        source,
        "signal-seen",
        BranchChanges(
            signal_plans={"j1": ((400, "red"), (600, "green"))},
            vehicle_reaction_delays={
                "follower": {"signal_red": 100, "signal_green": 100}
            },
        ),
        at_ms=300,
    )
    seen_branch.run_to(500)
    branched = branch_from(
        seen_branch,
        "signal-after-fact",
        BranchChanges(signal_plans={"j1": ((700, "green"),)}),
        at_ms=500,
    )
    branched.run_to(500)
    assert branched.state.observed_signal_phases["j1"] == "red"
    assert any(p.source_id == "j1" and p.observed["phase"] == "red" for p in branched.perceptions)
    branched.run_to(700)
    seen = [p for p in branched.perceptions if p.source_id == "j1"]
    assert [p.observed["phase"] for p in seen] == ["red", "green"]


def test_vehicle_exit_cancels_pending_response_and_siblings_remain_isolated():
    source = ReplayEngine(delayed_lead_brake_scenario())
    exited = branch_from(
        source,
        "exits",
        BranchChanges(
            vehicle_reaction_params={"follower": {"brake_rate": 20.0}},
            vehicle_reaction_delays={"follower": {"lead_brake": 600}},
            extra_events=(
                ScriptedEvent(
                    "follower-exits",
                    500,
                    {"type": "exit_actor", "actor_id": "follower"},
                ),
            ),
        ),
        at_ms=300,
    )
    sibling = delay_branch(600, "stays")
    exited.run_to(1000)
    assert all(r.status != "executed" for r in exited.responses)
    sibling.run_to(400)
    assert sibling.pending_response_records()[0].status == "scheduled"


def test_branch_within_branch_does_not_leak_or_reexecute_response():
    first = delay_branch(500, "delay-level-1")
    first.run_to(400)
    second = branch_from(
        first,
        "delay-level-2",
        BranchChanges(vehicle_reaction_delays={"follower": {"lead_brake": 200}}),
        at_ms=400,
    )
    assert second.pending_response_records()[0].due_tick_ms == 900
    second.run_to(900)
    second.seek(400)
    second.run_to(900)
    executed = [r for r in second.responses if r.status == "executed"]
    assert len(executed) == 1
    assert executed[0].due_tick_ms == 900
