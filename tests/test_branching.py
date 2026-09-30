import pytest

from scenario_replay import (
    BranchChanges,
    Pedestrian,
    ReplayEngine,
    Scenario,
    ScriptedEvent,
    Signal,
    Vehicle,
    compare_at,
    create_branch,
    load_run,
    save_run,
)


def make_crossing_scenario(interrupt: bool = True) -> Scenario:
    events = [
        ScriptedEvent(
            "log-crossing",
            1500,
            {"type": "set_signal", "signal_id": "j1", "phase": "yellow"},
        )
    ]
    if interrupt:
        events.append(ScriptedEvent("scene-incident", 3000, {"type": "pause"}))
    return Scenario(
        name="crossing-branch",
        version="2.0.0",
        seed=7,
        vehicles=(Vehicle("car1", ((0.0, 0.0), (40.0, 0.0)), 8.0),),
        pedestrians=(
            Pedestrian("ped1", ((20.0, -25.0), (20.0, 5.0)), 10.0),
        ),
        signals=(Signal("j1", ((0, "green"), (3000, "red"))),),
        events=tuple(events),
        tick_ms=100,
    )


def make_avoid_branch(parent: Scenario) -> Scenario:
    return create_branch(
        parent,
        branch_tick_ms=500,
        changes=BranchChanges(
            vehicle_speeds={"car1": 0.0},
            pedestrian_paths={"ped1": ((20.0, -20.0), (20.0, -30.0))},
            signal_plans={"j1": ((0, "green"), (500, "red"))},
        ),
        name="avoid",
    )


def test_branch_keeps_prefix_identical_and_applies_same_tick_changes():
    base = make_crossing_scenario()
    branch = make_avoid_branch(base)

    base_engine = ReplayEngine(base)
    stepped_branch = ReplayEngine(branch)
    jumped_branch = ReplayEngine(branch)

    base_engine.run_to(400)
    stepped_branch.advance(4)
    assert stepped_branch.now_ms == 400
    assert stepped_branch.positions() == base_engine.positions()
    assert stepped_branch.signal_phases() == base_engine.signal_phases()

    stepped_branch.advance(1)
    jumped_branch.seek(500)

    assert stepped_branch.now_ms == jumped_branch.now_ms == 500
    assert stepped_branch.positions() == jumped_branch.positions()
    assert stepped_branch.signal_phases() == jumped_branch.signal_phases()
    assert stepped_branch.fired_event_ids() == jumped_branch.fired_event_ids()
    assert stepped_branch.signal_phase("j1") == "red"
    assert stepped_branch.state.speeds["car1"] == 0.0
    assert stepped_branch.fired_event_ids()[-3:] == (
        "branch:avoid:speed:car1",
        "branch:avoid:path:ped1",
        "branch:avoid:signal:j1",
    )


def test_compare_branches_reports_position_collision_and_reasons():
    base = make_crossing_scenario()
    branch = make_avoid_branch(base)

    before = compare_at(ReplayEngine(base), ReplayEngine(branch), 400)
    assert before.positions_equal is True
    assert before.collisions_equal is True

    comparison = compare_at(ReplayEngine(base), ReplayEngine(branch), 2400)
    assert comparison.positions_equal is False
    assert len(comparison.left_collisions) == 1
    assert comparison.right_collisions == ()
    assert comparison.collisions_equal is False
    assert comparison.left_collisions[0].vehicle_id == "car1"
    assert comparison.left_collisions[0].pedestrian_id == "ped1"
    assert "collided" in comparison.left_reasons[0]
    assert comparison.reasons_equal is False


def test_stepwise_and_jump_agree_after_interrupt_resume():
    branch = make_avoid_branch(make_crossing_scenario())

    stepped = ReplayEngine(branch)
    stepped.run_to(3000)
    assert stepped.now_ms == 3000
    assert stepped.state.interrupted is True
    stepped.advance(5)
    assert stepped.now_ms == 3000

    stepped.state.interrupted = False
    stepped.run_to(3500)

    jumped = ReplayEngine(branch)
    jumped.seek(3500)
    jumped.state.interrupted = False

    assert stepped.positions() == jumped.positions()
    assert stepped.signal_phases() == jumped.signal_phases()
    assert stepped.fired_event_ids() == jumped.fired_event_ids()


def test_pause_seek_backward_and_forward_does_not_duplicate_events():
    branch = make_avoid_branch(make_crossing_scenario())
    engine = ReplayEngine(branch)
    engine.run_to(1200)
    engine.pause()
    engine.advance(4)
    assert engine.now_ms == 1200

    engine.resume()
    engine.seek(500)
    engine.run_to(3000)

    ids = engine.fired_event_ids()
    for event_id in (
        "branch:avoid:speed:car1",
        "branch:avoid:path:ped1",
        "branch:avoid:signal:j1",
        "log-crossing",
        "scene-incident",
    ):
        assert ids.count(event_id) == 1


def test_saved_branch_reopens_without_parent_scenario(tmp_path):
    parent = make_crossing_scenario()
    branch = make_avoid_branch(parent)
    engine = ReplayEngine(branch)
    engine.run_to(1000)
    path = save_run(engine, tmp_path / "branch.json")

    restored = load_run(path)
    assert restored.scenario.branch == {
        "name": "avoid",
        "parent": "root",
        "branch_tick_ms": 500,
    }
    assert restored.now_ms == 1000
    assert restored.signal_phase("j1") == "red"
    assert restored.position_of("car1")[0] == pytest.approx(4.0)

    restored.run_to(2800)
    assert restored.collisions == []
    assert restored.fired_event_ids().count("branch:avoid:path:ped1") == 1
