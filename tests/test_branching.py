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
    load_run,
    save_run,
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
