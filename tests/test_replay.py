"""Behavioural tests for the scenario replay engine."""

import pathlib

import pytest

from scenario_replay import (
    Event, ReplayEngine, Scenario, load_scenario, save_run,
    verify_run,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent


def make_intersection():
    return load_scenario(ROOT / "examples" / "intersection.json")


def run_stepwise(scenario, target_time):
    engine = ReplayEngine(scenario)
    while engine.clock.time < target_time and not engine.interrupted:
        engine.step()
    return engine


def run_jump(scenario, target_time):
    engine = ReplayEngine(scenario)
    engine.seek(target_time)
    return engine


def test_single_clock_drives_vehicle_pedestrian_signal():
    engine = run_stepwise(make_intersection(), 1.0)
    assert engine.clock.tick == 2
    ego = engine.participants["ego"]
    walker = engine.participants["walker"]
    light = engine.participants["light"]
    assert ego.x == pytest.approx(-20.0 + 4.0 * 1.0)
    assert walker.x == pytest.approx(-3.5)
    assert walker.y < 2.0
    assert light.phase == "red"


def test_stepwise_and_jump_agree_exactly():
    target = 6.0
    stepped = run_stepwise(make_intersection(), target)
    jumped = run_jump(make_intersection(), target)
    assert stepped.clock.time == jumped.clock.time
    assert ([r.to_dict() for r in stepped.log] ==
            [r.to_dict() for r in jumped.log])
    assert stepped.positions == jumped.positions
    for pid in stepped.participants:
        assert (stepped.participants[pid].state() ==
                jumped.participants[pid].state())


def test_same_instant_event_order_is_fixed():
    order_seen = []

    def make_recorder(name):
        def _effect(engine):
            order_seen.append(name)
        return _effect

    scenario = Scenario(name="ordered", step_seconds=1.0, duration=3.0)
    engine = ReplayEngine(scenario)
    engine.add_event(Event("z_last", 1, "mark", order=20,
                           effect=make_recorder("z")))
    engine.add_event(Event("a_first", 1, "mark", order=1,
                           effect=make_recorder("a")))
    engine.add_event(Event("m_mid", 1, "mark", order=10,
                           effect=make_recorder("m")))
    engine.step()
    assert order_seen == ["a", "m", "z"]
    assert [r.event_id for r in engine.log] == ["a_first", "m_mid",
                                                "z_last"]
    assert [r.seq for r in engine.log] == [1, 2, 3]


def test_rewind_prevents_retrigger_and_can_continue():
    engine = run_stepwise(make_intersection(), 5.0)
    fired_before = [r.to_dict() for r in engine.log]
    assert fired_before
    engine.seek(2.0)
    assert engine.clock.time == 2.0
    ids_at_2 = {r.event_id for r in engine.log}
    assert "turn_red" not in ids_at_2
    engine.run_until(5.0)
    ids_after = [r.event_id for r in engine.log]
    assert ids_after == [r["event_id"] for r in fired_before]


def test_processed_event_never_fires_twice_on_forward_pass():
    engine = ReplayEngine(make_intersection())
    engine.run_until(12.0)
    ids = [r.event_id for r in engine.log]
    for event_id in set(ids):
        assert ids.count(event_id) == 1


def test_pause_blocks_bulk_run_but_not_explicit_step_or_seek():
    engine = ReplayEngine(make_intersection())
    engine.run_until(2.0)
    engine.pause()
    fired = engine.run_until(6.0)
    assert fired == [] and engine.clock.time == 2.0
    engine.step()
    assert engine.clock.time == 2.5
    engine.seek(4.0)
    assert engine.clock.time == 4.0 and engine.clock.paused


def test_interrupt_pauses_scenario_and_resume_continues():
    engine = ReplayEngine(make_intersection())
    hit = engine.run_until(12.0)
    assert engine.interrupted
    assert any(r.etype == "interrupt" for r in hit)
    stopped_at = engine.clock.time
    engine.run_until(12.0)
    assert engine.clock.time == stopped_at
    engine.resume()
    assert not engine.interrupted and not engine.clock.paused
    engine.run_until(stopped_at + 1.0)
    assert engine.clock.time == stopped_at + 1.0


def test_interleaved_vehicle_and_pedestrian_positions_logged():
    engine = run_stepwise(make_intersection(), 3.0)
    ticks = [p["tick"] for p in engine.positions]
    assert ticks == list(range(1, 7))
    row = engine.positions[-1]["participants"]
    assert set(row) == {"ego", "walker", "light", "crossing_light"}


def test_seed_makes_rng_driven_motion_reproducible():
    a = run_stepwise(make_intersection(), 6.0)
    b = run_stepwise(make_intersection(), 6.0)
    assert a.positions == b.positions
    data = make_intersection().to_dict()
    data["seed"] = 999
    c = run_stepwise(Scenario.from_dict(data), 6.0)
    assert c.positions != a.positions


def test_save_and_verify_roundtrip(tmp_path):
    engine = run_stepwise(make_intersection(), 4.0)
    path = save_run(engine, tmp_path / "run.json")
    report = verify_run(path)
    assert report["ok"], report


def test_manual_rewind_then_forward_matches_fresh_run():
    fresh = run_jump(make_intersection(), 8.0)
    engine = ReplayEngine(make_intersection())
    engine.seek(8.0)
    engine.seek(1.5)
    engine.seek(8.0)
    assert engine.positions == fresh.positions
    assert ([r.to_dict() for r in engine.log] ==
            [r.to_dict() for r in fresh.log])


def test_history_lookup_returns_past_positions():
    engine = ReplayEngine(make_intersection())
    engine.run_until(4.0)
    past = engine.state_at(1)["ego"]["x"]
    now = engine.state_at(4)["ego"]["x"]
    assert past < now
