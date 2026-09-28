import math

import pytest

from scenario_replay import (
    Pedestrian,
    ReplayEngine,
    Scenario,
    ScriptedEvent,
    Signal,
    Vehicle,
    load_run,
    save_run,
)


def make_scenario(interrupt: bool = False) -> Scenario:
    events = [
        # 同一时刻两个事件，用 seq 固定顺序
        ScriptedEvent("car-brake", 300, {"type": "set_speed", "actor_id": "car1", "speed": 2.0}),
        ScriptedEvent("car-slow", 300, {"type": "stop_actor", "actor_id": "car1"}, seq=1),
        ScriptedEvent("red", 500, {"type": "set_signal", "signal_id": "j1", "phase": "red"}),
        ScriptedEvent("ped-run", 700, {"type": "set_speed", "actor_id": "ped1", "speed": 3.0}),
    ]
    if interrupt:
        events.append(ScriptedEvent("incident", 600, {"type": "pause"}))
    return Scenario(
        name="crossroads",
        version="1.2.0",
        seed=42,
        vehicles=(Vehicle("car1", ((0.0, 0.0), (100.0, 0.0)), 10.0),),
        pedestrians=(Pedestrian("ped1", ((50.0, -10.0), (50.0, 10.0)), 1.0),),
        signals=(Signal("j1", ((0, "green"), (400, "yellow"), (800, "red"))),),
        events=tuple(events),
        tick_ms=100,
    )


def test_same_tick_event_order_is_fixed():
    engine = ReplayEngine(make_scenario())
    engine.run_to(300)
    fired = engine.fired_event_ids()
    # 300ms 处先刹车后停车，顺序固定；车最终速度为 0
    assert fired[-2:] == ("car-brake", "car-slow")
    assert engine.state.speeds["car1"] == 0.0


def test_stepwise_matches_jump_to_target():
    stepped = ReplayEngine(make_scenario())
    stepped.advance(10)

    jumped = ReplayEngine(make_scenario())
    jumped.seek(1000)

    assert stepped.now_ms == jumped.now_ms == 1000
    assert stepped.positions() == jumped.positions()
    assert stepped.signal_phases() == jumped.signal_phases()
    assert stepped.fired_event_ids() == jumped.fired_event_ids()
    # 车辆 0-300ms 以 10m/s 行驶 3m，事件后停止
    assert stepped.position_of("car1")[0] == pytest.approx(3.0)
    # 行人 0-700ms 以 1m/s，之后 3m/s：共行驶 1.6m，y 从 -10 到 -8.4
    assert stepped.position_of("ped1")[1] == pytest.approx(-8.4)


def test_pause_blocks_advance_and_run_to():
    engine = ReplayEngine(make_scenario())
    engine.run_to(200)
    engine.pause()
    engine.advance(5)
    assert engine.now_ms == 200
    engine.resume()
    engine.advance(1)
    assert engine.now_ms == 300


def test_interrupt_event_stops_scenario():
    engine = ReplayEngine(make_scenario(interrupt=True))
    engine.run_to(5000)
    assert engine.now_ms == 600
    assert engine.state.interrupted is True
    engine.advance(3)
    assert engine.now_ms == 600
    # 恢复事件后可继续
    engine.state.interrupted = False
    engine.run_to(700)
    assert engine.now_ms == 700
    assert "ped-run" in engine.fired_event_ids()


def test_seek_to_past_does_not_refire_events():
    engine = ReplayEngine(make_scenario())
    engine.run_to(1000)
    journal_before = list(engine.journal)

    engine.seek(300)  # 跳回过去
    assert engine.now_ms == 300
    assert engine.position_of("car1")[0] == pytest.approx(3.0)

    engine.run_to(1000)  # 继续推进
    ids = engine.fired_event_ids()
    # 300ms 的事件在首次运行已处理，不能重复出现
    assert ids.count("car-brake") == 1
    assert ids.count("car-slow") == 1
    # 日志保持按时间有序且不重复
    assert [r.event_id for r in engine.journal] == [
        r.event_id for r in journal_before
    ]


def test_seek_past_then_new_path_is_consistent_with_fresh_run_to_past():
    engine = ReplayEngine(make_scenario())
    engine.run_to(900)
    engine.seek(400)

    fresh = ReplayEngine(make_scenario())
    fresh.seek(400)
    assert engine.positions() == fresh.positions()
    assert engine.signal_phases() == fresh.signal_phases()


def test_save_and_load_roundtrip(tmp_path):
    engine = ReplayEngine(make_scenario(interrupt=True))
    engine.run_to(600)
    path = save_run(engine, tmp_path / "run.json")

    restored = load_run(path)
    assert restored.scenario.version == "1.2.0"
    assert restored.scenario.seed == 42
    assert restored.now_ms == 600
    assert restored.state.interrupted is True
    assert restored.positions() == engine.positions()
    assert restored.fired_event_ids() == engine.fired_event_ids()
    # 恢复后继续运行，已处理事件不重复触发
    restored.state.interrupted = False
    restored.run_to(1000)
    assert restored.fired_event_ids().count("red") == 1


def test_seed_and_version_replay_determinism():
    positions = []
    for _ in range(2):
        engine = ReplayEngine(make_scenario())
        engine.run_to(800)
        positions.append((engine.positions(), engine.signal_phases()))
    assert positions[0] == positions[1]
    assert not math.isnan(engine.position_of("car1")[0])
