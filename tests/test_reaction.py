"""反应延迟实验：感知与响应分离、延迟响应、方案对比与分支隔离。"""
from scenario_replay import (
    BranchChanges,
    ReactionScheme,
    ReplayEngine,
    Scenario,
    ScriptedEvent,
    Signal,
    Vehicle,
    branch_from,
    event_chain_before,
    first_response_difference,
    load_run,
    minimum_gap,
    save_run,
)


def signal_scenario(delay_ms: int = 200, extra_events=()) -> Scenario:
    return Scenario(
        name="reaction-signal",
        version="3.0.0",
        seed=11,
        vehicles=(Vehicle("car1", ((0.0, 0.0), (100.0, 0.0)), 10.0),),
        signals=(Signal("j1", ((0, "green"), (300, "red"))),),
        events=tuple(extra_events),
        reaction_schemes=(
            ReactionScheme(
                vehicle_id="car1",
                delay_ms=delay_ms,
                action={"type": "stop_actor"},
                signal_id="j1",
            ),
        ),
        tick_ms=100,
    )


def following_scenario(delay_ms: int, extra_events=()) -> Scenario:
    return Scenario(
        name="reaction-follow",
        version="3.0.0",
        seed=13,
        vehicles=(
            Vehicle("lead", ((0.0, 0.0), (100.0, 0.0)), 10.0),
            Vehicle("follower", ((-20.0, 0.0), (80.0, 0.0)), 10.0),
        ),
        events=(
            ScriptedEvent("lead-brake", 500, {"type": "stop_actor", "actor_id": "lead"}),
        )
        + tuple(extra_events),
        reaction_schemes=(
            ReactionScheme(
                vehicle_id="follower",
                delay_ms=delay_ms,
                action={"type": "stop_actor"},
                lead_id="lead",
            ),
        ),
        tick_ms=100,
    )


def react_events(engine):
    return [r for r in engine.journal if r.event_id.startswith("react:")]


def test_perception_and_action_recorded_separately_and_vehicle_not_delayed():
    engine = ReplayEngine(signal_scenario(delay_ms=200))
    engine.run_to(1000)

    # 感知事实：300ms 看到变红；实际动作：500ms 才刹车，分离记录
    assert [(p.tick_ms, p.vehicle_id, p.stimulus) for p in engine.perceptions] == [
        (300, "car1", "signal_phase")
    ]
    assert "green" in engine.perceptions[0].detail
    reacts = react_events(engine)
    assert len(reacts) == 1
    assert reacts[0].tick_ms == 500
    assert "延迟 200ms" in reacts[0].reason

    # 延迟不是把车辆晚生成：300-500ms 之间车辆照常推进
    engine.seek(300)
    at_perceive = engine.position_of("car1")
    engine.seek(400)
    assert engine.position_of("car1")[0] > at_perceive[0]
    engine.seek(600)
    assert engine.position_of("car1") == (5.0, 0.0)  # 响应后停住


def test_pending_response_survives_save_and_interrupt_and_fires_on_schedule(tmp_path):
    scenario = signal_scenario(
        delay_ms=400,  # 感知 300ms，原定响应 700ms
        extra_events=(ScriptedEvent("incident", 500, {"type": "pause"}),),
    )
    engine = ReplayEngine(scenario)
    engine.run_to(1000)
    assert engine.now_ms == 500  # 中断在感知之后、响应之前
    assert len(engine.pending_responses) == 1
    assert react_events(engine) == []

    path = save_run(engine, tmp_path / "run.json")
    reopened = load_run(path)
    assert len(reopened.pending_responses) == 1
    assert reopened.pending_responses[0].respond_tick_ms == 700

    # 恢复后仍在原定时刻响应，不随中断平移
    reopened.state.interrupted = False
    reopened.run_to(1000)
    reacts = react_events(reopened)
    assert [r.tick_ms for r in reacts] == [700]


def test_branch_signal_plan_does_not_rewrite_seen_facts():
    parent = ReplayEngine(signal_scenario(delay_ms=200))
    parent.run_to(400)  # 300ms 的变红已被看到

    # 分支把信号安排改为 500ms 变绿：只影响尚未发生的感知
    branch = branch_from(
        parent,
        "early-green",
        BranchChanges(signal_plans={"j1": ((500, "green"),)}),
        at_ms=400,
    )
    branch.run_to(1000)
    seen = [(p.tick_ms, p.detail) for p in branch.perceptions]
    assert seen[0][0] == 300  # 已看到的变红事实保留，不被回写
    assert "red" in seen[0][1]
    assert any(tick == 500 and "green" in detail for tick, detail in seen[1:])
    # 变红引发的响应仍在原定 500ms 执行；变绿引发的新响应在 700ms
    assert [r.tick_ms for r in react_events(branch)] == [500, 700]

    # 兄弟分支不改信号安排：只看到变红，互不影响
    sibling = branch_from(
        ReplayEngine(signal_scenario(delay_ms=200)),
        "keep-red",
        BranchChanges(),
        at_ms=400,
    )
    sibling.run_to(1000)
    assert [p.tick_ms for p in sibling.perceptions] == [300]
    assert [r.tick_ms for r in react_events(sibling)] == [500]


def test_same_tick_stimuli_order_and_superseded_reaction():
    scenario = Scenario(
        name="same-tick",
        version="3.0.0",
        seed=17,
        vehicles=(
            Vehicle("lead", ((0.0, 0.0), (100.0, 0.0)), 10.0),
            Vehicle("follower", ((-20.0, 0.0), (80.0, 0.0)), 10.0),
        ),
        signals=(Signal("j1", ((0, "green"),)),),
        events=(
            ScriptedEvent("lead-brake", 400, {"type": "stop_actor", "actor_id": "lead"}),
            ScriptedEvent(
                "turn-red", 400, {"type": "set_signal", "signal_id": "j1", "phase": "red"}
            ),
        ),
        reaction_schemes=(
            ReactionScheme(
                vehicle_id="follower",
                delay_ms=100,
                action={"type": "stop_actor"},
                signal_id="j1",
                lead_id="lead",
            ),
        ),
        tick_ms=100,
    )
    engine = ReplayEngine(scenario)
    engine.run_to(1000)

    # 同刻两个刺激：固定顺序先信号后前车，都记录为感知事实
    assert [p.stimulus for p in engine.perceptions] == ["signal_phase", "lead_brake"]
    # 先安排的反应被后续决定替代：保留记录但不执行
    assert len(engine.superseded_responses) == 1
    reacts = react_events(engine)
    assert len(reacts) == 1
    assert reacts[0].tick_ms == 500


def test_seek_replay_and_rebranch_do_not_duplicate_or_leak_responses():
    engine = ReplayEngine(signal_scenario(delay_ms=200))
    engine.run_to(1000)
    assert len(react_events(engine)) == 1

    # 回跳重放：响应不重复执行、不重复登记
    engine.seek(200)
    engine.run_to(1000)
    assert len(react_events(engine)) == 1
    assert len(engine.perceptions) == 1

    # 再次分支：分支点前的感知带入，分支点后的父场景响应不泄漏
    branch = branch_from(engine, "re-branch", BranchChanges(), at_ms=400)
    assert [p.tick_ms for p in branch.perceptions] == [300]
    branch.run_to(1000)
    assert len(react_events(branch)) == 1
    # 父场景继续推进也不受分支影响
    assert len(react_events(engine)) == 1


def test_exited_vehicle_pending_response_cancelled():
    scenario = signal_scenario(
        delay_ms=400,  # 原定 700ms 响应
        extra_events=(ScriptedEvent("car-exit", 500, {"type": "exit_actor", "actor_id": "car1"}),),
    )
    engine = ReplayEngine(scenario)
    engine.run_to(1000)
    assert react_events(engine) == []  # 退出后待响应动作不执行
    assert engine.pending_responses == ()
    engine.seek(300)
    engine.run_to(1000)
    assert react_events(engine) == []  # 回跳重放也不复活

    # 分支中让车辆退出：待响应动作取消且不泄漏到该分支
    parent = ReplayEngine(signal_scenario(delay_ms=400))
    parent.run_to(400)
    branch = branch_from(parent, "exit", BranchChanges(exit_vehicles=("car1",)), at_ms=400)
    branch.run_to(1000)
    assert react_events(branch) == []
    # 父方案不受影响，仍在原定时刻响应
    parent.run_to(1000)
    assert [r.tick_ms for r in react_events(parent)] == [700]


def test_compare_reaction_schemes_first_response_gap_and_chain():
    # 共同历史：跑到 400ms（lead 尚未制动），派生两个延迟不同的方案
    base = ReplayEngine(following_scenario(delay_ms=0))
    base.run_to(400)
    quick = branch_from(
        base,
        "quick",
        BranchChanges(
            reaction_schemes={
                "follower": ReactionScheme(
                    vehicle_id="follower",
                    delay_ms=0,
                    action={"type": "stop_actor"},
                    lead_id="lead",
                )
            }
        ),
        at_ms=400,
    )
    slow = branch_from(
        base,
        "slow",
        BranchChanges(
            reaction_schemes={
                "follower": ReactionScheme(
                    vehicle_id="follower",
                    delay_ms=2000,
                    action={"type": "stop_actor"},
                    lead_id="lead",
                )
            }
        ),
        at_ms=400,
    )
    quick.run_to(3000)
    slow.run_to(3000)

    # 首次响应差异：quick 在 500ms 响应，slow 在 2500ms
    difference = first_response_difference(quick, slow, until_ms=3000, start_ms=400)
    assert difference is not None
    assert difference.tick_ms == 500
    assert difference.vehicles == ("follower",)

    # 最小间距：反应越慢间距越小
    gap_quick = minimum_gap(quick, "follower", "lead", until_ms=3000)
    gap_slow = minimum_gap(slow, "follower", "lead", until_ms=3000)
    assert gap_slow.distance < gap_quick.distance

    # 冲突发生前的事件链：感知与响应分别成条目
    assert slow.collisions, "slow 方案应发生碰撞"
    first_conflict = slow.collisions[0].tick_ms
    chain = event_chain_before(slow, first_conflict)
    kinds = [entry.kind for entry in chain]
    assert "perception" in kinds and "event" in kinds
    perception_entry = next(e for e in chain if e.kind == "perception")
    assert perception_entry.tick_ms == 500
    assert "制动" in perception_entry.detail
