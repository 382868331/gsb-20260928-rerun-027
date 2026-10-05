"""同一交通方案多份未确认感知修订候选的并存、比较、选择与撤销。"""
import json
import pytest

from scenario_replay import (
    PerceptionWindow,
    ReplayEngine,
    WindowConstraint,
    adopted_perception_facts,
    candidate_comparisons_to_dict,
    compare_revision_candidates,
    derive_revision_candidate,
    discard_revision_candidate,
    export_revision_comparison,
    list_revision_candidates,
    load_revision_candidate,
    load_run,
    save_run,
    stage_revision_candidate,
    stage_joint_revision_candidate,
    undo_revision_selection,
    withdraw_joint_candidate_source,
    JointRevisionConflict,
)
from tests.test_uncertainty import (
    brake_base_changes,
    brake_window,
    uncertain_lead_brake_scenario,
)


def _engine():
    return ReplayEngine(uncertain_lead_brake_scenario())


def _two_windows_order(kind: str):
    first = PerceptionWindow(
        "w-signal", "follower", "signal_red", "j1", 400, 500, 300
    )
    second = PerceptionWindow(
        "w-brake", "follower", "lead_brake", "leader", 400, 500, 300
    )
    return [first, second], [WindowConstraint(kind, "w-signal", "w-brake")]


def test_multiple_candidates_coexist_without_overwriting():
    engine = _engine()
    stage_revision_candidate(
        engine, "late", [brake_window(600, 800, 500)],
        base_changes=brake_base_changes(), horizon_ms=2500,
        note="晚感知",
    )
    # 再编辑另一份候选：不得覆盖第一份
    stage_revision_candidate(
        engine, "early", [brake_window(400, 500, 300)],
        base_changes=brake_base_changes(), horizon_ms=2500,
        note="早感知快反应",
    )
    candidates = {c.candidate_id: c for c in list_revision_candidates(engine)}
    assert set(candidates) == {"late", "early"}
    assert candidates["late"].windows[0].earliest_ms == 600
    assert candidates["late"].windows[0].delay_ms == 500
    assert candidates["early"].windows[0].earliest_ms == 400
    assert candidates["early"].windows[0].delay_ms == 300

    # 同名候选重复暂存是幂等更新，不产生重复候选
    stage_revision_candidate(
        engine, "late", [brake_window(700, 800, 500)],
        base_changes=brake_base_changes(), horizon_ms=2500,
        note="晚感知-修订",
    )
    assert [c.candidate_id for c in list_revision_candidates(engine)] == [
        "late",
        "early",
    ]
    assert load_revision_candidate(engine, "late").windows[0].earliest_ms == 700

    # 派生其中一份：另一份候选与原方案事实保留
    late_result = derive_revision_candidate(engine, "late", evaluate_until_ms=2500)
    assert engine.selected_revision_id == "late"
    assert {c.candidate_id for c in list_revision_candidates(engine)} == {"early"}
    original_facts = adopted_perception_facts(engine)
    assert all(
        "late" not in fact.window_id for fact in original_facts
    )
    # 原方案场景事件不被候选派生改写
    assert not any(
        event.event_id.startswith("late:") for event in engine.scenario.events
    )
    assert [s.assignment["w-brake"] for s in late_result.schemes] == [700, 800]


def test_undo_selection_and_switch_to_another_candidate():
    engine = _engine()
    stage_revision_candidate(
        engine, "late", [brake_window(600, 800, 500)],
        base_changes=brake_base_changes(), horizon_ms=2500,
    )
    stage_revision_candidate(
        engine, "early", [brake_window(400, 500, 300)],
        base_changes=brake_base_changes(), horizon_ms=2500,
    )
    derive_revision_candidate(engine, "late", evaluate_until_ms=2500)
    assert engine.selected_revision_id == "late"

    # 撤销一次选择：late 退回未决集合，选择回到原方案
    assert undo_revision_selection(engine) is None
    assert engine.selected_revision_id is None
    assert {c.candidate_id for c in list_revision_candidates(engine)} == {
        "late",
        "early",
    }
    # 撤销后可再次派生同一份（事件身份稳定）
    relate = derive_revision_candidate(engine, "late", evaluate_until_ms=2500)
    assert relate.schemes[0].scheme_id == "late-0"

    # 改用另一份候选
    undo_revision_selection(engine)
    early_result = derive_revision_candidate(
        engine, "early", evaluate_until_ms=2500
    )
    assert engine.selected_revision_id == "early"
    assert {c.candidate_id for c in list_revision_candidates(engine)} == {"late"}
    assert [s.assignment["w-brake"] for s in early_result.schemes] == [400, 500]

    # 重复确认同一候选幂等，不重复压栈
    derive_revision_candidate(engine, "early", evaluate_until_ms=2500)
    assert engine.selected_revision_id == "early"
    assert undo_revision_selection(engine) is None


def test_shared_windows_different_order_constraints_compare_and_exclude(tmp_path):
    engine = _engine()
    after_windows, after_constraints = _two_windows_order("after")
    same_windows, same_constraints = _two_windows_order("same")
    stage_revision_candidate(
        engine, "after", after_windows, after_constraints,
        base_changes=brake_base_changes(), horizon_ms=2500,
    )
    stage_revision_candidate(
        engine, "same", same_windows, same_constraints,
        base_changes=brake_base_changes(), horizon_ms=2500,
    )
    after_result = derive_revision_candidate(
        engine, "after", adopt=False, evaluate_until_ms=2500
    )
    same_result = derive_revision_candidate(
        engine, "same", adopt=False, evaluate_until_ms=2500
    )
    # adopt=False：候选仍在未决集合中
    assert {c.candidate_id for c in list_revision_candidates(engine)} == {
        "after",
        "same",
    }

    comparisons = compare_revision_candidates(
        [after_result, same_result], until_ms=2500, base_engine=engine
    )
    by_id = {c.candidate_id: c for c in comparisons}
    # after：w2 >= w1 共 3 组合（1 组合被约束排除）；
    # same：w2 == w1 共 2 组合（2 组合被约束排除）
    assert by_id["after"].explored_count == 3
    assert by_id["after"].constraint_excluded_count == 1
    assert by_id["after"].total_assignments == 4
    assert by_id["after"].unexplored_count == 0
    assert by_id["same"].explored_count == 2
    assert by_id["same"].constraint_excluded_count == 2
    assert by_id["same"].total_assignments == 4
    # 约束排除不进入实际碰撞统计：碰撞条目只含实际重放出碰撞的方案
    actual = [
        (scheme_id, collisions)
        for scheme_id, collisions in by_id["same"].collisions_by_scheme
        if collisions
    ]
    for scheme_id, collisions in actual:
        assert scheme_id.startswith("same-")
        assert collisions[0].actor_a
    # 两份候选共享同样的观测窗口，但刺激先后约束不同
    assert [w.window_id for w in by_id["after"].windows] == [
        "w-signal",
        "w-brake",
    ]
    assert by_id["after"].stimulus_order[0].kind == "after"
    assert by_id["same"].stimulus_order[0].kind == "same"
    # 首处分歧可追溯（注入事实在 400ms 起）
    assert by_id["after"].fact_divergence_from_base is not None
    assert by_id["after"].fact_divergence_from_base.tick_ms == 400
    assert by_id["after"].first_divergence_scheme_id == "after-0"

    # 导出：约束排除、实际碰撞、已探索组合、未验证范围分别成段
    path = tmp_path / "compare.json"
    payload = export_revision_comparison(
        [after_result, same_result],
        str(path),
        until_ms=2500,
        base_engine=engine,
    )
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    ids = {candidate["candidate_id"] for candidate in payload["candidates"]}
    assert ids == {"after", "same"}
    same_payload = next(
        candidate
        for candidate in payload["candidates"]
        if candidate["candidate_id"] == "same"
    )
    assert same_payload["constraint_excluded_count"] == 2
    assert same_payload["explored_count"] == 2
    assert same_payload["unverified_count"] == 0
    assert "actual_collisions" in same_payload
    assert same_payload["first_fact_divergence_from_base"]["tick_ms"] == 400
    # JSON 可序列化且观测范围、反应延迟分别列出
    ranges = same_payload["observation_ranges"]
    assert [r["window_id"] for r in ranges] == ["w-signal", "w-brake"]
    assert {r["reaction_delay_ms"] for r in ranges} == {300}


def test_pending_candidates_persist_across_save_reload(tmp_path):
    engine = _engine()
    stage_revision_candidate(
        engine, "late", [brake_window(600, 800, 500)],
        base_changes=brake_base_changes(), horizon_ms=2500,
        note="保存后仍未决",
    )
    stage_revision_candidate(
        engine, "early", [brake_window(400, 500, 300)],
        base_changes=brake_base_changes(), horizon_ms=2500,
        based_on="late",
    )
    result_late = derive_revision_candidate(
        engine, "late", evaluate_until_ms=2500
    )
    event_ids_before = {
        scheme.scheme_id: [
            event.event_id for event in scheme.engine.journal
        ]
        for scheme in result_late.schemes
    }
    path = save_run(engine, tmp_path / "candidates.json")

    reopened = load_run(path)
    # 未决候选重开后仍在；已选择候选与历史链保留
    pending = {c.candidate_id: c for c in list_revision_candidates(reopened)}
    assert set(pending) == {"early"}
    assert pending["early"].based_on == "late"
    assert reopened.selected_revision_id == "late"

    # 撤销重放：late 回到未决集合；再派生时事件身份稳定
    undo_revision_selection(reopened)
    assert set(pending) | {"late"} == {
        c.candidate_id for c in list_revision_candidates(reopened)
    }
    result_again = derive_revision_candidate(
        reopened, "late", evaluate_until_ms=2500
    )
    for scheme in result_again.schemes:
        assert [event.event_id for event in scheme.engine.journal] == (
            event_ids_before[scheme.scheme_id]
        )

    # 回跳到分歧前再前进，不重复确认、不重复事件
    reopened.seek(0)
    reopened.run_to(2500)
    derive_revision_candidate(reopened, "late", evaluate_until_ms=2500)
    assert reopened.selected_revision_id == "late"
    first_scheme = result_again.schemes[0].engine
    first_scheme.seek(0)
    first_scheme.run_to(2500)
    injected = [
        p for p in first_scheme.perceptions if p.observed.get("injected")
    ]
    assert len(injected) == 1
    assert injected[0].tick_ms == 600


def test_joint_candidate_merges_shared_window_on_one_timeline_and_collides(tmp_path):
    engine = _engine()
    signal = PerceptionWindow(
        "w-signal", "follower", "signal_red", "j1", 400, 600, 300
    )
    brake_a = PerceptionWindow(
        "w-brake", "follower", "lead_brake", "leader", 400, 700, 500,
        observed={"observer": "a"},
    )
    brake_b = PerceptionWindow(
        "w-brake", "follower", "lead_brake", "leader", 600, 800, 500,
        observed={"observer": "b"},
    )
    stage_revision_candidate(
        engine, "observer-a", [signal, brake_a],
        [WindowConstraint("after", "w-signal", "w-brake")],
        base_changes=brake_base_changes(), horizon_ms=2500,
    )
    stage_revision_candidate(
        engine, "observer-b", [brake_b],
        base_changes=brake_base_changes(), horizon_ms=2500,
    )
    joint = stage_joint_revision_candidate(
        engine, "joint-ab", ["observer-a", "observer-b"],
        horizon_ms=2500,
    )
    by_id = {window.window_id: window for window in joint.windows}
    assert set(by_id) == {"w-signal", "w-brake"}
    assert (by_id["w-brake"].earliest_ms, by_id["w-brake"].latest_ms) == (600, 700)
    assert joint.joint_window_sources["w-brake"] == ("observer-a", "observer-b")
    # 仍未确认：不能派生正式方案，来源候选也仍在待审集合
    assert "joint-ab" in {c.candidate_id for c in list_revision_candidates(engine)}
    assert "observer-a" in {c.candidate_id for c in list_revision_candidates(engine)}

    result = derive_revision_candidate(engine, "joint-ab", evaluate_until_ms=2500)
    assert engine.selected_revision_id == "joint-ab"
    assert {c.candidate_id for c in list_revision_candidates(engine)} == set()
    # 每个方案是包含两个窗口的一条联合时间线，而不是拼接单候选运行结果
    assert sorted(result.schemes[0].assignment) == ["w-brake", "w-signal"]
    assert result.report is not None
    assert {evaluation.scheme_id for evaluation in result.report.conflicts}
    comparisons = compare_revision_candidates([result], until_ms=2500, base_engine=engine)
    assert comparisons[0].joint_source_ids == ("observer-a", "observer-b")
    assert comparisons[0].collisions_by_scheme

    path = save_run(engine, tmp_path / "confirmed-joint.json")
    reopened = load_run(path)
    assert reopened.selected_revision_id == "joint-ab"
    repeated = derive_revision_candidate(
        reopened, "joint-ab", evaluate_until_ms=2500
    )
    assert repeated.schemes[0].scheme_id == result.schemes[0].scheme_id
    assert repeated.revision.joint_candidate_ids == ("observer-a", "observer-b")
    assert repeated.revision.joint_window_sources["w-brake"] == (
        "observer-a", "observer-b"
    )


def test_joint_candidate_conflicts_keep_witness_without_dropping_constraints():
    engine = _engine()
    windows_a, constraints_a = _two_windows_order("after")
    windows_b, constraints_b = _two_windows_order("after")
    stage_revision_candidate(
        engine, "a", windows_a, constraints_a,
        base_changes=brake_base_changes(), horizon_ms=2500,
    )
    stage_revision_candidate(
        engine, "b", windows_b,
        [WindowConstraint("after", "w-brake", "w-signal")],
        base_changes=brake_base_changes(), horizon_ms=2500,
    )
    with pytest.raises(JointRevisionConflict) as exc:
        stage_joint_revision_candidate(engine, "bad", ["a", "b"])
    assert exc.value.candidate_ids == ("a", "b")
    assert set(exc.value.window_ids) == {"w-signal", "w-brake"}
    assert len(exc.value.constraints) == 2
    assert "bad" not in {c.candidate_id for c in list_revision_candidates(engine)}

    narrow = PerceptionWindow(
        "w-signal", "follower", "signal_red", "j1", 600, 700, 300
    )
    other = PerceptionWindow(
        "w-brake", "follower", "lead_brake", "leader", 400, 500, 300
    )
    stage_revision_candidate(
        engine, "no-overlap", [narrow, other],
        base_changes=brake_base_changes(), horizon_ms=2500,
    )
    with pytest.raises(JointRevisionConflict) as exc:
        stage_joint_revision_candidate(engine, "bad-range", ["a", "no-overlap"])
    assert exc.value.candidate_ids == ("a", "no-overlap")
    assert exc.value.window_ids == ("w-signal",)


def test_partial_joint_withdrawal_restores_only_unsupported_windows(tmp_path):
    engine = _engine()
    signal = PerceptionWindow(
        "w-signal", "follower", "signal_red", "j1", 400, 600, 300
    )
    brake = brake_window(600, 800, 500)
    pedestrian = PerceptionWindow(
        "w-pedestrian", "follower", "pedestrian_crossing", "p1", 500, 700, 200
    )
    stage_revision_candidate(
        engine, "main", [signal, brake],
        base_changes=brake_base_changes(), horizon_ms=2500,
    )
    stage_revision_candidate(
        engine, "extra", [pedestrian],
        base_changes=brake_base_changes(), horizon_ms=2500,
    )
    stage_joint_revision_candidate(
        engine, "joint", ["main", "extra"], horizon_ms=2500
    )
    derive_revision_candidate(engine, "joint", evaluate_until_ms=2500)
    original_events = [event.event_id for event in engine.scenario.events]

    updated = withdraw_joint_candidate_source(engine, "joint", "extra")
    assert updated is not None
    assert updated.review_window_ids == ("w-pedestrian",)
    assert "extra" in {c.candidate_id for c in list_revision_candidates(engine)}
    assert "main" not in {c.candidate_id for c in list_revision_candidates(engine)}
    assert load_revision_candidate(engine, "extra") is not None
    assert [window.window_id for window in updated.windows] == ["w-signal", "w-brake"]
    rederived = derive_revision_candidate(
        engine, "joint", adopt=False, evaluate_until_ms=2500
    )
    assert sorted(rederived.schemes[0].assignment) == ["w-brake", "w-signal"]
    again = derive_revision_candidate(
        engine, "joint", adopt=False, evaluate_until_ms=2500
    )
    assert again.schemes[0].scheme_id == rederived.schemes[0].scheme_id

    path = save_run(engine, tmp_path / "joint-withdraw.json")
    reopened = load_run(path)
    reopened_joint = load_revision_candidate(reopened, "joint")
    assert reopened_joint is not None
    assert reopened_joint.joint_candidate_ids == ("main",)
    assert reopened_joint.withdrawn_candidate_ids == ("extra",)
    assert reopened_joint.review_window_ids == ("w-pedestrian",)
    assert [event.event_id for event in reopened.scenario.events] == original_events


def test_discard_candidate_keeps_others_and_facts():
    engine = _engine()
    stage_revision_candidate(
        engine, "late", [brake_window(600, 800, 500)],
        base_changes=brake_base_changes(),
    )
    stage_revision_candidate(
        engine, "early", [brake_window(400, 500, 300)],
        base_changes=brake_base_changes(),
    )
    discard_revision_candidate(engine, "late")
    assert [c.candidate_id for c in list_revision_candidates(engine)] == [
        "early"
    ]
    assert not any(
        event.event_id.startswith("late:") for event in engine.scenario.events
    )
    payload = candidate_comparisons_to_dict(())
    assert payload == {"candidates": []}
