import pytest

from scenario_replay import (
    PerceptionWindow,
    ReplayEngine,
    WindowConstraint,
    JointProposalIncompatible,
    compare_joint_revisions,
    compare_revision_candidates,
    confirm_joint_proposal,
    derive_revision_candidate,
    load_joint_proposal,
    load_run,
    save_run,
    stage_joint_proposal,
    stage_revision_candidate,
    withdraw_joint_candidate,
)
from tests.test_uncertainty import (
    brake_base_changes,
    uncertain_lead_brake_scenario,
)


def _brake_window(window_id, earliest, latest, delay=300):
    return PerceptionWindow(
        window_id,
        "follower",
        "lead_brake",
        "leader",
        earliest,
        latest,
        delay,
        observed={"note": window_id},
    )


def _signal_window(window_id, earliest, latest, delay=300):
    return PerceptionWindow(
        window_id,
        "follower",
        "signal_red",
        "j1",
        earliest,
        latest,
        delay,
        observed={"note": window_id},
    )


def _stage_pair():
    engine = ReplayEngine(uncertain_lead_brake_scenario())
    stage_revision_candidate(
        engine,
        "observer-a",
        [_brake_window("w-brake", 400, 500)],
        base_changes=brake_base_changes(),
        horizon_ms=2500,
    )
    stage_revision_candidate(
        engine,
        "observer-b",
        [_signal_window("w-signal", 400, 500)],
        base_changes=brake_base_changes(),
        horizon_ms=2500,
    )
    return engine


def test_joint_proposal_remains_pending_and_reports_range_conflict():
    engine = ReplayEngine(uncertain_lead_brake_scenario())
    stage_revision_candidate(
        engine,
        "a",
        [_brake_window("shared", 400, 400)],
        base_changes=brake_base_changes(),
    )
    stage_revision_candidate(
        engine,
        "b",
        [_brake_window("shared", 500, 500)],
        base_changes=brake_base_changes(),
    )
    proposal = stage_joint_proposal(engine, "joint", ["a", "b"])
    assert proposal.conflicts[0].kind == "shared_window_range"
    assert proposal.conflicts[0].candidate_ids == ("a", "b")
    assert proposal.conflicts[0].window_ids == ("shared",)
    assert not any(
        event.event_id.startswith("joint:")
        for event in engine.scenario.events
    )
    with pytest.raises(JointProposalIncompatible) as exc:
        confirm_joint_proposal(engine, "joint", evaluate_until_ms=2500)
    assert exc.value.proposal.conflicts[0].kind == "shared_window_range"


def test_joint_constraint_cycle_is_conflict_evidence_not_silent_deletion():
    engine = _stage_pair()
    proposal = stage_joint_proposal(
        engine,
        "cycle",
        ["observer-a", "observer-b"],
        joint_constraints=[
            WindowConstraint("after", "w-signal", "w-brake"),
            WindowConstraint("after", "w-brake", "w-signal"),
        ],
    )
    assert {conflict.kind for conflict in proposal.conflicts} == {
        "constraint_cycle"
    }
    conflict = proposal.conflicts[0]
    assert set(conflict.window_ids) == {"w-brake", "w-signal"}
    assert len(conflict.constraints) == 2
    with pytest.raises(JointProposalIncompatible):
        confirm_joint_proposal(engine, "cycle")


def test_confirm_joint_derives_one_timeline_and_preserves_provenance(tmp_path):
    engine = _stage_pair()
    proposal = stage_joint_proposal(
        engine,
        "joint-ab",
        ["observer-a", "observer-b"],
        joint_constraints=[
            WindowConstraint("after", "w-signal", "w-brake")
        ],
        horizon_ms=2500,
    )
    assert not proposal.conflicts
    assert dict(proposal.window_sources) == {
        "w-brake": ("observer-a",),
        "w-signal": ("observer-b",),
    }

    single_a = derive_revision_candidate(
        engine, "observer-a", adopt=False, evaluate_until_ms=2500
    )
    single_b = derive_revision_candidate(
        engine, "observer-b", adopt=False, evaluate_until_ms=2500
    )
    result = confirm_joint_proposal(
        engine, "joint-ab", evaluate_until_ms=2500
    )
    assert len(result.schemes) == 3
    assert len(result.constraint_violations) == 1
    for scheme in result.schemes:
        injected = [
            record
            for record in scheme.engine.perceptions
            if record.observed.get("injected")
        ]
        assert {record.stimulus for record in injected} == {
            "lead_brake",
            "signal_red",
        }
        assert all(
            record.observed["joint_proposal_id"] == "joint-ab"
            for record in injected
        )

    engine.run_to(2500)
    original_collisions = len(engine.collisions)
    single_comparisons = compare_revision_candidates(
        [single_a, single_b], until_ms=2500, base_engine=engine
    )
    joint_comparisons = compare_joint_revisions(
        [result], until_ms=2500, base_engine=engine
    )[0]
    assert joint_comparisons.explored_count == 3
    assert joint_comparisons.constraint_excluded_count == 1
    assert joint_comparisons.total_assignments == 4
    assert joint_comparisons.unexplored_count == 0
    assert all(
        collisions or scheme_id
        for scheme_id, collisions in joint_comparisons.collisions_by_scheme
    )
    assert original_collisions >= 0
    assert len(single_comparisons) == 2
    assert engine.selected_joint_proposal_id == "joint-ab"
    assert "observer-a" not in engine.pending_revisions
    assert "observer-b" not in engine.pending_revisions

    confirm_joint_proposal(engine, "joint-ab", evaluate_until_ms=2500)
    assert engine.joint_selection_history.count(None) == 1
    path = save_run(engine, tmp_path / "joint.json")
    reopened = load_run(path)
    reloaded = load_joint_proposal(reopened, "joint-ab")
    assert reloaded.candidate_ids == ("observer-a", "observer-b")
    assert dict(reloaded.window_sources)["w-signal"] == (
        "observer-b",
    )
    again = confirm_joint_proposal(
        reopened, "joint-ab", evaluate_until_ms=2500
    )
    assert [scheme.scheme_id for scheme in again.schemes] == [
        scheme.scheme_id for scheme in result.schemes
    ]


def test_partial_withdrawal_restores_only_lost_windows(tmp_path):
    engine = _stage_pair()
    stage_joint_proposal(
        engine,
        "joint-ab",
        ["observer-a", "observer-b"],
    )
    confirm_joint_proposal(engine, "joint-ab", evaluate_until_ms=2500)
    path = save_run(engine, tmp_path / "confirmed.json")

    reopened = load_run(path)
    partial = withdraw_joint_candidate(
        reopened, "joint-ab", "observer-b"
    )
    assert partial.restored_window_ids == ("w-signal",)
    assert [window.window_id for window in partial.windows] == [
        "w-brake"
    ]
    assert partial.candidate_ids == ("observer-a",)
    assert partial.withdrawn_candidate_ids == ("observer-b",)
    assert reopened.selected_joint_proposal_id is None
    assert "observer-b" not in reopened.adopted_revisions
    assert "observer-b" in reopened.pending_revisions
    assert "observer-a" in reopened.adopted_revisions
    original = load_joint_proposal(reopened, "joint-ab")
    assert original.restored_window_ids == ("w-signal",)
    assert any(
        conflict.kind == "restored_pending"
        for conflict in original.conflicts
    )
