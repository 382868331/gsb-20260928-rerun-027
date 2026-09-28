"""Tiny demo: run, pause, rewind, continue and save the run record."""

from pathlib import Path

from scenario_replay import ReplayEngine, load_scenario, save_run

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    scenario = load_scenario(ROOT / "examples" / "intersection.json")
    engine = ReplayEngine(scenario)

    engine.run_until(3.5)
    print(f"t={engine.clock.time:5.1f} paused={engine.clock.paused} "
          f"interrupted={engine.interrupted}")
    for record in engine.log:
        print(" ", record)

    # Rewind to 3.0s after the scheduled events, then continue: the two
    # handled events must not fire again on the way back to 3.5s.
    engine.seek(3.0)
    handled = [r.event_id for r in engine.log]
    engine.resume()
    engine.run_until(3.5)
    ids = [r.event_id for r in engine.log]
    assert all(ids.count(i) == 1 for i in ids), "duplicate event after rewind"
    assert ids[:len(handled)] == handled
    print(f"rewind-and-continue: {handled} did not re-trigger")

    out = save_run(engine, ROOT / "examples" / "last_run.json")
    print(f"saved run record: {out}")


if __name__ == "__main__":
    main()
