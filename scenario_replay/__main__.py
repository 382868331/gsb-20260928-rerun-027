"""Command line: replay scenarios and verify saved run records.

Examples:
    python -m scenario_replay run examples/intersection.json --until 6
    python -m scenario_replay verify examples/last_run.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .engine import ReplayEngine
from .records import save_run, verify_run
from .scenario import load_scenario


def _cmd_run(args) -> int:
    engine = ReplayEngine(load_scenario(args.scenario))
    if args.seed is not None:
        engine.scenario.seed = args.seed
        engine.rng.seed(args.seed)
    engine.run_until(args.until)
    for record in engine.log:
        print(f"t={record.time:6.2f} #{record.seq} {record.event_id} "
              f"({record.etype}) {record.payload}")
    if engine.interrupted:
        print(f"INTERRUPTED at t={engine.clock.time}: "
              f"{engine.interrupt_reason}")
    else:
        print(f"clock at t={engine.clock.time} (tick {engine.clock.tick})")
    for pid, participant in engine.participants.items():
        state = participant.state()
        print(f"  {pid}: {state}")
    if args.out:
        path = save_run(engine, Path(args.out))
        print(f"saved run record -> {path}")
    return 0


def _cmd_verify(args) -> int:
    report = verify_run(args.record)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="scenario_replay")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="advance a scenario on the clock")
    run_p.add_argument("scenario")
    run_p.add_argument("--until", type=float, required=True,
                       help="target scenario time in seconds")
    run_p.add_argument("--seed", type=int, default=None)
    run_p.add_argument("--out", default=None, help="save run record here")
    run_p.set_defaults(func=_cmd_run)

    ver_p = sub.add_parser("verify", help="replay and compare a run record")
    ver_p.add_argument("record")
    ver_p.set_defaults(func=_cmd_verify)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
