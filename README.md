# 自动驾驶场景回放系统 (Scenario Replay)

A deterministic, discrete-time replay core for autonomous-driving scenarios.
Vehicles, pedestrians and traffic signals advance on **one shared scenario
clock**; it is the same clock a later perception module would consume.
This is a simulation/recording engine, not an animation.

## Design

- `scenario_replay/clock.py` — `ScenarioClock`: fixed-timestep, pausable,
  relocat-able clock.
- `scenario_replay/participants.py` — `Vehicle`, `Pedestrian`,
  `TrafficSignal`; each integrates motion per tick using a seeded RNG.
- `scenario_replay/events.py` — scheduled and rule-based `Event`s with a
  fixed total order at one instant `(tick, order, event_id)`; immutable
  `EventRecord`s and an `InterruptScenario` signal.
- `scenario_replay/engine.py` — `ReplayEngine`: steps participants, updates
  signals, dispatches due events, and snapshots full state every tick.
- `scenario_replay/records.py` — saves/loads a run record (schema version,
  seed, event log, per-tick positions) and re-verifies determinism.
- `scenario_replay/scenario.py` / `builders.py` — JSON scenario loading and
  declarative event construction.

## Guarantees

1. **One clock**: every participant updates from the same `step_seconds`.
2. **Pause / seek**: pause blocks bulk runs but not an explicit `step()` or
   a `seek()`; backward seeks restore an exact per-tick snapshot (including
   RNG state and the processed-event set).
3. **No re-triggering**: an event already handled before the relocated
   moment is in the restored processed set and never fires twice.
4. **Step == jump**: forward `seek()` replays ticks via the same `step()`
   path as a tick-by-tick run; positions and the event log are identical.
5. **Fixed event order**: simultaneous events dispatch by
   `(tick, order, event_id)` and get a monotonic sequence number.
6. **Interruptions**: an interrupt event pauses the scenario and is logged;
   `resume()` continues from the same instant.
7. **Versioned + seeded runs**: the schema version and seed are pinned in
   the scenario and the saved run record; `verify_run` re-runs and compares
   events and positions exactly.

## Usage

```bash
python -m scenario_replay run examples/intersection.json --until 6
python -m scenario_replay run examples/intersection.json --until 6 --out run.json
python -m scenario_replay verify run.json
python examples/run_demo.py
pytest -q
```

## Programmatic example

```python
from scenario_replay import ReplayEngine, load_scenario

engine = ReplayEngine(load_scenario("examples/intersection.json"))
engine.run_until(6.0)           # bulk advance (honours pause/interrupt)
engine.seek(2.0)                # jump back to a past moment
engine.run_until(6.0)           # continue: no duplicate events
assert engine.positions        # per-tick participant states
assert engine.log              # immutable, ordered event records
```
