"""The replay engine: one clock drives motion, rules and signals."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Optional

from .clock import ScenarioClock
from .builders import build_event
from .events import Event, EventRecord, InterruptScenario
from .participants import PHASE_CYCLE, TrafficSignal
from .scenario import SCENARIO_SCHEMA_VERSION, Scenario


@dataclass
class EngineConfig:
    capture_history: bool = True  # keep every tick snapshot for seeking


@dataclass
class _Snapshot:
    tick: int
    time: float
    clock_paused: bool
    rng_state: object
    participants: dict
    events: list
    processed: set
    log: list
    positions: list
    seq: int
    interrupted: bool
    interrupt_reason: Optional[str]


class ReplayEngine:
    """Advances a scenario deterministically and supports time relocation.

    The engine state at each tick is snapshotted.  :meth:`seek` restores a
    past snapshot (already fired events are therefore forgotten in the
    restored state, but retained in history up to that tick) or replays
    forward.  A forward jump executes the *same* code path as repeated
    :meth:`step` calls, so step-wise runs and jumps agree exactly.
    """

    def __init__(self, scenario: Scenario, config: Optional[EngineConfig] = None):
        self.scenario = scenario
        self.config = config or EngineConfig()
        self.clock = ScenarioClock(scenario.step_seconds)
        self.rng = random.Random(scenario.seed)
        self.participants = {p.pid: p for p in scenario.participants.values()}
        self.events: list[Event] = [
            build_event(spec, scenario.step_seconds)
            for spec in scenario.events]
        self.processed: set[str] = set()
        self.log: list[EventRecord] = []
        self.positions: list[dict] = []
        self.history: dict[int, _Snapshot] = {}
        self.seq = 0
        self.interrupted = False
        self.interrupt_reason: Optional[str] = None
        self._save_snapshot()

    # -- event management --------------------------------------------------

    def add_event(self, event: Event) -> None:
        if any(e.event_id == event.event_id for e in self.events):
            raise ValueError(f"duplicate event_id {event.event_id!r}")
        self.events.append(event)

    # -- core stepping -----------------------------------------------------

    def step(self) -> Optional[EventRecord]:
        """Advance exactly one tick; allowed even while paused."""
        if self.interrupted:
            return None
        if self.clock.tick >= self.scenario.end_tick():
            return None
        dt = self.clock.step_seconds
        # Fixed participant update order: vehicles, pedestrians, signals.
        for participant in self._ordered_participants():
            participant.step(dt, self.rng)
        self.clock.advance()
        self._update_signal_cycles()
        return self._dispatch_due_events()

    def _ordered_participants(self):
        rank = {"vehicle": 0, "pedestrian": 1, "signal": 2}
        return sorted(self.participants.values(),
                      key=lambda p: (rank.get(p.kind, 9), p.pid))

    def _update_signal_cycles(self) -> None:
        tick = self.clock.tick
        for sig in self.participants.values():
            if isinstance(sig, TrafficSignal) and sig.auto:
                idx = (tick // max(sig.phase_ticks, 1)) % len(PHASE_CYCLE)
                sig.set_phase(PHASE_CYCLE[idx])

    def _dispatch_due_events(self) -> Optional[EventRecord]:
        """Fire every event due at this tick in a fixed total order."""
        tick = self.clock.tick
        due = [e for e in self.events
               if e.event_id not in self.processed and e.is_due(tick, self)]
        due.sort(key=lambda e: e.sort_key)
        first_record = None
        for event in due:
            record = self._fire(event)
            if record is not None and first_record is None:
                first_record = record
            if self.interrupted:
                break
        self._record_positions()
        self._save_snapshot()
        return first_record

    def _fire(self, event: Event) -> Optional[EventRecord]:
        payload = dict(event.payload)
        try:
            if event.effect is not None:
                result = event.effect(self)
                if isinstance(result, dict):
                    payload.update(result)
        except InterruptScenario as exc:
            payload.update(exc.payload)
            payload["reason"] = exc.reason
            self.interrupted = True
            self.interrupt_reason = exc.reason
            self.clock.pause()
        self.seq += 1
        record = EventRecord(self.clock.tick, self.clock.time, self.seq,
                             event.event_id, event.etype, payload)
        self.log.append(record)
        self.processed.add(event.event_id)
        if event.recurring and not event.once:
            # Re-arm recurring events for a later tick; they keep their id
            # but a fresh instance will match future conditions.
            event.tick = self.clock.tick + 1
            self.processed.discard(event.event_id)
        return record

    # -- pause / run / seek ------------------------------------------------

    def pause(self) -> None:
        self.clock.pause()
        self._save_snapshot()

    def resume(self) -> None:
        """Clear an interruption or paused state and continue."""
        self.interrupted = False
        self.interrupt_reason = None
        self.clock.resume()
        self._save_snapshot()

    def run_until(self, target_time: float) -> list[EventRecord]:
        """Bulk advance while not paused/interrupted; returns fired events."""
        fired: list[EventRecord] = []
        target_tick = self.clock.tick_for(target_time)
        while self.clock.tick < target_tick:
            if self.clock.paused or self.interrupted:
                break
            before = len(self.log)
            self.step()
            fired.extend(self.log[before:])
        return fired

    def seek(self, target_time: float) -> int:
        """Relocate the clock to target_time.

        Backward seeks restore the exact tick snapshot, including the RNG and
        the processed-event set, so events already handled never fire again
        when continuing.  Forward seeks replay ticks through :meth:`step` and
        therefore match a tick-by-tick run precisely.  Seeking always works
        even when the clock is paused.
        """
        target_tick = self.clock.tick_for(target_time)
        if target_tick < 0 or target_tick > self.scenario.end_tick():
            raise ValueError("seek target outside scenario duration")
        if target_tick == self.clock.tick:
            return target_tick
        if target_tick < self.clock.tick:
            if target_tick not in self.history:
                raise RuntimeError("no snapshot for target tick; history lost")
            self._restore_snapshot(self.history[target_tick])
        else:
            saved_paused = self.clock.paused
            self.clock.resume()
            try:
                while self.clock.tick < target_tick and not self.interrupted:
                    self.step()
            finally:
                if saved_paused and not self.interrupted:
                    self.clock.pause()
        return target_tick

    # -- snapshots ---------------------------------------------------------

    def _record_positions(self) -> None:
        self.positions.append({"tick": self.clock.tick,
                              "time": self.clock.time,
                              "participants": {pid: p.state()
                                               for pid, p in
                                               self.participants.items()}})

    def _save_snapshot(self) -> None:
        if not self.config.capture_history:
            return
        self.history[self.clock.tick] = _Snapshot(
            tick=self.clock.tick, time=self.clock.time,
            clock_paused=self.clock.paused, rng_state=self.rng.getstate(),
            participants=copy.deepcopy(
                {pid: p.state() for pid, p in self.participants.items()}),
            events=copy.deepcopy(
                [(e.event_id, e.tick) for e in self.events]),
            processed=set(self.processed), log=copy.deepcopy(self.log),
            positions=copy.deepcopy(self.positions), seq=self.seq,
            interrupted=self.interrupted,
            interrupt_reason=self.interrupt_reason)

    def _restore_snapshot(self, snap: _Snapshot) -> None:
        self.clock.restore(snap.tick, snap.time, snap.clock_paused)
        self.rng.setstate(snap.rng_state)
        for pid, state in snap.participants.items():
            self.participants[pid].restore_state(state)
        for event, (_, tick) in zip(self.events, snap.events):
            event.tick = tick
        self.processed = set(snap.processed)
        self.log = copy.deepcopy(snap.log)
        self.positions = copy.deepcopy(snap.positions)
        self.seq = snap.seq
        self.interrupted = snap.interrupted
        self.interrupt_reason = snap.interrupt_reason

    # -- inspection --------------------------------------------------------

    def state_at(self, tick: Optional[int] = None) -> dict:
        """Participant states at a past tick (default: current)."""
        if tick is None or tick == self.clock.tick:
            return {pid: p.state() for pid, p in self.participants.items()}
        if tick not in self.history:
            raise KeyError(f"no snapshot for tick {tick}")
        return copy.deepcopy(self.history[tick].participants)

    def events_between(self, start_tick: int, end_tick: int) -> list[EventRecord]:
        return [r for r in self.log if start_tick <= r.tick <= end_tick]

    def metadata(self) -> dict:
        return {"schema_version": SCENARIO_SCHEMA_VERSION,
                "scenario": self.scenario.name,
                "seed": self.scenario.seed,
                "step_seconds": self.scenario.step_seconds}
