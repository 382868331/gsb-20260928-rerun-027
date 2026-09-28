"""The single scenario clock shared by every participant and algorithm."""

from __future__ import annotations


class ScenarioClock:
    """Discrete, pausable clock measured in scenario seconds.

    Time only moves forward in fixed ``step_seconds`` ticks.  ``paused``
    blocks bulk :meth:`ReplayEngine.run_until` advances but never blocks an
    explicit single tick or a relocation request.
    """

    def __init__(self, step_seconds: float = 0.5, start: float = 0.0):
        if step_seconds <= 0:
            raise ValueError("step_seconds must be positive")
        self.step_seconds = float(step_seconds)
        self.tick = 0
        self.time = float(start)
        self._start = float(start)
        self.paused = False

    @property
    def start_time(self) -> float:
        return self._start

    def advance(self) -> float:
        self.tick += 1
        self.time = round(self._start + self.tick * self.step_seconds, 9)
        return self.time

    def restore(self, tick: int, time: float, paused: bool) -> None:
        self.tick = int(tick)
        self.time = float(time)
        self.paused = bool(paused)

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def tick_for(self, time: float) -> int:
        delta = time - self._start
        n = round(delta / self.step_seconds)
        if abs(delta - n * self.step_seconds) > 1e-9:
            raise ValueError(
                f"time {time} is not aligned to step {self.step_seconds}"
            )
        return n

    def snapshot(self) -> dict:
        return {
            "tick": self.tick, "time": self.time, "paused": self.paused,
            "start": self._start, "step": self.step_seconds,
        }
