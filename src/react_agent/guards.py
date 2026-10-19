"""Execution guards: the only place that decides when a run must stop.

``IterationGuard`` enforces the iteration limit, the wall-clock time budget and the
consecutive-error limit. ``LoopDetector`` classifies repeated actions so the agent can
serve cached observations for repeats and stop on loops (design D3).
"""

from __future__ import annotations

import time
from collections import Counter
from collections.abc import Callable

from react_agent.types import Action, AgentConfig, ConfigError, LoopVerdict, TerminationReason

Clock = Callable[[], float]
"""Returns the current time in seconds; only differences between calls matter."""


class IterationGuard:
    """Tracks iterations, elapsed time and consecutive errors for one run.

    Call ``check()`` at the top of every iteration; a non-``None`` result is the reason
    the run must stop. ``check()`` starts the clock automatically if ``start()`` was not
    called, so a guard is always usable.
    """

    def __init__(self, config: AgentConfig, clock: Clock = time.monotonic) -> None:
        self._config = config
        self._clock = clock
        self._started_at: float | None = None
        self._iterations = 0
        self._consecutive_errors = 0

    def start(self) -> None:
        """Start (or restart) the clock and reset all counters."""
        self._started_at = self._clock()
        self._iterations = 0
        self._consecutive_errors = 0

    def check(self) -> TerminationReason | None:
        """Return why the run must stop now, or ``None`` if another iteration may start.

        Priority: ``ERROR_LIMIT``, then ``TIME_BUDGET``, then ``MAX_ITERATIONS``.
        """
        if self._started_at is None:
            self.start()
        if self._consecutive_errors >= self._config.max_consecutive_errors:
            return TerminationReason.ERROR_LIMIT
        budget = self._config.time_budget_seconds
        if budget is not None and self.elapsed >= budget:
            return TerminationReason.TIME_BUDGET
        if self._iterations >= self._config.max_iterations:
            return TerminationReason.MAX_ITERATIONS
        return None

    def record_iteration(self) -> None:
        """Count one started think/act cycle."""
        self._iterations += 1

    def record_step(self, *, is_error: bool) -> None:
        """Record a step outcome: errors accumulate, any success resets the count."""
        self._consecutive_errors = self._consecutive_errors + 1 if is_error else 0

    @property
    def iterations(self) -> int:
        return self._iterations

    @property
    def consecutive_errors(self) -> int:
        return self._consecutive_errors

    @property
    def elapsed(self) -> float:
        """Seconds since ``start()``; 0.0 before the guard has started."""
        if self._started_at is None:
            return 0.0
        return self._clock() - self._started_at

    @property
    def remaining_iterations(self) -> int:
        return max(0, self._config.max_iterations - self._iterations)


class LoopDetector:
    """Counts identical actions (same tool, normalised input) over the whole run.

    Counting total occurrences rather than consecutive ones also bounds alternating
    cycles such as A, B, A, B, A.
    """

    def __init__(self, max_repeats: int = 3) -> None:
        if max_repeats < 2:
            raise ConfigError("max_repeats must be >= 2")
        self.max_repeats = max_repeats
        self._counts: Counter[tuple[str, str]] = Counter()
        self._observations: dict[tuple[str, str], str] = {}

    def register(self, action: Action) -> LoopVerdict:
        """Count one occurrence of ``action`` and classify it."""
        signature = action.signature
        self._counts[signature] += 1
        count = self._counts[signature]
        if count >= self.max_repeats:
            return LoopVerdict.LOOP
        if count > 1:
            return LoopVerdict.REPEAT
        return LoopVerdict.NEW

    def remember(self, action: Action, observation: str) -> None:
        """Store the observation produced by ``action`` for later repeats."""
        self._observations[action.signature] = observation

    def cached(self, action: Action) -> str | None:
        """Return the stored observation for ``action``, if any."""
        return self._observations.get(action.signature)

    def count(self, action: Action) -> int:
        """Return how many times ``action`` has been registered."""
        return self._counts[action.signature]


def repeat_warning(action: Action, cached: str) -> str:
    """Observation text served instead of re-executing a repeated action."""
    return (
        f"WARNING: you already performed {action}. Do not repeat it; "
        f"use the earlier result. Earlier observation: {cached}"
    )
