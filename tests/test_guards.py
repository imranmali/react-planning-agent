"""Tests for execution guards (spec: execution-limits)."""

from __future__ import annotations

from typing import Any

import pytest

from react_agent.guards import IterationGuard, LoopDetector, repeat_warning
from react_agent.types import Action, AgentConfig, ConfigError, LoopVerdict, TerminationReason


class FakeClock:
    """A manually advanced clock."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def make_guard(clock: FakeClock | None = None, **overrides: Any) -> IterationGuard:
    config = AgentConfig(**overrides)
    return IterationGuard(config, clock=clock or FakeClock())


# --------------------------------------------------------------------------- IterationGuard


def test_model_never_finishes_stops_after_exactly_max_iterations() -> None:
    guard = make_guard(max_iterations=4, time_budget_seconds=None)
    guard.start()
    executed = 0
    while (reason := guard.check()) is None:
        guard.record_iteration()
        guard.record_step(is_error=False)
        executed += 1
    assert executed == 4
    assert guard.iterations == 4
    assert reason is TerminationReason.MAX_ITERATIONS
    assert guard.remaining_iterations == 0


def test_invalid_max_iterations_rejected_before_any_run() -> None:
    with pytest.raises(ConfigError, match="max_iterations"):
        AgentConfig(max_iterations=0)


def test_time_budget_exhausted_stops_next_iteration() -> None:
    clock = FakeClock()
    guard = make_guard(clock, time_budget_seconds=1.0)
    guard.start()
    assert guard.check() is None
    guard.record_iteration()
    clock.advance(1.5)
    assert guard.elapsed == pytest.approx(1.5)
    assert guard.check() is TerminationReason.TIME_BUDGET


def test_time_budget_boundary_is_inclusive() -> None:
    clock = FakeClock()
    guard = make_guard(clock, time_budget_seconds=1.0)
    guard.start()
    clock.advance(0.999)
    assert guard.check() is None
    clock.advance(0.001)
    assert guard.check() is TerminationReason.TIME_BUDGET


def test_no_time_budget_never_times_out() -> None:
    clock = FakeClock()
    guard = make_guard(clock, time_budget_seconds=None)
    guard.start()
    clock.advance(1e9)
    assert guard.check() is None


def test_three_consecutive_errors_hit_error_limit() -> None:
    guard = make_guard(max_consecutive_errors=3)
    guard.start()
    for _ in range(3):
        assert guard.check() is None
        guard.record_iteration()
        guard.record_step(is_error=True)
    assert guard.consecutive_errors == 3
    assert guard.check() is TerminationReason.ERROR_LIMIT


def test_success_resets_consecutive_error_count() -> None:
    guard = make_guard(max_consecutive_errors=3)
    guard.start()
    for is_error in (True, True, False, True):
        assert guard.check() is None
        guard.record_iteration()
        guard.record_step(is_error=is_error)
    assert guard.consecutive_errors == 1
    assert guard.check() is None


def test_priority_error_limit_then_time_budget_then_max_iterations() -> None:
    clock = FakeClock()
    guard = make_guard(clock, max_iterations=1, time_budget_seconds=1.0, max_consecutive_errors=1)
    guard.start()
    guard.record_iteration()
    clock.advance(5.0)
    guard.record_step(is_error=True)
    assert guard.check() is TerminationReason.ERROR_LIMIT
    guard.record_step(is_error=False)
    assert guard.check() is TerminationReason.TIME_BUDGET


def test_check_auto_starts_the_clock() -> None:
    clock = FakeClock(now=50.0)
    guard = make_guard(clock, time_budget_seconds=1.0)
    assert guard.elapsed == 0.0
    clock.advance(10.0)  # time before the first check does not count
    assert guard.check() is None
    clock.advance(0.25)
    assert guard.elapsed == pytest.approx(0.25)


def test_start_resets_counters_and_clock() -> None:
    clock = FakeClock()
    guard = make_guard(clock, max_iterations=2)
    guard.start()
    guard.record_iteration()
    guard.record_step(is_error=True)
    clock.advance(3.0)
    guard.start()
    assert guard.iterations == 0
    assert guard.consecutive_errors == 0
    assert guard.elapsed == 0.0
    assert guard.remaining_iterations == 2


def test_default_clock_is_monotonic() -> None:
    guard = IterationGuard(AgentConfig())
    assert guard.check() is None
    assert guard.elapsed >= 0.0


# --------------------------------------------------------------------------- LoopDetector


def test_second_request_is_repeat_and_third_is_loop() -> None:
    detector = LoopDetector(max_repeats=3)
    action = Action("lookup", "france")
    assert detector.register(action) is LoopVerdict.NEW
    detector.remember(action, "Paris")
    assert detector.register(action) is LoopVerdict.REPEAT
    assert detector.cached(action) == "Paris"
    assert detector.register(action) is LoopVerdict.LOOP
    assert detector.count(action) == 3


def test_alternating_cycle_is_detected() -> None:
    detector = LoopDetector(max_repeats=3)
    a, b = Action("lookup", "a"), Action("lookup", "b")
    verdicts = [detector.register(action) for action in (a, b, a, b, a)]
    assert verdicts == [
        LoopVerdict.NEW,
        LoopVerdict.NEW,
        LoopVerdict.REPEAT,
        LoopVerdict.REPEAT,
        LoopVerdict.LOOP,
    ]


@pytest.mark.parametrize(
    "variant",
    [Action("lookup", " France "), Action("LOOKUP", "france"), Action(" Lookup ", "FRANCE")],
)
def test_signature_normalisation_matches_variants(variant: Action) -> None:
    detector = LoopDetector()
    detector.register(Action("lookup", "france"))
    detector.remember(Action("lookup", "france"), "Paris")
    assert detector.register(variant) is LoopVerdict.REPEAT
    assert detector.cached(variant) == "Paris"


def test_distinct_actions_are_new() -> None:
    detector = LoopDetector()
    assert detector.register(Action("lookup", "france")) is LoopVerdict.NEW
    assert detector.register(Action("calculator", "france")) is LoopVerdict.NEW
    assert detector.register(Action("lookup", "germany")) is LoopVerdict.NEW


def test_unknown_action_has_no_cache_and_zero_count() -> None:
    detector = LoopDetector()
    action = Action("lookup", "x")
    assert detector.cached(action) is None
    assert detector.count(action) == 0


def test_max_repeats_two_loops_on_first_repeat() -> None:
    detector = LoopDetector(max_repeats=2)
    action = Action("calculator", "2+2")
    assert detector.register(action) is LoopVerdict.NEW
    assert detector.register(action) is LoopVerdict.LOOP


def test_finish_actions_are_treated_normally_if_registered() -> None:
    detector = LoopDetector()
    finish = Action("finish", "42")
    assert detector.register(finish) is LoopVerdict.NEW
    assert detector.register(finish) is LoopVerdict.REPEAT


@pytest.mark.parametrize("max_repeats", [1, 0, -3])
def test_invalid_max_repeats_rejected(max_repeats: int) -> None:
    with pytest.raises(ConfigError, match="max_repeats"):
        LoopDetector(max_repeats=max_repeats)


def test_repeat_warning_text() -> None:
    text = repeat_warning(Action("calculator", "2+2"), "4")
    assert text == (
        "WARNING: you already performed calculator[2+2]. Do not repeat it; "
        "use the earlier result. Earlier observation: 4"
    )
