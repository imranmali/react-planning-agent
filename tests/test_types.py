"""Tests for the shared contract types."""

from __future__ import annotations

import json

import pytest

from react_agent.types import (
    Action,
    AgentConfig,
    AgentResult,
    ConfigError,
    Status,
    Step,
    TerminationReason,
)


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_iterations": 0},
        {"time_budget_seconds": 0},
        {"max_repeats": 1},
        {"max_consecutive_errors": 0},
        {"max_revisions": -1},
        {"tool_timeout_seconds": 0},
        {"max_observation_chars": 0},
    ],
)
def test_invalid_config_is_rejected(overrides: dict[str, float]) -> None:
    with pytest.raises(ConfigError):
        AgentConfig(**overrides)  # type: ignore[arg-type]


def test_time_budget_can_be_disabled() -> None:
    assert AgentConfig(time_budget_seconds=None).time_budget_seconds is None


def test_action_signature_normalises_case_and_whitespace() -> None:
    assert Action("Lookup", "  France   Population ").signature == ("lookup", "france population")
    assert Action("FINISH", "x").is_finish
    assert str(Action("calculator", "1+1")) == "calculator[1+1]"


def test_result_round_trips_to_json() -> None:
    step = Step(1, "think", None, "bad output", "follow the format", is_error=True)
    result = AgentResult(
        "partial", Status.DEGRADED, TerminationReason.ERROR_LIMIT, [step], iterations=1
    )

    document = json.loads(result.to_json())

    assert not result.succeeded
    assert document["steps"][0]["action"] is None
    assert document["termination_reason"] == "error_limit"
