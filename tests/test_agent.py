"""Integration tests for the observe/think/act/reflect loop and its exit paths."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence

import pytest

from react_agent.agent import ReActAgent
from react_agent.critic import LLMCritic
from react_agent.llm import OfflinePlannerLLM, ScriptedLLM, StubbornLLM
from react_agent.tools import Tool, ToolRegistry, default_registry
from react_agent.types import (
    AgentConfig,
    ConfigError,
    LLMError,
    Message,
    Status,
    Step,
    TerminationReason,
    ToolError,
)


def no_sleep(_: float) -> None:
    """Keep retry backoff out of the test suite's wall-clock time."""


def make_agent(
    responses: Sequence[str | Exception],
    *,
    tools: ToolRegistry | None = None,
    **config: int | float | None,
) -> tuple[ReActAgent, ScriptedLLM]:
    llm = ScriptedLLM(responses)
    agent = ReActAgent(llm, tools, AgentConfig(**config), sleep=no_sleep)  # type: ignore[arg-type]
    return agent, llm


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class CountingTool:
    def __init__(self, output: str = "ok") -> None:
        self.calls = 0
        self.output = output

    def __call__(self, text: str) -> str:
        self.calls += 1
        return f"{self.output}:{text}"


# --------------------------------------------------------------------------- happy path


def test_single_tool_call_then_finish() -> None:
    agent, _ = make_agent(
        ["Thought: add\nAction: calculator[2 + 3]", "Thought: done\nFinal Answer: 5"]
    )

    result = agent.run("What is 2 + 3?")

    assert result.answer == "5"
    assert result.status is Status.SUCCESS
    assert result.termination_reason is TerminationReason.FINAL_ANSWER
    assert result.succeeded
    assert len(result.steps) == 2
    first = result.steps[0]
    assert first.thought == "add"
    assert str(first.action) == "calculator[2 + 3]"
    assert first.observation == "5"
    assert first.reflection
    assert not first.is_error
    assert result.steps[1].reflection == "Self-critique approved the answer."


def test_observation_and_reflection_feed_the_next_thought() -> None:
    agent, llm = make_agent(
        ["Thought: add\nAction: calculator[2 + 3]", "Thought: done\nFinal Answer: 5"]
    )

    result = agent.run("What is 2 + 3?")

    second_prompt = llm.calls[1]
    rendered = "\n".join(message.content for message in second_prompt)
    assert "Observation: 5" in rendered
    assert result.steps[0].reflection in rendered


def test_json_model_output_is_accepted() -> None:
    agent, _ = make_agent(['{"thought": "done", "action": "finish", "input": "42"}'])

    result = agent.run("What is the answer?")

    assert result.answer == "42"
    assert result.status is Status.SUCCESS


def test_result_is_json_serialisable() -> None:
    agent, _ = make_agent(["Action: calculator[6*7]", "Final Answer: 42"])

    document = json.loads(agent.run("6*7?").to_json())

    assert document["answer"] == "42"
    assert document["status"] == "success"
    assert document["termination_reason"] == "final_answer"
    assert document["iterations"] == 2
    assert document["elapsed_seconds"] >= 0
    assert document["steps"][0]["action"] == {"tool": "calculator", "input": "6*7"}


def test_on_step_callback_receives_every_step() -> None:
    agent, _ = make_agent(["Action: calculator[1+1]", "Final Answer: 2"])
    seen: list[Step] = []

    result = agent.run("1+1?", on_step=seen.append)

    assert seen == result.steps


# --------------------------------------------------------------------------- limits


def test_max_iterations_stops_a_model_that_never_finishes() -> None:
    responses = [f"Thought: try more\nAction: calculator[{i} + 1]" for i in range(10)]
    agent, llm = make_agent(responses, max_iterations=4)

    result = agent.run("Count forever")

    assert result.termination_reason is TerminationReason.MAX_ITERATIONS
    assert result.iterations == 4
    assert len(llm.calls) == 4
    assert result.status is Status.DEGRADED
    assert "4" in result.answer  # last successful observation: 3 + 1


def test_invalid_iteration_limit_is_rejected_before_running() -> None:
    with pytest.raises(ConfigError):
        AgentConfig(max_iterations=0)


def test_time_budget_stops_the_run() -> None:
    clock = FakeClock()

    class SlowLLM:
        name = "slow"

        def complete(self, messages: Sequence[Message]) -> str:
            clock.now += 1.5
            return "Action: calculator[1 + 1]"

    agent = ReActAgent(SlowLLM(), config=AgentConfig(time_budget_seconds=1.0), clock=clock)

    result = agent.run("anything")

    assert result.termination_reason is TerminationReason.TIME_BUDGET
    assert result.iterations == 1
    assert result.status is Status.DEGRADED


def test_repeated_action_is_served_from_cache_with_a_warning() -> None:
    tool = CountingTool()
    registry = ToolRegistry()
    registry.register(Tool("lookup", "test lookup", tool))
    agent, _ = make_agent(
        ["Action: lookup[france]", "Action: lookup[ France ]", "Final Answer: ok:france"],
        tools=registry,
    )

    result = agent.run("France?")

    assert tool.calls == 1
    assert "already" in result.steps[1].observation.lower()
    assert "ok:france" in result.steps[1].observation
    assert result.status is Status.SUCCESS


def test_third_identical_action_stops_with_loop_detected() -> None:
    agent = ReActAgent(StubbornLLM("calculator[1 + 1]"), config=AgentConfig(max_repeats=3))

    result = agent.run("anything")

    assert result.termination_reason is TerminationReason.LOOP_DETECTED
    assert result.iterations == 3
    assert result.status is Status.DEGRADED
    assert result.answer.endswith("2")


def test_alternating_cycle_is_stopped() -> None:
    registry = ToolRegistry()
    registry.register(Tool("lookup", "test lookup", CountingTool()))
    pattern = ["Action: lookup[a]", "Action: lookup[b]"] * 3
    agent, _ = make_agent(pattern, tools=registry, max_iterations=10)

    result = agent.run("cycle")

    assert result.termination_reason is TerminationReason.LOOP_DETECTED
    assert result.iterations == 5


def test_consecutive_errors_stop_the_run() -> None:
    agent, _ = make_agent(["Action: teleport[x]", "gibberish", "Action: fly[y]", "Final Answer: x"])

    result = agent.run("anything")

    assert result.termination_reason is TerminationReason.ERROR_LIMIT
    assert result.status is Status.FAILED
    assert "too many consecutive steps failed" in result.answer
    assert all(step.is_error for step in result.steps)


def test_successful_step_resets_the_error_counter() -> None:
    agent, _ = make_agent(
        [
            "Action: teleport[x]",
            "Action: teleport[y]",
            "Action: calculator[1 + 1]",
            "Action: teleport[z]",
            "Final Answer: 2",
        ]
    )

    result = agent.run("anything")

    assert result.status is Status.SUCCESS
    assert [step.is_error for step in result.steps] == [True, True, False, True, False]


def test_unparseable_output_explains_the_expected_format() -> None:
    agent, _ = make_agent(["I am not sure what to do", "Final Answer: done"])

    result = agent.run("anything")

    failed = result.steps[0]
    assert failed.is_error
    assert failed.action is None
    assert "Action:" in failed.observation
    assert result.status is Status.SUCCESS


def test_unknown_tool_lists_available_tools() -> None:
    agent, _ = make_agent(["Action: teleport[x]", "Final Answer: done"])

    result = agent.run("anything")

    assert "calculator" in result.steps[0].observation
    assert "lookup" in result.steps[0].observation


# --------------------------------------------------------------------------- self-critique


def test_reflection_reports_failed_tool() -> None:
    agent, _ = make_agent(["Action: calculator[1/0]", "Final Answer: undefined"])

    result = agent.run("1/0?")

    assert result.steps[0].is_error
    assert "fail" in result.steps[0].reflection.lower()


def test_empty_answer_is_rejected_and_revised() -> None:
    agent, llm = make_agent(
        ["Action: calculator[2 + 2]", "Final Answer: ", "Thought: fix\nFinal Answer: 4"]
    )

    result = agent.run("2 + 2?")

    assert result.answer == "4"
    assert result.status is Status.SUCCESS
    rejection = result.steps[1]
    assert "rejected" in rejection.observation
    assert "rejected" in "\n".join(m.content for m in llm.calls[2])


def test_revisions_are_bounded() -> None:
    agent, _ = make_agent(
        ["Action: calculator[2 + 2]"] + ["Final Answer: unknown"] * 3,
        max_revisions=2,
    )

    result = agent.run("2 + 2?")

    assert result.answer == "unknown"
    assert result.status is Status.DEGRADED
    assert result.termination_reason is TerminationReason.FINAL_ANSWER
    assert any("rejected" in note for note in result.notes)


def test_critic_failure_fails_open() -> None:
    critic_llm = ScriptedLLM([LLMError("critic down")])
    agent = ReActAgent(
        ScriptedLLM(["Action: calculator[2 + 2]", "Final Answer: 4"]),
        critic=LLMCritic(critic_llm),
    )

    result = agent.run("2 + 2?")

    assert result.status is Status.SUCCESS
    assert any("unavailable" in note.lower() for note in result.notes)


# --------------------------------------------------------------------------- degradation


def test_tool_exception_becomes_an_observation() -> None:
    def explode(_: str) -> str:
        raise RuntimeError("kaboom")

    registry = ToolRegistry()
    registry.register(Tool("explode", "always fails", explode))
    agent, _ = make_agent(["Action: explode[x]", "Final Answer: gave up"], tools=registry)

    result = agent.run("anything")

    assert result.steps[0].is_error
    assert "kaboom" in result.steps[0].observation
    assert result.status is Status.SUCCESS


def test_tool_error_message_is_reported() -> None:
    def reject(_: str) -> str:
        raise ToolError("bad input")

    registry = ToolRegistry()
    registry.register(Tool("strict", "rejects input", reject))
    agent, _ = make_agent(["Action: strict[x]", "Final Answer: n/a"], tools=registry)

    assert "bad input" in agent.run("x").steps[0].observation


def test_slow_tool_times_out() -> None:
    def slow(_: str) -> str:
        time.sleep(2)
        return "late"

    registry = ToolRegistry()
    registry.register(Tool("slow", "sleeps", slow))
    agent, _ = make_agent(
        ["Action: slow[x]", "Final Answer: gave up"], tools=registry, tool_timeout_seconds=0.05
    )

    started = time.monotonic()
    result = agent.run("anything")

    assert time.monotonic() - started < 1.5
    assert result.steps[0].is_error
    assert "time" in result.steps[0].observation.lower()


def test_primary_model_recovers_on_retry() -> None:
    agent, _ = make_agent([LLMError("blip"), "Final Answer: fine"])

    result = agent.run("anything")

    assert result.answer == "fine"
    assert result.status is Status.SUCCESS


def test_fallback_model_is_used_and_noted() -> None:
    primary = ScriptedLLM([LLMError("down")] * 3, name="primary")
    fallback = ScriptedLLM(["Final Answer: from fallback"], name="backup")
    agent = ReActAgent(primary, fallbacks=[fallback], sleep=no_sleep)

    result = agent.run("anything")

    assert result.answer == "from fallback"
    assert any("backup" in note for note in result.notes)


def test_all_models_unavailable_without_progress_fails() -> None:
    agent, _ = make_agent([LLMError("down")] * 3)

    result = agent.run("anything")

    assert result.termination_reason is TerminationReason.LLM_UNAVAILABLE
    assert result.status is Status.FAILED
    assert "no language model was available" in result.answer


def test_model_lost_after_progress_returns_best_effort_answer() -> None:
    agent, _ = make_agent(["Action: calculator[6 * 7]", *[LLMError("down")] * 3])

    result = agent.run("6 * 7?")

    assert result.termination_reason is TerminationReason.LLM_UNAVAILABLE
    assert result.status is Status.DEGRADED
    assert "42" in result.answer


def test_notes_do_not_leak_between_runs() -> None:
    primary = ScriptedLLM([LLMError("down")] * 3 + ["Final Answer: two"], name="primary")
    fallback = ScriptedLLM(["Final Answer: one"], name="backup")
    agent = ReActAgent(primary, fallbacks=[fallback], sleep=no_sleep)

    first = agent.run("first")
    second = agent.run("second")

    assert first.notes
    assert second.notes == []
    assert second.answer == "two"


# --------------------------------------------------------------------------- offline planner


@pytest.mark.parametrize(
    ("question", "answer"),
    [
        ("What is 12 * 7?", "84"),
        ("What is the capital of japan?", "Tokyo"),
        ("What is the france population times 2?", "136340000"),
    ],
)
def test_offline_planner_end_to_end(question: str, answer: str) -> None:
    agent = ReActAgent(OfflinePlannerLLM(), default_registry())

    result = agent.run(question)

    assert result.status is Status.SUCCESS
    assert result.answer == answer
