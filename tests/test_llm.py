"""Tests for the model backends in ``react_agent.llm`` (no network, no SDK needed)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from react_agent.llm import (
    DEFAULT_ANTHROPIC_MODEL,
    AnthropicLLM,
    OfflinePlannerLLM,
    ResilientLLM,
    ScriptedLLM,
    StubbornLLM,
    find_expression,
)
from react_agent.prompts import build_messages
from react_agent.types import (
    ConfigError,
    LLMClient,
    LLMError,
    LLMUnavailableError,
    Message,
    Step,
)

SYSTEM = "You are a test agent."


def conversation(
    question: str, turns: Sequence[tuple[str, str]] = (), reflection: str = ""
) -> list[Message]:
    """Messages for ``question`` after ``turns`` of (model reply, observation)."""
    steps = [
        Step(
            index=i, thought="", action=None, observation=obs, reflection=reflection, raw_output=raw
        )
        for i, (raw, obs) in enumerate(turns)
    ]
    return build_messages(SYSTEM, question, steps)


def drive(
    planner: LLMClient, question: str, tool: Callable[[str], str], limit: int = 6
) -> list[str]:
    """Run the planner like the agent would, feeding ``tool(reply)`` back as observations."""
    turns: list[tuple[str, str]] = []
    replies: list[str] = []
    for _ in range(limit):
        reply = planner.complete(conversation(question, turns))
        replies.append(reply)
        if "Final Answer:" in reply:
            break
        turns.append((reply, tool(reply)))
    return replies


# --------------------------------------------------------------------------- scripted / stubborn


def test_scripted_returns_responses_in_order_and_records_calls() -> None:
    llm = ScriptedLLM(["one", "two"])
    first = conversation("q1")
    assert llm.complete(first) == "one"
    assert llm.complete(conversation("q2")) == "two"
    assert llm.name == "scripted"
    assert llm.calls[0] == first
    assert len(llm.calls) == 2


def test_scripted_raises_exception_items_then_continues() -> None:
    llm = ScriptedLLM([LLMError("boom"), "ok"], name="script")
    with pytest.raises(LLMError, match="boom"):
        llm.complete([])
    assert llm.complete([]) == "ok"
    assert llm.name == "script"


def test_scripted_exhaustion_raises_llm_error() -> None:
    llm = ScriptedLLM([])
    with pytest.raises(LLMError, match="script exhausted"):
        llm.complete([])
    assert len(llm.calls) == 1


def test_stubborn_always_repeats_the_same_action() -> None:
    llm = StubbornLLM()
    replies = {llm.complete(conversation("anything")) for _ in range(3)}
    assert replies == {"Thought: I should check this again.\nAction: lookup[france population]"}
    assert llm.call_count == 3
    assert llm.name == "stubborn"
    assert StubbornLLM("calculator[1 + 1]").complete([]).endswith("Action: calculator[1 + 1]")


# --------------------------------------------------------------------------- offline planner


def fake_tools(reply: str) -> str:
    """Minimal stand-in for the tool registry used to drive the planner."""
    action = reply.split("Action: ", 1)[1]
    tool, arg = action[:-1].split("[", 1)
    if tool == "lookup":
        kb = {"france population": "68170000", "capital of japan": "Tokyo"}
        return kb.get(arg, f"No entry for '{arg}'")
    results = {"12 * 7": "84", "68170000 * 2": "136340000", "(3 + 4) * 5": "35"}
    return results.get(arg, "Error: unexpected expression")


def test_planner_calculates_then_answers() -> None:
    replies = drive(OfflinePlannerLLM(), "What is 12 * 7?", fake_tools)
    assert replies[0].endswith("Action: calculator[12 * 7]")
    assert replies[0].startswith("Thought: ")
    assert replies[1].endswith("\nFinal Answer: 84")
    assert len(replies) == 2


def test_planner_chains_lookup_into_calculator() -> None:
    replies = drive(OfflinePlannerLLM(), "What is the france population times 2?", fake_tools)
    assert [r.splitlines()[-1] for r in replies] == [
        "Action: lookup[france population]",
        "Action: calculator[68170000 * 2]",
        "Final Answer: 136340000",
    ]


def test_planner_keeps_parentheses() -> None:
    replies = drive(OfflinePlannerLLM(), "What is (3 + 4) * 5?", fake_tools)
    assert replies[0].endswith("Action: calculator[(3 + 4) * 5]")
    assert replies[-1].endswith("Final Answer: 35")


def test_planner_lookup_only_question() -> None:
    replies = drive(OfflinePlannerLLM(), "What is the capital of japan?", fake_tools)
    assert replies[0].endswith("Action: lookup[capital of japan]")
    assert replies[1].endswith("Final Answer: Tokyo")


def test_planner_answers_honestly_without_tools() -> None:
    reply = OfflinePlannerLLM().complete(conversation("Who wrote Hamlet?"))
    assert reply.endswith("Final Answer: I could not find an answer to: Who wrote Hamlet?")


def test_planner_handles_error_observation() -> None:
    planner = OfflinePlannerLLM(knowledge_base={"moon distance": "384400"})
    replies = drive(planner, "What is the moon distance plus 1?", fake_tools)
    assert replies[0].endswith("Action: lookup[moon distance]")
    # The lookup failed, so nothing numeric can be substituted and no calculation is possible.
    assert replies[1].endswith("I could not find an answer to: What is the moon distance plus 1?")


def test_planner_answers_after_critique_without_looping() -> None:
    question = "What is 12 * 7?"
    turns = [
        ("Thought: compute.\nAction: calculator[12 * 7]", "84"),
        ("Thought: done.\nFinal Answer: 84", "Your answer was rejected: explain more."),
    ]
    reply = OfflinePlannerLLM().complete(conversation(question, turns))
    assert reply.endswith("Final Answer: 84")


def test_planner_ignores_reflections_and_unparseable_turns() -> None:
    turns = [
        ("I am rambling without an action", "Could not parse your reply."),
        ("Thought: look.\nAction: lookup[capital of japan]", "Tokyo"),
    ]
    messages = conversation("What is the capital of japan?", turns, reflection="Good progress.")
    assert OfflinePlannerLLM().complete(messages).endswith("Final Answer: Tokyo")


def test_planner_looks_up_each_key_once_in_question_order() -> None:
    planner = OfflinePlannerLLM()
    question = "Add the germany population to the france population and the germany population."
    replies = drive(planner, question, lambda reply: "1")
    actions = [r.splitlines()[-1] for r in replies if "Action:" in r]
    assert actions[:2] == [
        "Action: lookup[germany population]",
        "Action: lookup[france population]",
    ]


def test_planner_prefers_longest_overlapping_key() -> None:
    planner = OfflinePlannerLLM(knowledge_base={"tower": "1", "eiffel tower": "330"})
    reply = planner.complete(conversation("How tall is the Eiffel Tower?"))
    assert reply.endswith("Action: lookup[eiffel tower]")


def test_planner_does_not_substitute_non_numeric_lookups() -> None:
    turns = [("Thought: look.\nAction: lookup[capital of japan]", "Tokyo")]
    reply = OfflinePlannerLLM().complete(conversation("capital of japan times 2", turns))
    assert reply.endswith("Final Answer: Tokyo")


def test_planner_survives_empty_and_odd_conversations() -> None:
    planner = OfflinePlannerLLM()
    assert planner.complete([]).endswith("I could not find an answer to: the question")
    odd = [Message("assistant", "Action: calculator[1 + 1]"), Message("user", "hello")]
    assert "Final Answer:" in planner.complete(odd)
    assert planner.name == "offline-planner"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What is 12 * 7?", "12 * 7"),
        ("What is 2^10?", "2**10"),
        ("What is 10 divided by 4 plus 1?", "10 / 4 + 1"),
        ("What is 6 multiplied by 7 minus 2?", "6 * 7 - 2"),
        ("Compute 3 times 4.", "3 * 4"),
        ("((1 + 2)", "(1 + 2)"),
        ("(1 + 2))", "(1 + 2)"),
        ("1 + 2, then 10 * 20 % 7", "10 * 20 % 7"),
        ("10 * 20 % 7, then 1 + 2", "10 * 20 % 7"),
        ("It costs 5 dollars", None),
        ("well-known", None),
        ("", None),
    ],
)
def test_find_expression(text: str, expected: str | None) -> None:
    assert find_expression(text) == expected


# --------------------------------------------------------------------------- resilient


class FlakyLLM:
    """Fails ``failures`` times, then answers ``reply`` forever."""

    def __init__(self, name: str, failures: int, reply: str = "ok") -> None:
        self.name = name
        self.failures = failures
        self.reply = reply
        self.calls = 0

    def complete(self, messages: Sequence[Message]) -> str:
        self.calls += 1
        if self.calls <= self.failures:
            raise RuntimeError(f"{self.name} down #{self.calls}")
        return self.reply


def test_resilient_primary_recovers_on_retry() -> None:
    sleeps: list[float] = []
    primary = FlakyLLM("primary", failures=1, reply="Final Answer: 84")
    llm = ResilientLLM(primary, sleep=sleeps.append)
    assert llm.complete([]) == "Final Answer: 84"
    assert primary.calls == 2
    assert sleeps == [0.5]
    assert llm.last_backend == "primary"
    assert llm.name == "primary"
    assert llm.notes == [
        "primary 'primary' recovered on attempt 2/3 after RuntimeError: primary down #1"
    ]


def test_resilient_uses_fallback_and_notes_it_once() -> None:
    sleeps: list[float] = []
    primary = FlakyLLM("main", failures=100)
    fallback = FlakyLLM("backup", failures=0, reply="from backup")
    llm = ResilientLLM(primary, [fallback], retries=2, sleep=sleeps.append)
    assert llm.complete([]) == "from backup"
    assert llm.complete([]) == "from backup"
    assert primary.calls == 6
    assert sleeps == [0.5, 1.0, 0.5, 1.0]
    assert llm.last_backend == "backup"
    assert llm.notes == [
        "primary 'main' failed after 3 attempts (RuntimeError: main down #3); "
        "answered by fallback 'backup'"
    ]


def test_resilient_all_backends_fail() -> None:
    llm = ResilientLLM(
        FlakyLLM("a", failures=100), [FlakyLLM("b", failures=100)], retries=0, sleep=lambda _: None
    )
    with pytest.raises(LLMUnavailableError) as info:
        llm.complete([])
    message = str(info.value)
    assert "primary 'a' failed after 1 attempt (RuntimeError: a down #1)" in message
    assert "fallback 'b' failed after 1 attempt (RuntimeError: b down #1)" in message
    assert llm.last_backend is None
    assert isinstance(info.value, LLMError)


def test_resilient_backoff_is_exponential_and_capped() -> None:
    sleeps: list[float] = []
    llm = ResilientLLM(
        FlakyLLM("x", failures=100),
        retries=4,
        backoff_seconds=1.0,
        max_backoff_seconds=5.0,
        sleep=sleeps.append,
    )
    with pytest.raises(LLMUnavailableError):
        llm.complete([])
    assert sleeps == [1.0, 2.0, 4.0, 5.0]


def test_resilient_default_sleep_is_real_but_unused_on_success() -> None:
    assert ResilientLLM(ScriptedLLM(["hi"])).complete([]) == "hi"


@pytest.mark.parametrize(
    "kwargs",
    [{"retries": -1}, {"backoff_seconds": -0.1}, {"max_backoff_seconds": -1.0}],
)
def test_resilient_rejects_invalid_config(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigError):
        ResilientLLM(ScriptedLLM([]), **kwargs)


# --------------------------------------------------------------------------- anthropic


class FakeMessages:
    def __init__(self, response: Any = None, error: Exception | None = None) -> None:
        self.response = response
        self.error = error
        self.requests: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.response


def text_response(*texts: str, stop_reason: str = "end_turn") -> SimpleNamespace:
    blocks = [SimpleNamespace(type="thinking", thinking="")]
    blocks += [SimpleNamespace(type="text", text=t) for t in texts]
    return SimpleNamespace(stop_reason=stop_reason, content=blocks)


def fake_client(messages: FakeMessages) -> SimpleNamespace:
    return SimpleNamespace(messages=messages)


@pytest.fixture
def no_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("REACT_AGENT_MODEL", raising=False)


@pytest.mark.usefixtures("no_model_env")
def test_anthropic_request_shape_and_reply() -> None:
    api = FakeMessages(text_response("Thought: x\n", "Final Answer: 84"))
    llm = AnthropicLLM(client=fake_client(api))
    messages = [
        Message("system", "rules"),
        Message("system", "more rules"),
        Message("user", "Question: q"),
        Message("assistant", "Action: a[b]"),
        Message("user", "Observation: o"),
        Message("user", "extra"),
    ]
    assert llm.complete(messages) == "Thought: x\nFinal Answer: 84"
    assert llm.name == f"anthropic:{DEFAULT_ANTHROPIC_MODEL}"
    assert api.requests == [
        {
            "model": DEFAULT_ANTHROPIC_MODEL,
            "max_tokens": 16000,
            "system": "rules\n\nmore rules",
            "messages": [
                {"role": "user", "content": "Question: q"},
                {"role": "assistant", "content": "Action: a[b]"},
                {"role": "user", "content": "Observation: o\n\nextra"},
            ],
        }
    ]


def test_anthropic_model_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    client = fake_client(FakeMessages())
    monkeypatch.setenv("REACT_AGENT_MODEL", "env-model")
    assert AnthropicLLM(client=client).model == "env-model"
    llm = AnthropicLLM("arg-model", max_tokens=10, client=client, name="custom")
    assert (llm.model, llm.max_tokens, llm.name) == ("arg-model", 10, "custom")


@pytest.mark.usefixtures("no_model_env")
def test_anthropic_refusal_raises() -> None:
    api = FakeMessages(text_response("partial", stop_reason="refusal"))
    with pytest.raises(LLMError, match="refused"):
        AnthropicLLM(client=fake_client(api)).complete([Message("user", "q")])


@pytest.mark.usefixtures("no_model_env")
def test_anthropic_empty_text_raises() -> None:
    api = FakeMessages(text_response("  "))
    with pytest.raises(LLMError, match="no text"):
        AnthropicLLM(client=fake_client(api)).complete([Message("user", "q")])


@pytest.mark.usefixtures("no_model_env")
def test_anthropic_wraps_sdk_errors() -> None:
    cause = RuntimeError("rate limited")
    api = FakeMessages(error=cause)
    with pytest.raises(LLMError, match="anthropic call failed: rate limited") as info:
        AnthropicLLM(client=fake_client(api)).complete([Message("user", "q")])
    assert info.value.__cause__ is cause
    assert not isinstance(info.value, LLMUnavailableError)


def test_anthropic_missing_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", None)
    with pytest.raises(LLMUnavailableError, match="anthropic SDK not installed"):
        AnthropicLLM()


class FakeAnthropic:
    """Stands in for ``anthropic.Anthropic``."""

    def __init__(self) -> None:
        self.messages = FakeMessages(text_response("hi"))


class BrokenAnthropic:
    def __init__(self) -> None:
        raise RuntimeError("bad base url")


@pytest.fixture
def no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)


@pytest.mark.usefixtures("no_credentials")
def test_anthropic_missing_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    with pytest.raises(LLMUnavailableError, match="ANTHROPIC_API_KEY"):
        AnthropicLLM()


@pytest.mark.usefixtures("no_credentials", "no_model_env")
@pytest.mark.parametrize("variable", ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"])
def test_anthropic_builds_default_client(monkeypatch: pytest.MonkeyPatch, variable: str) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=FakeAnthropic))
    monkeypatch.setenv(variable, "secret")
    assert AnthropicLLM().complete([Message("user", "q")]) == "hi"


@pytest.mark.usefixtures("no_credentials")
def test_anthropic_client_construction_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "anthropic", SimpleNamespace(Anthropic=BrokenAnthropic))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    with pytest.raises(LLMUnavailableError, match="bad base url"):
        AnthropicLLM()
