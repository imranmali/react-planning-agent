"""Tests for the model backends in ``react_agent.llm`` (no network, no SDK needed)."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import pytest

from react_agent.llm import (
    OfflinePlannerLLM,
    ScriptedLLM,
    StubbornLLM,
    find_expression,
)
from react_agent.prompts import build_messages
from react_agent.types import (
    LLMClient,
    LLMError,
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
