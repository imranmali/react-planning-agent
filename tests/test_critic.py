"""Tests for step reflection and final-answer critics (spec: self-critique)."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from react_agent.critic import (
    LLM_CRITIC_SYSTEM,
    ChainCritic,
    HeuristicCritic,
    LLMCritic,
    Reflector,
    default_critic,
    parse_verdict,
    summarize_steps,
)
from react_agent.types import Action, ConfigError, Critique, LLMError, Message, Step


class FakeLLM:
    """Minimal ``LLMClient``: returns a fixed reply or raises, and records calls."""

    name = "fake"

    def __init__(self, reply: str = "APPROVE", error: Exception | None = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message]) -> str:
        self.calls.append(list(messages))
        if self.error is not None:
            raise self.error
        return self.reply


class FixedCritic:
    def __init__(self, critique: Critique) -> None:
        self.result = critique
        self.calls = 0

    def critique(self, question: str, answer: str, steps: Sequence[Step]) -> Critique:
        self.calls += 1
        return self.result


def step(
    index: int = 1,
    action: Action | None = None,
    observation: str = "Paris",
    *,
    is_error: bool = False,
) -> Step:
    return Step(index, "thinking", action, observation, "", is_error=is_error)


SUCCESSFUL_LOOKUP = step(action=Action("lookup", "france"), observation="Capital: Paris")


# --------------------------------------------------------------------------- Reflector


def test_reflection_on_failed_tool_recommends_different_approach() -> None:
    text = Reflector().reflect(
        Action("calculator", "1/0"), "Error: division by zero\nTraceback...", is_error=True
    )
    assert "calculator[1/0] failed" in text
    assert "division by zero" in text
    assert "Traceback" not in text
    assert "different tool or a different input" in text


def test_reflection_on_failed_tool_without_details() -> None:
    text = Reflector().reflect(Action("lookup", "x"), "   ", is_error=True)
    assert "(no details)" in text


def test_reflection_truncates_long_error_reason() -> None:
    text = Reflector().reflect(Action("lookup", "x"), "e" * 500, is_error=True)
    assert "e" * 500 not in text
    assert "...)" in text


def test_reflection_on_progress() -> None:
    text = Reflector().reflect(Action("lookup", "france"), "Capital: Paris", is_error=False)
    assert text.startswith("Progress: lookup returned new information.")
    assert "Decide whether this is enough to answer." in text


def test_reflection_on_parse_failure() -> None:
    text = Reflector().reflect(None, "Could not find an action", is_error=True)
    assert "could not be parsed" in text
    assert "format" in text


def test_reflection_on_repeat() -> None:
    text = Reflector().reflect(
        Action("lookup", "france"), "WARNING: ...", is_error=False, repeated=True
    )
    assert "repeat" in text
    assert "no new information" in text
    assert "different action or finish" in text


def test_reflection_on_rejected_finish() -> None:
    text = Reflector().reflect(Action("finish", ""), "The final answer is empty.", is_error=False)
    assert "not accepted" in text


@pytest.mark.parametrize(
    ("remaining", "warned"), [(None, False), (5, False), (2, False), (1, True), (0, True)]
)
def test_reflection_warns_when_few_steps_remain(remaining: int | None, warned: bool) -> None:
    text = Reflector().reflect(
        Action("lookup", "france"), "Paris", is_error=False, remaining_iterations=remaining
    )
    assert ("Few steps remain: finish with your best answer." in text) is warned


# --------------------------------------------------------------------------- HeuristicCritic


@pytest.mark.parametrize("answer", ["", "   ", "\n\t"])
def test_empty_answer_rejected(answer: str) -> None:
    result = HeuristicCritic().critique("What is the capital of France?", answer, [])
    assert not result.approved
    assert "empty" in result.feedback
    assert not result.unavailable


def test_good_answer_accepted() -> None:
    result = HeuristicCritic().critique("Capital of France?", "Paris", [SUCCESSFUL_LOOKUP])
    assert result == Critique(approved=True)


@pytest.mark.parametrize(
    "answer",
    [
        "unknown",
        "N/A",
        "na",
        "None.",
        "TODO",
        "tbd",
        "I don't know.",
        "I don\u2019t know",
        "...",
        "  I do not   know!  ",
    ],
)
def test_placeholder_rejected_after_progress(answer: str) -> None:
    result = HeuristicCritic().critique("Capital of France?", answer, [SUCCESSFUL_LOOKUP])
    assert not result.approved
    assert "placeholder" in result.feedback


@pytest.mark.parametrize(
    "steps",
    [
        [],
        [step(action=Action("lookup", "x"), observation="Error", is_error=True)],
        [step(action=None, observation="parse error")],
        [step(action=Action("finish", ""), observation="The final answer is empty.")],
    ],
)
def test_placeholder_accepted_without_progress(steps: list[Step]) -> None:
    assert HeuristicCritic().critique("Q?", "I don't know", steps).approved


@pytest.mark.parametrize("answer", ["None of them", "unknowns: 3", "nan"])
def test_answers_containing_placeholder_words_accepted(answer: str) -> None:
    assert HeuristicCritic().critique("Q?", answer, [SUCCESSFUL_LOOKUP]).approved


# --------------------------------------------------------------------------- LLMCritic


def test_llm_critic_sends_question_answer_and_observations() -> None:
    llm = FakeLLM("APPROVE")
    steps = [step(i, Action("lookup", f"k{i}"), f"obs {i}") for i in range(1, 8)]
    steps.append(step(8, None, "unparseable", is_error=True))
    result = LLMCritic(llm).critique("Capital of France?", "Paris", steps)
    assert result == Critique(approved=True)
    (messages,) = llm.calls
    assert messages[0] == Message("system", LLM_CRITIC_SYSTEM)
    user = messages[1]
    assert user.role == "user"
    assert "Question: Capital of France?" in user.content
    assert "Proposed answer: Paris" in user.content
    assert "lookup[k7]: obs 7" in user.content
    assert "(unparseable output) [error]: unparseable" in user.content
    assert "lookup[k3]" not in user.content  # only the last five steps


def test_llm_critic_rejection_carries_reason() -> None:
    result = LLMCritic(FakeLLM("REJECT: the answer ignores the lookup")).critique("Q", "A", [])
    assert result == Critique(approved=False, feedback="the answer ignores the lookup")


@pytest.mark.parametrize("error", [LLMError("rate limited"), RuntimeError("boom"), ValueError()])
def test_critic_failure_fails_open(error: Exception) -> None:
    result = LLMCritic(FakeLLM(error=error)).critique("Q", "A", [SUCCESSFUL_LOOKUP])
    assert result.approved
    assert result.unavailable
    assert result.feedback.startswith("critique unavailable: ")
    assert (str(error) or type(error).__name__) in result.feedback


@pytest.mark.parametrize(
    ("reply", "approved", "feedback"),
    [
        ("APPROVE", True, ""),
        ("approve.", True, ""),
        ("  **Approved** - well supported", True, "well supported"),
        ("REJECT: missing units", False, "missing units"),
        ("reject - wrong country", False, "wrong country"),
        ("Rejected:\nthe answer is unsupported", False, "the answer is unsupported"),
        ("REJECT", False, "The critic rejected the answer."),
    ],
)
def test_parse_verdict(reply: str, approved: bool, feedback: str) -> None:
    assert parse_verdict(reply) == Critique(approved=approved, feedback=feedback)


@pytest.mark.parametrize("reply", ["", "Looks fine to me", "I would reject this"])
def test_parse_verdict_ambiguous_is_approved_with_note(reply: str) -> None:
    result = parse_verdict(reply)
    assert result.approved
    assert not result.unavailable
    assert result.feedback.startswith("ambiguous critic verdict")


def test_summarize_steps_empty_and_truncated() -> None:
    assert summarize_steps([]) == "(no steps taken)"
    summary = summarize_steps([step(action=Action("lookup", "x"), observation="z" * 1000)])
    assert len(summary) < 300
    assert summary.endswith("...")


# --------------------------------------------------------------------------- ChainCritic


def test_chain_returns_first_rejection_and_stops() -> None:
    first = FixedCritic(Critique(approved=True, feedback="ok"))
    second = FixedCritic(Critique(approved=False, feedback="no"))
    third = FixedCritic(Critique(approved=False, feedback="never asked"))
    result = ChainCritic(first, second, third).critique("Q", "A", [])
    assert result == Critique(approved=False, feedback="no")
    assert third.calls == 0


def test_chain_approval_merges_feedback_and_unavailability() -> None:
    chain = ChainCritic(
        FixedCritic(Critique(approved=True)),
        FixedCritic(Critique(approved=True, feedback="critique unavailable: x", unavailable=True)),
        FixedCritic(Critique(approved=True, feedback="looks right")),
    )
    result = chain.critique("Q", "A", [])
    assert result == Critique(
        approved=True, feedback="critique unavailable: x; looks right", unavailable=True
    )


def test_chain_requires_a_critic() -> None:
    with pytest.raises(ConfigError):
        ChainCritic()


# --------------------------------------------------------------------------- default_critic


def test_default_critic_without_llm_is_heuristic() -> None:
    assert isinstance(default_critic(), HeuristicCritic)


def test_default_critic_with_llm_runs_heuristic_first() -> None:
    llm = FakeLLM("REJECT: wrong")
    critic = default_critic(llm)
    assert isinstance(critic, ChainCritic)
    empty = critic.critique("Q", "", [])
    assert not empty.approved
    assert "empty" in empty.feedback
    assert llm.calls == []  # heuristic rejection short-circuits the model
    assert critic.critique("Q", "Paris", []) == Critique(approved=False, feedback="wrong")


def test_default_critic_fails_open_when_model_breaks() -> None:
    critic = default_critic(FakeLLM(error=LLMError("down")))
    result = critic.critique("Q", "Paris", [SUCCESSFUL_LOOKUP])
    assert result.approved
    assert result.unavailable
    assert "critique unavailable: down" in result.feedback
