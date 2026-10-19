"""Two-level self-critique (design D4).

``Reflector`` annotates every step with a cheap heuristic reflection. Critics gate the
final answer: ``HeuristicCritic`` rejects empty or placeholder answers, ``LLMCritic`` asks
a model for ``APPROVE`` / ``REJECT: <reason>`` and fails open when the model is unavailable.
"""

from __future__ import annotations

import re
import string
from collections.abc import Sequence

from react_agent.types import Action, ConfigError, Critic, Critique, LLMClient, Message, Step

_SNIPPET_CHARS = 120


def _snippet(text: str, limit: int = _SNIPPET_CHARS) -> str:
    """First line of ``text``, collapsed and truncated to ``limit`` characters."""
    line = " ".join(text.strip().splitlines()[0].split()) if text.strip() else ""
    return line if len(line) <= limit else line[: limit - 3].rstrip() + "..."


# --------------------------------------------------------------------------- per step


class Reflector:
    """Heuristic per-step reflection: what the step achieved and how to proceed."""

    def reflect(
        self,
        action: Action | None,
        observation: str,
        *,
        is_error: bool,
        repeated: bool = False,
        remaining_iterations: int | None = None,
    ) -> str:
        """Return a one- or two-sentence reflection on a completed step.

        ``action`` is ``None`` when the model output could not be parsed. A ``finish``
        action reaching reflection means the proposed answer was not accepted.
        """
        if action is None:
            text = (
                "The previous output could not be parsed. Follow the required format: "
                "'Thought: ...' then 'Action: tool[input]' or 'Final Answer: ...'."
            )
        elif repeated:
            text = (
                f"{action} was a repeat and gained no new information. "
                "Plan a different action or finish with your best answer."
            )
        elif action.is_finish:
            text = "The proposed answer was not accepted. Address the critique, then finish again."
        elif is_error:
            reason = _snippet(observation) or "no details"
            text = (
                f"The action {action} failed ({reason}). "
                "Try a different tool or a different input instead of retrying it."
            )
        else:
            text = (
                f"Progress: {action.tool} returned new information. "
                "Decide whether this is enough to answer."
            )
        if remaining_iterations is not None and remaining_iterations <= 1:
            text += " Few steps remain: finish with your best answer."
        return text


# --------------------------------------------------------------------------- final answer

_PLACEHOLDERS = frozenset(
    {"", "unknown", "n/a", "na", "none", "todo", "tbd", "i don't know", "i do not know"}
)
_EDGE_CHARS = string.whitespace + string.punctuation


def _normalize_answer(answer: str) -> str:
    """Lower-case, unify apostrophes, collapse whitespace, strip edge punctuation."""
    text = " ".join(answer.replace("\u2019", "'").lower().split())
    return text.strip(_EDGE_CHARS)


def _made_progress(steps: Sequence[Step]) -> bool:
    """True if any step ran a (non-finish) action successfully."""
    return any(
        step.action is not None and not step.action.is_finish and not step.is_error
        for step in steps
    )


class HeuristicCritic:
    """Rejects empty answers, and placeholder answers when the run gathered information."""

    def critique(self, question: str, answer: str, steps: Sequence[Step]) -> Critique:
        if not answer.strip():
            return Critique(
                approved=False,
                feedback="The final answer is empty. Give a concrete answer to the question "
                "based on the observations so far.",
            )
        if _normalize_answer(answer) in _PLACEHOLDERS and _made_progress(steps):
            return Critique(
                approved=False,
                feedback=f"The final answer {answer.strip()!r} is a placeholder, but earlier "
                "steps returned information. Use those observations to give a concrete answer.",
            )
        return Critique(approved=True)


LLM_CRITIC_SYSTEM = (
    "You review a proposed final answer produced by a tool-using agent. "
    "Check that it answers the question and is supported by the observations. "
    "Reply with exactly one line: 'APPROVE' or 'REJECT: <short reason>'."
)

_VERDICT = re.compile(r"^\W*(APPROVE|REJECT)\w*\W*(.*)$", re.IGNORECASE | re.DOTALL)
_SUMMARY_STEPS = 5
_SUMMARY_CHARS = 200


def summarize_steps(steps: Sequence[Step], limit: int = _SUMMARY_STEPS) -> str:
    """Compact, one-line-per-step summary of the last ``limit`` steps."""
    if not steps:
        return "(no steps taken)"
    lines = []
    for step in steps[-limit:]:
        action = "(unparseable output)" if step.action is None else str(step.action)
        marker = " [error]" if step.is_error else ""
        lines.append(f"- {action}{marker}: {_snippet(step.observation, _SUMMARY_CHARS)}")
    return "\n".join(lines)


class LLMCritic:
    """Asks a model to approve or reject the answer; approves if the model fails."""

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def critique(self, question: str, answer: str, steps: Sequence[Step]) -> Critique:
        prompt = (
            f"Question: {question}\n"
            f"Proposed answer: {answer}\n"
            f"Observations:\n{summarize_steps(steps)}\n\n"
            "Reply 'APPROVE' or 'REJECT: <reason>'."
        )
        messages = [Message("system", LLM_CRITIC_SYSTEM), Message("user", prompt)]
        try:
            reply = self.llm.complete(messages)
        except Exception as exc:  # fail open: a broken critic must not lose the answer
            detail = str(exc) or type(exc).__name__
            return Critique(
                approved=True, feedback=f"critique unavailable: {detail}", unavailable=True
            )
        return parse_verdict(reply)


def parse_verdict(reply: str) -> Critique:
    """Turn an ``APPROVE`` / ``REJECT: <reason>`` reply into a ``Critique``.

    Unrecognised replies are approved with a note, in keeping with failing open.
    """
    match = _VERDICT.match(reply.strip())
    if match is None:
        return Critique(
            approved=True,
            feedback=f"ambiguous critic verdict treated as approval: {_snippet(reply)!r}",
        )
    verdict, reason = match.group(1).upper(), match.group(2).strip()
    if verdict == "APPROVE":
        return Critique(approved=True, feedback=reason)
    return Critique(approved=False, feedback=reason or "The critic rejected the answer.")


class ChainCritic:
    """Runs critics in order and returns the first rejection."""

    def __init__(self, *critics: Critic) -> None:
        if not critics:
            raise ConfigError("ChainCritic needs at least one critic")
        self.critics = critics

    def critique(self, question: str, answer: str, steps: Sequence[Step]) -> Critique:
        feedback: list[str] = []
        unavailable = False
        for critic in self.critics:
            result = critic.critique(question, answer, steps)
            if not result.approved:
                return result
            unavailable = unavailable or result.unavailable
            if result.feedback:
                feedback.append(result.feedback)
        return Critique(approved=True, feedback="; ".join(feedback), unavailable=unavailable)


def default_critic(llm: LLMClient | None = None) -> Critic:
    """Heuristic critic, chained with a model critic when ``llm`` is given."""
    if llm is None:
        return HeuristicCritic()
    return ChainCritic(HeuristicCritic(), LLMCritic(llm))
