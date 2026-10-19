"""Model backends: test doubles, an offline planner, a resilience wrapper and Anthropic.

Every class here implements the ``LLMClient`` protocol from ``types``: a ``name`` and a
``complete(messages) -> str`` method that returns ReAct-formatted text or raises
``LLMError``.
"""

from __future__ import annotations

import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from react_agent.prompts import extract_observations, extract_question
from react_agent.types import (
    ConfigError,
    LLMClient,
    LLMError,
    LLMUnavailableError,
    Message,
    normalize_input,
)

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
"""Model used by ``AnthropicLLM`` unless overridden by argument or ``REACT_AGENT_MODEL``."""

MODEL_ENV_VAR = "REACT_AGENT_MODEL"

# --------------------------------------------------------------------------- test doubles


class ScriptedLLM:
    """Replays canned responses in order; an ``Exception`` item is raised instead."""

    def __init__(self, responses: Sequence[str | Exception], name: str = "scripted") -> None:
        self.name = name
        self._responses = list(responses)
        self._position = 0
        self.calls: list[list[Message]] = []

    def complete(self, messages: Sequence[Message]) -> str:
        self.calls.append(list(messages))
        if self._position >= len(self._responses):
            raise LLMError("script exhausted")
        response = self._responses[self._position]
        self._position += 1
        if isinstance(response, Exception):
            raise response
        return response


class StubbornLLM:
    """Demo model that repeats the same action forever (exercises the loop guard)."""

    def __init__(self, action: str = "lookup[france population]", name: str = "stubborn") -> None:
        self.name = name
        self.action = action
        self.call_count = 0

    def complete(self, messages: Sequence[Message]) -> str:
        self.call_count += 1
        return f"Thought: I should check this again.\nAction: {self.action}"


# --------------------------------------------------------------------------- offline planner

_FALLBACK_KB: dict[str, str] = {
    "france population": "68170000",
    "germany population": "84480000",
    "speed of light": "299792458",
    "earth radius": "6371",
    "boiling point of water": "100",
    "eiffel tower height": "330",
    "capital of france": "Paris",
    "capital of japan": "Tokyo",
}
"""Mirror of the default lookup knowledge base (kept local to avoid importing tools)."""

_ACTION_LINE = re.compile(r"^\s*Action\s*:\s*([A-Za-z_][\w-]*)\s*\[(.*)\]\s*$", re.I | re.M)
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")
_WORD_OPERATORS = (
    (re.compile(r"\bdivided\s+by\b", re.I), "/"),
    (re.compile(r"\bmultiplied\s+by\b", re.I), "*"),
    (re.compile(r"\btimes\b", re.I), "*"),
    (re.compile(r"\bplus\b", re.I), "+"),
    (re.compile(r"\bminus\b", re.I), "-"),
)
_EXPRESSION_RUN = re.compile(r"[\d.\s()+\-*/%^]+")
_BINARY_OPERATION = re.compile(r"\d\.?\)*\s*[-+*/%^]+\s*[-+(]*\s*\.?\d")
_ERROR_PREFIXES = ("error", "unknown tool", "no entry", "tool '", "invalid", "timeout", "timed out")


def _looks_like_error(observation: str) -> bool:
    return observation.strip().lower().startswith(_ERROR_PREFIXES)


def _numeric_value(observation: str) -> str | None:
    """Return the single number in a successful observation, if there is exactly one."""
    text = observation.strip()
    if _looks_like_error(text):
        return None
    numbers = _NUMBER.findall(text)
    return numbers[0] if len(numbers) == 1 else None


def _strip_unbalanced_parens(expression: str) -> str:
    while expression.startswith("(") and expression.count("(") > expression.count(")"):
        expression = expression[1:].strip()
    while expression.endswith(")") and expression.count(")") > expression.count("("):
        expression = expression[:-1].strip()
    return expression


def find_expression(text: str) -> str | None:
    """Extract the longest arithmetic expression from free text, or ``None``.

    Word operators (``plus``, ``times``, ``divided by`` ...) are mapped to symbols and
    ``^`` to ``**``. A candidate must contain an operator between two operands.
    """
    for pattern, symbol in _WORD_OPERATORS:
        text = pattern.sub(f" {symbol} ", text)
    best: str | None = None
    for match in _EXPRESSION_RUN.finditer(text):
        candidate = _strip_unbalanced_parens(match.group().strip().rstrip("."))
        if not _BINARY_OPERATION.search(candidate):
            continue
        candidate = " ".join(candidate.split())
        if best is None or len(candidate) > len(best):
            best = candidate
    return None if best is None else best.replace("^", "**")


class OfflinePlannerLLM:
    """Deterministic rule-based "model" for demos, the CLI default and integration tests.

    Plan: look up every knowledge-base key mentioned in the question, then evaluate any
    arithmetic in the question once (with looked-up numbers substituted), then answer
    with the latest tool observation. It is deliberately simple, never raises, and never
    repeats an action it already took.
    """

    def __init__(
        self, knowledge_base: Mapping[str, str] | None = None, name: str = "offline-planner"
    ) -> None:
        self.name = name
        kb = _FALLBACK_KB if knowledge_base is None else knowledge_base
        self._keys = sorted((key.lower() for key in kb), key=len, reverse=True)

    def complete(self, messages: Sequence[Message]) -> str:
        question = extract_question(messages)
        history = self._tool_history(messages)
        taken = {(tool, normalize_input(arg)) for tool, arg, _ in history}
        lookups = {
            normalize_input(arg): obs for tool, arg, obs in history if tool == "lookup" and obs
        }
        mentions = self._mentioned_keys(question)

        for key in dict.fromkeys(key for _, _, key in mentions):
            if ("lookup", key) not in taken:
                return f"Thought: I need the {key} to answer the question.\nAction: lookup[{key}]"

        if not any(tool == "calculator" for tool, _ in taken):
            expression = find_expression(self._substitute(question, mentions, lookups))
            if expression is not None:
                return (
                    f"Thought: I can compute {expression} with the calculator.\n"
                    f"Action: calculator[{expression}]"
                )

        answered = [obs for _, _, obs in history if obs is not None]
        if answered and not _looks_like_error(answered[-1]) and answered[-1].strip():
            return (
                "Thought: The latest observation answers the question.\n"
                f"Final Answer: {answered[-1].strip()}"
            )
        subject = question.strip() or "the question"
        return (
            "Thought: My tools cannot answer this question.\n"
            f"Final Answer: I could not find an answer to: {subject}"
        )

    def _mentioned_keys(self, question: str) -> list[tuple[int, int, str]]:
        """Non-overlapping KB keys in the question as ``(start, end, key)``, by position."""
        lowered = question.lower()
        spans: list[tuple[int, int, str]] = []
        for key in self._keys:  # longest first, so longer keys win overlaps
            for match in re.finditer(re.escape(key), lowered):
                start, end = match.span()
                if not any(start < e and s < end for s, e, _ in spans):
                    spans.append((start, end, key))
        return sorted(spans)

    @staticmethod
    def _substitute(
        question: str, mentions: list[tuple[int, int, str]], lookups: Mapping[str, str]
    ) -> str:
        """Replace mentioned keys that resolved to a number with that number."""
        text = question
        for start, end, key in reversed(mentions):
            value = _numeric_value(lookups.get(key, ""))
            if value is not None:
                text = f"{text[:start]}{value}{text[end:]}"
        return text

    @staticmethod
    def _tool_history(messages: Sequence[Message]) -> list[tuple[str, str, str | None]]:
        """``(tool, input, observation)`` for each earlier assistant action.

        Assistant turns without an action (final answers, unparseable text) are skipped,
        so critiques and format hints sent back as observations are never mistaken for
        tool output.
        """
        history: list[tuple[str, str, str | None]] = []
        pending: tuple[str, str] | None = None
        for message in messages:
            if message.role == "assistant":
                match = _ACTION_LINE.search(message.content)
                pending = None if match is None else (match.group(1).lower(), match.group(2))
                if pending is not None:
                    history.append((*pending, None))
            elif pending is not None and (observation := extract_observations([message])):
                history[-1] = (*pending, observation[0])
                pending = None
        return history


# --------------------------------------------------------------------------- resilience


def _describe(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


class ResilientLLM:
    """Retry each backend with exponential backoff, then fall back to the next one.

    Implements ``LLMClient`` itself, so the agent never knows retries happened; what did
    happen is recorded in ``notes`` and ``last_backend``. Raises ``LLMUnavailableError``
    when every backend fails.
    """

    def __init__(
        self,
        primary: LLMClient,
        fallbacks: Sequence[LLMClient] = (),
        *,
        retries: int = 2,
        backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 8.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if retries < 0:
            raise ConfigError("retries must be >= 0")
        if backoff_seconds < 0:
            raise ConfigError("backoff_seconds must be >= 0")
        if max_backoff_seconds < 0:
            raise ConfigError("max_backoff_seconds must be >= 0")
        self.name = primary.name
        self.retries = retries
        self.backoff_seconds = backoff_seconds
        self.max_backoff_seconds = max_backoff_seconds
        self.notes: list[str] = []
        self.last_backend: str | None = None
        self._backends: list[LLMClient] = [primary, *fallbacks]
        self._sleep = sleep
        self._fallbacks_noted: set[int] = set()

    def backoff(self, attempt: int) -> float:
        """Delay after failed attempt number ``attempt`` (0-based) of one backend."""
        return float(min(self.backoff_seconds * 2**attempt, self.max_backoff_seconds))

    def complete(self, messages: Sequence[Message]) -> str:
        failures: list[str] = []
        for position, backend in enumerate(self._backends):
            role = "primary" if position == 0 else "fallback"
            attempts = self.retries + 1
            last_error: Exception | None = None
            for attempt in range(attempts):
                if attempt:
                    self._sleep(self.backoff(attempt - 1))
                try:
                    reply = backend.complete(messages)
                except Exception as exc:  # SDKs raise all sorts; every failure is retryable
                    last_error = exc
                    continue
                if last_error is not None:
                    self.notes.append(
                        f"{role} '{backend.name}' recovered on attempt {attempt + 1}/{attempts} "
                        f"after {_describe(last_error)}"
                    )
                if position and position not in self._fallbacks_noted:
                    self._fallbacks_noted.add(position)
                    self.notes.append(
                        f"{'; '.join(failures)}; answered by fallback '{backend.name}'"
                    )
                self.last_backend = backend.name
                return reply
            assert last_error is not None
            failures.append(
                f"{role} '{backend.name}' failed after {attempts} attempt"
                f"{'s' if attempts > 1 else ''} ({_describe(last_error)})"
            )
        self.last_backend = None
        raise LLMUnavailableError("all models failed: " + "; ".join(failures))


# --------------------------------------------------------------------------- anthropic


class AnthropicLLM:
    """Hosted Claude backend via the optional ``anthropic`` SDK (imported lazily).

    Construction raises ``LLMUnavailableError`` when the SDK or credentials are missing,
    so callers can warn and fall back. Call failures and refusals raise ``LLMError``.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        max_tokens: int = 16000,
        client: Any | None = None,
        name: str | None = None,
    ) -> None:
        self.model = model or os.environ.get(MODEL_ENV_VAR) or DEFAULT_ANTHROPIC_MODEL
        self.max_tokens = max_tokens
        self.name = name or f"anthropic:{self.model}"
        self._client = client if client is not None else self._default_client()

    @staticmethod
    def _default_client() -> Any:
        try:
            import anthropic
        except ImportError as exc:
            raise LLMUnavailableError(
                "anthropic SDK not installed; pip install 'react-planning-agent[anthropic]'"
            ) from exc
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise LLMUnavailableError(
                "no Anthropic credentials: set ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN)"
            )
        try:
            return anthropic.Anthropic()
        except Exception as exc:
            raise LLMUnavailableError(f"could not create Anthropic client: {exc}") from exc

    @staticmethod
    def to_request(messages: Sequence[Message]) -> tuple[str, list[dict[str, str]]]:
        """Split a conversation into the API's ``system`` text and alternating messages."""
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        chat: list[dict[str, str]] = []
        for message in messages:
            if message.role == "system":
                continue
            if chat and chat[-1]["role"] == message.role:
                chat[-1]["content"] += "\n\n" + message.content
            else:
                chat.append({"role": message.role, "content": message.content})
        return system, chat

    def complete(self, messages: Sequence[Message]) -> str:
        system, chat = self.to_request(messages)
        try:
            response = self._client.messages.create(
                model=self.model, max_tokens=self.max_tokens, system=system, messages=chat
            )
        except Exception as exc:
            raise LLMError(f"anthropic call failed: {exc}") from exc
        if getattr(response, "stop_reason", None) == "refusal":
            raise LLMError("model refused the request")
        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text.strip():
            raise LLMError("model returned no text")
        return text
