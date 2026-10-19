"""Shared data types, enums, errors and protocols.

This module is the contract between the agent's components (parser, tools, guards,
critic, LLM backends and the loop itself). It has no dependencies on other modules.
"""

from __future__ import annotations

import enum
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

FINISH_ACTION = "finish"
"""Reserved action name: its input is the proposed final answer."""

_WHITESPACE = re.compile(r"\s+")


class Status(str, enum.Enum):
    """How trustworthy the run's answer is."""

    SUCCESS = "success"
    DEGRADED = "degraded"
    FAILED = "failed"


class TerminationReason(str, enum.Enum):
    """Why the agent loop stopped. Every run ends with exactly one of these."""

    FINAL_ANSWER = "final_answer"
    MAX_ITERATIONS = "max_iterations"
    TIME_BUDGET = "time_budget"
    LOOP_DETECTED = "loop_detected"
    ERROR_LIMIT = "error_limit"
    LLM_UNAVAILABLE = "llm_unavailable"


class LoopVerdict(str, enum.Enum):
    """Classification of an action by the loop detector."""

    NEW = "new"
    REPEAT = "repeat"
    LOOP = "loop"


# --------------------------------------------------------------------------- errors


class AgentError(Exception):
    """Base class for all errors raised by this package."""


class ConfigError(AgentError, ValueError):
    """Invalid agent configuration. The only error a caller should expect to see."""


class LLMError(AgentError):
    """A single model call failed (retryable by ResilientLLM)."""


class LLMUnavailableError(LLMError):
    """No configured model could produce a response."""


class ToolError(AgentError):
    """A tool rejected its input or failed in an expected way."""


# --------------------------------------------------------------------------- messages


@dataclass(frozen=True)
class Message:
    """One chat message sent to a model.

    Conversation layout produced by ``prompts.build_messages``:
    ``[system, user("Question: ..."), assistant(raw model output),
    user("Observation: ...\\nReflection: ..."), assistant(...), ...]``
    """

    role: Literal["system", "user", "assistant"]
    content: str


class LLMClient(Protocol):
    """Anything that can turn a conversation into the next model reply."""

    name: str

    def complete(self, messages: Sequence[Message]) -> str:
        """Return the model's reply text. Raise ``LLMError`` on failure."""
        ...


# --------------------------------------------------------------------------- actions


def normalize_input(text: str) -> str:
    """Normalise a tool input for loop detection: lower-case, collapsed whitespace."""
    return _WHITESPACE.sub(" ", text).strip().lower()


@dataclass(frozen=True)
class Action:
    """A tool invocation requested by the model."""

    tool: str
    input: str

    @property
    def is_finish(self) -> bool:
        return self.tool.strip().lower() == FINISH_ACTION

    @property
    def signature(self) -> tuple[str, str]:
        """Identity used for loop detection."""
        return (self.tool.strip().lower(), normalize_input(self.input))

    def __str__(self) -> str:
        return f"{self.tool}[{self.input}]"


@dataclass(frozen=True)
class ParsedOutput:
    """Successfully parsed model output: a thought plus exactly one action."""

    thought: str
    action: Action


@dataclass(frozen=True)
class ParseFailure:
    """Model output that could not be parsed. ``message`` is shown to the model."""

    message: str
    raw: str


@dataclass(frozen=True)
class ToolResult:
    """Outcome of executing a tool. Tools never raise past the registry."""

    output: str
    is_error: bool = False


@dataclass(frozen=True)
class Critique:
    """A critic's verdict on a proposed final answer."""

    approved: bool
    feedback: str = ""
    unavailable: bool = False
    """True when the critic failed and approved by default (fail-open)."""


class Critic(Protocol):
    """Reviews a proposed final answer before the agent accepts it."""

    def critique(self, question: str, answer: str, steps: Sequence[Step]) -> Critique: ...


# --------------------------------------------------------------------------- trace


@dataclass
class Step:
    """One observe/think/act/reflect iteration."""

    index: int
    thought: str
    action: Action | None
    observation: str
    reflection: str
    is_error: bool = False
    raw_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "thought": self.thought,
            "action": None
            if self.action is None
            else {"tool": self.action.tool, "input": self.action.input},
            "observation": self.observation,
            "reflection": self.reflection,
            "is_error": self.is_error,
        }


@dataclass
class AgentResult:
    """What every run returns, whatever happened."""

    answer: str
    status: Status
    termination_reason: TerminationReason
    steps: list[Step] = field(default_factory=list)
    iterations: int = 0
    elapsed_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status is Status.SUCCESS

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "status": self.status.value,
            "termination_reason": self.termination_reason.value,
            "iterations": self.iterations,
            "elapsed_seconds": round(self.elapsed_seconds, 6),
            "notes": list(self.notes),
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class AgentConfig:
    """Execution limits. Validated on construction; invalid values raise ``ConfigError``."""

    max_iterations: int = 10
    time_budget_seconds: float | None = 60.0
    max_repeats: int = 3
    max_consecutive_errors: int = 3
    max_revisions: int = 2
    tool_timeout_seconds: float = 10.0
    max_observation_chars: int = 2000

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise ConfigError("max_iterations must be >= 1")
        if self.time_budget_seconds is not None and self.time_budget_seconds <= 0:
            raise ConfigError("time_budget_seconds must be > 0 or None")
        if self.max_repeats < 2:
            raise ConfigError("max_repeats must be >= 2")
        if self.max_consecutive_errors < 1:
            raise ConfigError("max_consecutive_errors must be >= 1")
        if self.max_revisions < 0:
            raise ConfigError("max_revisions must be >= 0")
        if self.tool_timeout_seconds <= 0:
            raise ConfigError("tool_timeout_seconds must be > 0")
        if self.max_observation_chars < 1:
            raise ConfigError("max_observation_chars must be >= 1")
