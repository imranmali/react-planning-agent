"""The ReAct planning agent: a bounded observe -> think -> act -> reflect loop.

Every exit path of :meth:`ReActAgent.run` is explicit: the loop only ends through a
:class:`~react_agent.types.TerminationReason`, and runtime failures (model, tool, critic)
never escape as exceptions.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from react_agent.critic import Reflector, default_critic
from react_agent.guards import Clock, IterationGuard, LoopDetector, repeat_warning
from react_agent.llm import ResilientLLM
from react_agent.parser import parse_output
from react_agent.prompts import build_messages, system_prompt
from react_agent.tools import ToolRegistry, default_registry
from react_agent.types import (
    Action,
    AgentConfig,
    AgentResult,
    Critic,
    LLMClient,
    LoopVerdict,
    ParseFailure,
    Status,
    Step,
    TerminationReason,
)

StepCallback = Callable[[Step], None]

_STOP_EXPLANATIONS = {
    TerminationReason.MAX_ITERATIONS: "the iteration limit was reached",
    TerminationReason.TIME_BUDGET: "the time budget was exhausted",
    TerminationReason.LOOP_DETECTED: "it kept repeating the same action",
    TerminationReason.ERROR_LIMIT: "too many consecutive steps failed",
    TerminationReason.LLM_UNAVAILABLE: "no language model was available",
}


class ReActAgent:
    """Answers a question by alternating model reasoning with tool use, within hard limits."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry | None = None,
        config: AgentConfig | None = None,
        *,
        critic: Critic | None = None,
        reflector: Reflector | None = None,
        fallbacks: Sequence[LLMClient] = (),
        llm_retries: int = 2,
        llm_backoff_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        clock: Clock = time.monotonic,
    ) -> None:
        self.config = config or AgentConfig()
        self.tools = tools if tools is not None else default_registry()
        self.llm = ResilientLLM(
            llm,
            fallbacks,
            retries=llm_retries,
            backoff_seconds=llm_backoff_seconds,
            sleep=sleep,
        )
        self.critic = critic if critic is not None else default_critic()
        self.reflector = reflector or Reflector()
        self._clock = clock

    def run(self, question: str, on_step: StepCallback | None = None) -> AgentResult:
        """Run the loop until a final answer is accepted or a limit stops it."""
        return _Run(self, question, on_step).execute()


class _Run:
    """State of a single agent run (kept separate so the agent itself is reusable)."""

    def __init__(self, agent: ReActAgent, question: str, on_step: StepCallback | None) -> None:
        self.agent = agent
        self.config = agent.config
        self.question = question
        self.on_step = on_step
        self.guard = IterationGuard(self.config, clock=agent._clock)
        self.loops = LoopDetector(self.config.max_repeats)
        self.steps: list[Step] = []
        self.notes: list[str] = []
        self.revisions = 0
        self.failed_signatures: set[tuple[str, str]] = set()
        self.last_success: str | None = None
        self.system = system_prompt(agent.tools.describe(), self.config.max_iterations)
        self._llm_notes_start = len(agent.llm.notes)

    # ------------------------------------------------------------------ main loop

    def execute(self) -> AgentResult:
        self.guard.start()
        while True:
            # Observe: have we hit a limit before spending another iteration?
            reason = self.guard.check()
            if reason is not None:
                return self._stop(reason)
            self.guard.record_iteration()

            # Think: ask the model for one thought and one action.
            raw = self._think()
            if raw is None:
                return self._stop(TerminationReason.LLM_UNAVAILABLE)

            parsed = parse_output(raw)
            if isinstance(parsed, ParseFailure):
                self._record(raw, "", None, parsed.message, is_error=True)
                continue

            # Act: either propose a final answer or call a tool.
            action = parsed.action
            if action.is_finish:
                result = self._propose_answer(raw, parsed.thought, action)
                if result is not None:
                    return result
                continue

            verdict = self.loops.register(action)
            if verdict is LoopVerdict.LOOP:
                observation = (
                    f"Loop detected: {action} was requested {self.loops.count(action)} times."
                )
                self._record(raw, parsed.thought, action, observation, is_error=True)
                return self._stop(TerminationReason.LOOP_DETECTED)
            if verdict is LoopVerdict.REPEAT:
                cached = self.loops.cached(action) or ""
                is_error = action.signature in self.failed_signatures
                self._record(
                    raw,
                    parsed.thought,
                    action,
                    repeat_warning(action, cached),
                    is_error=is_error,
                    repeated=True,
                )
                continue
            self._act(raw, parsed.thought, action)

    def _think(self) -> str | None:
        messages = build_messages(self.system, self.question, self.steps)
        try:
            return self.agent.llm.complete(messages)
        except Exception as exc:  # any model failure ends the run gracefully
            self.notes.append(f"LLM unavailable: {exc}")
            return None

    def _act(self, raw: str, thought: str, action: Action) -> None:
        result = self.agent.tools.execute(
            action,
            timeout=self.config.tool_timeout_seconds,
            max_chars=self.config.max_observation_chars,
        )
        self.loops.remember(action, result.output)
        if result.is_error:
            self.failed_signatures.add(action.signature)
        else:
            self.last_success = result.output
        self._record(raw, thought, action, result.output, is_error=result.is_error)

    def _propose_answer(self, raw: str, thought: str, action: Action) -> AgentResult | None:
        """Self-critique gate for a proposed final answer. Returns a result to end the run."""
        answer = action.input.strip()
        critique = self.agent.critic.critique(self.question, answer, self.steps)
        if critique.unavailable:
            self.notes.append(f"Self-critique unavailable; answer accepted: {critique.feedback}")

        if critique.approved:
            self._record(
                raw,
                thought,
                action,
                "Final answer accepted.",
                is_error=False,
                reflection="Self-critique approved the answer.",
            )
            return self._result(answer, Status.SUCCESS, TerminationReason.FINAL_ANSWER)

        self.revisions += 1
        if self.revisions > self.config.max_revisions:
            self._record(
                raw,
                thought,
                action,
                f"Critique: {critique.feedback}",
                is_error=False,
                reflection="Revision budget exhausted; returning the answer as degraded.",
            )
            self.notes.append(
                f"Answer returned after {self.config.max_revisions} rejected revisions; "
                f"last critique: {critique.feedback}"
            )
            return self._result(answer, Status.DEGRADED, TerminationReason.FINAL_ANSWER)

        observation = (
            f"Your final answer was rejected by the critic: {critique.feedback} "
            f"Revise your approach ({self.config.max_revisions - self.revisions + 1} "
            "revision(s) left)."
        )
        self._record(
            raw,
            thought,
            action,
            observation,
            is_error=False,
            reflection="The critic rejected the answer; address its feedback before finishing.",
        )
        return None

    # ------------------------------------------------------------------ bookkeeping

    def _record(
        self,
        raw: str,
        thought: str,
        action: Action | None,
        observation: str,
        *,
        is_error: bool,
        repeated: bool = False,
        reflection: str | None = None,
    ) -> None:
        """Reflect on the step, store it and update the error counter."""
        if reflection is None:
            reflection = self.agent.reflector.reflect(
                action,
                observation,
                is_error=is_error,
                repeated=repeated,
                remaining_iterations=self.guard.remaining_iterations,
            )
        step = Step(
            index=len(self.steps) + 1,
            thought=thought,
            action=action,
            observation=observation,
            reflection=reflection,
            is_error=is_error,
            raw_output=raw,
        )
        self.steps.append(step)
        self.guard.record_step(is_error=is_error)
        if self.on_step is not None:
            self.on_step(step)

    def _stop(self, reason: TerminationReason) -> AgentResult:
        """Graceful degradation: build the best answer we can from what we observed."""
        explanation = _STOP_EXPLANATIONS[reason]
        if self.last_success is not None:
            answer = (
                f"Best-effort answer (stopped early because {explanation}): {self.last_success}"
            )
            return self._result(answer, Status.DEGRADED, reason)
        answer = f"No answer: the agent stopped because {explanation} before making progress."
        return self._result(answer, Status.FAILED, reason)

    def _result(self, answer: str, status: Status, reason: TerminationReason) -> AgentResult:
        return AgentResult(
            answer=answer,
            status=status,
            termination_reason=reason,
            steps=list(self.steps),
            iterations=self.guard.iterations,
            elapsed_seconds=self.guard.elapsed,
            notes=[*self.agent.llm.notes[self._llm_notes_start :], *self.notes],
        )
