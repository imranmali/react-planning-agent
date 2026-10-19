"""Command-line interface: ``react-agent "question"``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from typing import NoReturn, TextIO

from react_agent import __version__
from react_agent.agent import ReActAgent
from react_agent.critic import default_critic
from react_agent.llm import AnthropicLLM, OfflinePlannerLLM, StubbornLLM
from react_agent.tools import DEFAULT_KNOWLEDGE_BASE, default_registry
from react_agent.types import (
    AgentConfig,
    AgentResult,
    ConfigError,
    LLMClient,
    LLMUnavailableError,
    Status,
    Step,
)

EXIT_CODES = {Status.SUCCESS: 0, Status.FAILED: 1, Status.DEGRADED: 2}
EXIT_USAGE = 64


class _ArgumentParser(argparse.ArgumentParser):
    """Use a dedicated usage exit code so 2 always means "degraded"."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_USAGE, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        prog="react-agent",
        description=(
            "Bounded ReAct planning agent: observe, think, act, reflect - "
            "with iteration limits, loop detection, self-critique and graceful degradation."
        ),
        epilog="Exit codes: 0 success, 2 degraded answer, 1 failed, 64 usage error.",
    )
    parser.add_argument("question", help="the task or question for the agent")
    parser.add_argument(
        "--llm",
        choices=["offline", "stubborn", "anthropic"],
        default="offline",
        help="reasoning model: deterministic offline planner (default), a stubborn demo "
        "model that repeats itself forever, or a hosted Anthropic model",
    )
    parser.add_argument("--model", help="model id for --llm anthropic")
    parser.add_argument(
        "--llm-retries",
        type=int,
        default=2,
        help="retries per model before falling back (default: %(default)s)",
    )
    defaults = AgentConfig()
    parser.add_argument("--max-iterations", type=int, default=defaults.max_iterations)
    parser.add_argument(
        "--time-budget",
        type=float,
        default=defaults.time_budget_seconds,
        help="wall-clock budget in seconds (default: %(default)s)",
    )
    parser.add_argument("--max-repeats", type=int, default=defaults.max_repeats)
    parser.add_argument(
        "--max-errors", type=int, default=defaults.max_consecutive_errors, dest="max_errors"
    )
    parser.add_argument("--max-revisions", type=int, default=defaults.max_revisions)
    parser.add_argument("--tool-timeout", type=float, default=defaults.tool_timeout_seconds)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--json", action="store_true", help="print the full result as JSON")
    output.add_argument(
        "--trace", action="store_true", help="print every step to stderr as it happens"
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _select_llm(args: argparse.Namespace, err: TextIO) -> tuple[LLMClient, list[LLMClient]]:
    """Return (primary, fallbacks). Hosted models fall back to the offline planner."""
    offline = OfflinePlannerLLM(DEFAULT_KNOWLEDGE_BASE)
    if args.llm == "stubborn":
        return StubbornLLM(), []
    if args.llm == "anthropic":
        try:
            return AnthropicLLM(args.model), [offline]
        except LLMUnavailableError as exc:
            print(f"warning: {exc}; falling back to the offline planner", file=err)
    return offline, []


def print_step(step: Step, err: TextIO) -> None:
    marker = "!" if step.is_error else "-"
    print(f"[{step.index}] Thought: {step.thought or '(none)'}", file=err)
    print(f"    Action: {step.action if step.action else '(unparseable output)'}", file=err)
    print(f"    {marker} Observation: {step.observation}", file=err)
    print(f"    Reflection: {step.reflection}", file=err)


def print_result(result: AgentResult, out: TextIO, err: TextIO, as_json: bool) -> None:
    if as_json:
        print(result.to_json(), file=out)
        return
    print(result.answer, file=out)
    if result.status is not Status.SUCCESS:
        print(
            f"[{result.status.value}: {result.termination_reason.value} "
            f"after {result.iterations} iteration(s)]",
            file=err,
        )
    for note in result.notes:
        print(f"note: {note}", file=err)


def main(
    argv: Sequence[str] | None = None,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    out = out or sys.stdout
    err = err or sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = AgentConfig(
            max_iterations=args.max_iterations,
            time_budget_seconds=args.time_budget,
            max_repeats=args.max_repeats,
            max_consecutive_errors=args.max_errors,
            max_revisions=args.max_revisions,
            tool_timeout_seconds=args.tool_timeout,
        )
    except ConfigError as exc:
        parser.error(str(exc))
    if args.llm_retries < 0:
        parser.error("--llm-retries must be >= 0")

    primary, fallbacks = _select_llm(args, err)
    critic = default_critic(primary if isinstance(primary, AnthropicLLM) else None)
    agent = ReActAgent(
        primary,
        default_registry(DEFAULT_KNOWLEDGE_BASE),
        config,
        critic=critic,
        fallbacks=fallbacks,
        llm_retries=args.llm_retries,
    )
    on_step = (lambda step: print_step(step, err)) if args.trace else None
    result = agent.run(args.question, on_step=on_step)
    print_result(result, out, err, args.json)
    return EXIT_CODES[result.status]
