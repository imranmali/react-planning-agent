# ReAct Planning Agent

[![CI](https://github.com/imranmali/react-planning-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/imranmali/react-planning-agent/actions/workflows/ci.yml)
[![Deploy](https://github.com/imranmali/react-planning-agent/actions/workflows/deploy.yml/badge.svg)](https://github.com/imranmali/react-planning-agent/actions/workflows/deploy.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)

A ReAct (Reason + Act) planning agent that **observes, thinks, acts and reflects** — and
is built so it **cannot loop forever**. Every run ends with an explicit termination
reason, an honest status, and the best answer it could get.

```text
$ react-agent "What is the france population times 2?"
136340000

$ react-agent --llm stubborn "Keep checking forever"
Best-effort answer (stopped early because it kept repeating the same action): 68170000
[degraded: loop_detected after 3 iteration(s)]
```

## Why

Naive agent loops fail in familiar ways: they repeat the same tool call, keep retrying a
broken tool, never decide they are done, or crash when the model provider hiccups. This
project is a small, readable reference for the opposite: an agent whose every exit path is
designed, specified and tested.

## Guarantees

| Failure mode | Guard | Outcome |
| --- | --- | --- |
| Model never finishes | `max_iterations` (default 10) | stops with `max_iterations` |
| Run takes too long | `time_budget_seconds` (default 60) | stops with `time_budget` |
| Same action over and over (incl. A,B,A,B cycles) | loop detector, `max_repeats` (default 3) | first repeat served from cache with a warning; then stops with `loop_detected` |
| Garbage output / unknown tools / failing tools | `max_consecutive_errors` (default 3) | errors become observations; stops with `error_limit` |
| Weak or empty final answer | self-critique, `max_revisions` (default 2) | answer sent back for revision; returned as `degraded` when revisions run out |
| Model provider down | retry with backoff, then fallback models | stops with `llm_unavailable` only when every model fails |
| Tool hangs | per-tool timeout (default 10 s) | timeout observation, run continues |
| Tool raises | exception boundary | error observation, run continues |

`ReActAgent.run()` never raises for runtime failures. Only invalid configuration raises
(`ConfigError`) — at construction time.

Every result carries a **status**:

- `success` — a final answer was proposed and approved by the critic.
- `degraded` — an answer is returned but it is partial (a limit stopped the run after some
  progress) or the critic never approved it. The answer says why.
- `failed` — the run stopped before making any progress; the answer explains the stop reason.

## How it works

```text
            ┌──────────────────────────────────────────────────────────────┐
            │                        IterationGuard                        │
            │      iterations · time budget · consecutive errors           │
            └──────────────┬───────────────────────────────────────────────┘
                           │ check() before every iteration
                           ▼
  Question ──► OBSERVE ──► THINK ──────────► ACT ──────────────► REFLECT ──┐
               transcript  model (with       tool call via        Reflector │
               + reflections retry/fallback)  ToolRegistry         notes    │
                           │                  │ (timeout, truncation)       │
                           │                  ▼                             │
                           │          LoopDetector: NEW / REPEAT / LOOP     │
                           │                                                │
                           └── "Final Answer" ──► Critic ── approve ──► result
                                                   │ reject (≤ max_revisions)
                                                   └──► back to OBSERVE ◄───┘

          any limit hit ──► best-effort answer from the last successful observation
```

| Module | Responsibility |
| --- | --- |
| `react_agent/types.py` | Shared contract: `Action`, `Step`, `AgentResult`, `AgentConfig`, enums, errors, protocols |
| `react_agent/prompts.py` | System prompt and transcript rendering |
| `react_agent/parser.py` | Tolerant parsing of ReAct text (`Thought:` / `Action: tool[input]` / `Final Answer:`) and JSON output |
| `react_agent/tools.py` | `ToolRegistry` with timeouts and truncation; safe AST `calculator`; `lookup` knowledge base |
| `react_agent/guards.py` | `IterationGuard` and `LoopDetector` — the termination guarantees |
| `react_agent/critic.py` | Per-step `Reflector`; `HeuristicCritic`, fail-open `LLMCritic`, `ChainCritic` |
| `react_agent/llm.py` | `OfflinePlannerLLM`, `StubbornLLM`, `ScriptedLLM`, `ResilientLLM`, optional `AnthropicLLM` |
| `react_agent/agent.py` | `ReActAgent` — the observe/think/act/reflect loop |
| `react_agent/cli.py` | `react-agent` command |

## Installation

Requires Python 3.10+. The core has **no runtime dependencies**.

```bash
pip install "git+https://github.com/imranmali/react-planning-agent"
# optional hosted model support
pip install "react-planning-agent[anthropic] @ git+https://github.com/imranmali/react-planning-agent"
```

Or run the container image published by the deploy workflow:

```bash
docker run --rm ghcr.io/imranmali/react-planning-agent:latest "What is 12 * 7?"
```

## Command-line usage

```bash
react-agent "What is 12 * 7?"                 # 84
react-agent --trace "What is the earth radius times 2?"   # show every step on stderr
react-agent --json "What is the capital of japan?"       # full result as JSON
react-agent --llm stubborn --max-repeats 3 "anything"     # watch the loop guard fire
react-agent --llm anthropic "What is 2 ** 10?"            # hosted model + offline fallback
python -m react_agent --help
```

| Option | Default | Meaning |
| --- | --- | --- |
| `--llm {offline,stubborn,anthropic}` | `offline` | reasoning model |
| `--model` | `claude-opus-5-5` | model id for `--llm anthropic` (or `REACT_AGENT_MODEL`) |
| `--llm-retries` | `2` | retries per model before falling back |
| `--max-iterations` | `10` | hard cap on think/act cycles |
| `--time-budget` | `60` | wall-clock budget in seconds |
| `--max-repeats` | `3` | identical actions allowed before `loop_detected` |
| `--max-errors` | `3` | consecutive failed steps before `error_limit` |
| `--max-revisions` | `2` | critic rejections before returning a degraded answer |
| `--tool-timeout` | `10` | per-tool timeout in seconds |
| `--trace` / `--json` | off | print steps to stderr / print the result as JSON |

Exit codes: `0` success, `2` degraded, `1` failed, `64` usage error.

`--llm anthropic` needs `pip install "react-planning-agent[anthropic]"` and an
`ANTHROPIC_API_KEY`. Without them the CLI prints a warning and uses the offline planner;
with them, the offline planner is still configured as a fallback model.

The offline planner is a deterministic, rule-based stand-in for a model: it looks up
knowledge-base facts mentioned in the question (`france population`, `earth radius`,
`capital of japan`, …), evaluates arithmetic, and answers with the latest observation. It
makes the agent runnable and testable without network access or API keys.

## Library usage

```python
from react_agent.agent import ReActAgent
from react_agent.critic import default_critic
from react_agent.llm import AnthropicLLM, OfflinePlannerLLM
from react_agent.tools import Tool, default_registry
from react_agent.types import AgentConfig

tools = default_registry()
tools.register(Tool("echo", "Repeat the input back.", lambda text: text))

primary = AnthropicLLM()  # or any object with .name and .complete(messages)
agent = ReActAgent(
    primary,
    tools,
    AgentConfig(max_iterations=8, time_budget_seconds=30, max_repeats=3),
    critic=default_critic(primary),  # heuristic + model-based critique
    fallbacks=[OfflinePlannerLLM()],  # graceful degradation
)

result = agent.run("What is the eiffel tower height times 3?")
print(result.status.value, result.termination_reason.value, result.answer)
for step in result.steps:
    print(step.index, step.action, "->", step.observation, "|", step.reflection)
```

Any object with a `name` attribute and a `complete(messages) -> str` method can be the
model. Tools are plain `str -> str` callables; raise `ToolError` for expected failures.

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

ruff check . && ruff format --check .   # lint + format
mypy                                    # strict type checking
pytest --cov                            # tests, coverage gate 90 %
openspec validate --all --strict        # spec validation (npm i -g @fission-ai/openspec)
```

The project is developed spec-first with [OpenSpec](https://github.com/Fission-AI/OpenSpec):

- `openspec/specs/` — the current behavior contract, one capability per directory
  (`react-loop`, `execution-limits`, `self-critique`, `graceful-degradation`,
  `tool-execution`, `agent-cli`).
- `openspec/changes/` — proposals in flight; `openspec/changes/archive/` — completed changes.

See [CONTRIBUTING.md](CONTRIBUTING.md) for the branching model, commit conventions and the
propose → apply → archive workflow.

## CI/CD

| Workflow | Trigger | What it does |
| --- | --- | --- |
| `ci.yml` | push to `main`, pull requests | ruff, mypy, pytest on Python 3.10–3.13 with coverage gate, OpenSpec validation, wheel build + CLI smoke test, Docker build + smoke test |
| `deploy.yml` | push to `main` | builds and publishes `ghcr.io/imranmali/react-planning-agent` (`latest` + commit SHA), then smoke-tests the published image |
| `release.yml` | tag `v*` | verifies the tag matches the package version, builds sdist + wheel, publishes a GitHub Release |

## Limitations

- Timed-out tools run in daemon threads that cannot be killed in CPython; they are
  abandoned, not stopped. Keep tools short and side-effect free.
- The heuristic critic only rejects empty or placeholder answers; use `LLMCritic` for
  semantic review.
- The offline planner is a demo, not a reasoner.

## License

[MIT](LICENSE) © 2026 Imran Ali
