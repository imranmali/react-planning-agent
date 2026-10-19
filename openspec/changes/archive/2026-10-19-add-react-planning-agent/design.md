# Design

## Context

Greenfield repository. See `proposal.md` for motivation and `specs/` for the behavior
contract. Constraints: Python 3.10+, zero required runtime dependencies, deterministic
tests (no network), strict typing.

## Goals / Non-Goals

**Goals:**
- A small, readable agent loop where *every* exit path is explicit and tested.
- Each concern (parsing, tools, guards, critique, model resilience) in its own module
  behind a narrow interface so it can be built and tested independently.
- Runs offline out of the box (deterministic planner) and with a hosted model when available.

**Non-Goals:**
- Native provider tool-calling APIs, streaming tokens, async execution.
- Long-term memory, multi-agent orchestration *inside* the agent, web access tools.

## Decisions

### D1. Package layout (src layout, one concern per module)

```
src/react_agent/
  types.py      # shared dataclasses / enums / protocols (the contract)
  parser.py     # model text -> ParsedAction | ParseError
  tools.py      # Tool, ToolRegistry, calculator, lookup, timeouts, truncation
  guards.py     # IterationGuard (iterations, time, errors), LoopDetector
  critic.py     # Reflector (per step), HeuristicCritic, LLMCritic
  llm.py        # ScriptedLLM, OfflinePlannerLLM, StubbornLLM, ResilientLLM, AnthropicLLM
  prompts.py    # system prompt + trajectory rendering
  agent.py      # ReActAgent: the observe/think/act/reflect loop
  cli.py        # argparse entry point; __main__.py delegates here
```

`types.py` is written first and frozen so modules can be built in parallel against it.

### D2. Text protocol between agent and model

Model output is plain text in ReAct format (`Thought:`, `Action: tool[input]`,
`Final Answer:`) or a JSON object `{"thought", "action", "input"}`. `finish` is a reserved
action name meaning "final answer". *Why:* works with any model, including the offline
planner and scripted test doubles; native tool-use APIs are provider specific.
*Alternative:* provider tool-calling — rejected for portability and offline testing.

### D3. Termination is centralised in guards

`IterationGuard.check()` is called at the top of every iteration and returns a
`TerminationReason | None`; `LoopDetector.register(action)` classifies each action as
`NEW`, `REPEAT` (serve cached observation + warning) or `LOOP` (stop). The agent loop
itself has no ad-hoc counters. Loop detection counts identical *(tool, normalised input)*
signatures over the whole run (normalisation: lower-case, collapsed whitespace), which also
catches alternating cycles (A,B,A,B,A). *Alternative:* sliding-window period detection —
more complex, and total-count already bounds every cycle.

### D4. Two-level self-critique

`Reflector` (heuristic, cheap) annotates every step. `Critic` gates final answers:
`HeuristicCritic` rejects empty/placeholder answers; `LLMCritic` asks a model for
`APPROVE` / `REJECT: <reason>` and fails open (approve + note) on error. Rejections are
bounded by `max_revisions`. *Why fail open:* a broken critic must not turn a good run into a
failed one — degraded transparency (notes) is preferred over losing the answer.

### D5. Model resilience as a decorator

`ResilientLLM(primary, fallbacks, retries, backoff, sleep)` implements the same
`LLMClient` protocol, so the agent is unaware of retries/fallbacks. It records which
backend answered in `notes`. `sleep` is injectable so tests never wait. Exhausting all
backends raises `LLMUnavailableError`, which the agent maps to `llm_unavailable`.

### D6. Tool timeouts via worker thread

Tools run in a `ThreadPoolExecutor` future with `result(timeout=...)`. A timed-out thread
cannot be killed in CPython; it is abandoned (daemon worker) and the step is an error.
Acceptable for a reference agent; documented in README.

### D7. Safe calculator via `ast`

Parse with `ast.parse(mode="eval")` and walk an allow-list of node types (numbers,
BinOp, UnaryOp, calls to a fixed `math` function table, named constants `pi`, `e`, `tau`).
Exponents are bounded (|exponent| ≤ 1000 and base magnitude check) to prevent CPU
exhaustion. *Alternative:* `eval` with empty builtins — rejected, trivially escapable.

### D8. Offline planner

`OfflinePlannerLLM` is a deterministic rule-based "model" reading the rendered prompt:
it looks up knowledge-base keys mentioned in the question, then evaluates any arithmetic
expression (substituting looked-up numbers), then answers with the latest observation.
It exists for demos/CLI defaults and integration tests; it is not meant to be smart.

### D9. Optional Anthropic backend

`AnthropicLLM` imports `anthropic` lazily; missing SDK or credentials raises
`LLMUnavailableError` at construction so the CLI can warn and fall back. Default model
`claude-opus-5-5` (overridable via `--model` / `REACT_AGENT_MODEL`). A `refusal` stop
reason or any `anthropic.APIError` becomes `LLMError`, so `ResilientLLM` can fall back.

### D10. CI/CD

- `ci.yml` (push + PR): ruff lint/format check, mypy --strict, pytest matrix 3.10–3.13 with
  coverage gate (90%), `openspec validate --all --strict`, package + Docker build smoke test.
- `deploy.yml` (push to `main`): build and push container image to GHCR
  (`ghcr.io/<owner>/react-planning-agent:{sha,latest}`) using `GITHUB_TOKEN`.
- `release.yml` (tag `v*`): build sdist/wheel and publish a GitHub Release with artifacts.

## Risks / Trade-offs

- [Abandoned tool threads keep running after timeout] → documented; tools should be
  short/idempotent; daemon threads do not block interpreter exit.
- [Heuristic critic can reject legitimate terse answers] → only empty/placeholder answers
  are rejected; revisions are bounded and the answer is still returned (degraded).
- [Text-format parsing is brittle with chatty models] → tolerant parser (case-insensitive
  labels, JSON alternative, fenced code stripping); parse failures feed back a format hint.
- [Cached repeat observations hide non-deterministic tool changes] → acceptable: repeating
  an identical action is exactly the behavior we want to discourage.

## Migration Plan

Greenfield; no migration. Rollback = revert the merge commit on `main`.
