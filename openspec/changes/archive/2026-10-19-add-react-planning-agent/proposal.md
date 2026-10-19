# Proposal

## Why

Naive LLM agents frequently spin forever: they repeat the same tool call, keep
retrying after a broken tool, or never decide they are done. We need a reference
ReAct planning agent whose loop is *provably bounded* and which still returns a
useful, honest result when it cannot fully succeed — demonstrating agents that
don't infinite-loop.

## What Changes

- Add a Python package `react_agent` implementing an explicit
  **observe → think → act → reflect** loop over a pluggable LLM and tool set.
- Add hard execution limits: maximum iterations, wall-clock budget, maximum
  consecutive errors, and repeated-action / cyclic-state loop detection.
- Add self-critique: a per-step reflection and a final-answer critique that can
  reject a weak answer and send the agent back for a bounded number of revisions.
- Add graceful degradation: retrying and falling back across LLM backends, tool
  failures surfaced as observations, and a best-effort answer (clearly flagged as
  degraded) whenever the loop stops without a confident final answer.
- Add a tool registry with safe built-in tools (arithmetic calculator without
  `eval`, knowledge-base lookup) and a deterministic offline planner so the agent
  runs without network access or API keys; an Anthropic backend is optional.
- Add a `react-agent` command-line interface.
- Add test suite, CI (lint, type-check, tests, spec validation), container image
  build/publish and release workflows, README and MIT license.

## Capabilities

### New Capabilities
- `react-loop`: the observe/think/act/reflect cycle, step trace and final result contract.
- `execution-limits`: guarantees that every run terminates (iteration cap, time budget, error cap, loop detection).
- `self-critique`: per-step reflection and final-answer critique with bounded revisions.
- `graceful-degradation`: behavior when LLMs, tools or limits fail — retries, fallbacks and best-effort answers.
- `tool-execution`: tool registration, invocation, input validation and safe built-in tools.
- `agent-cli`: command-line entry point for running the agent and inspecting its trace.

### Modified Capabilities
<!-- None: this is a greenfield project. -->

## Impact

- New code under `src/react_agent/`, tests under `tests/`.
- No required third-party runtime dependencies; optional extra `anthropic`.
- New GitHub Actions workflows (`ci.yml`, `deploy.yml`, `release.yml`) and a `Dockerfile`.
