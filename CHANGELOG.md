# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-10-19

### Added

- ReAct agent loop that observes, thinks, acts and reflects on each step, with a
  structured trace of every iteration.
- Hard execution limits: maximum iterations, wall-clock time budget and maximum
  consecutive errors, so a run always terminates.
- Loop detection for repeated and alternating actions (with normalised inputs) that
  stops a run with termination reason `loop_detected`.
- Self-critique of candidate answers with a bounded number of revisions.
- Graceful degradation: LLM retry with backoff and fallback model, and a best-effort
  answer with status `degraded` instead of an unhandled failure.
- Safe built-in tools (an arithmetic-only `calculator` that never executes arbitrary
  code, and `lookup`) with per-tool timeouts and observation truncation.
- Tolerant parser for text and JSON ReAct output.
- Offline deterministic planner by default, `stubborn` demo model, and optional
  Anthropic backend via the `anthropic` extra.
- `react-agent` CLI (also `python -m react_agent`) with limit flags, `--trace`, `--json`
  and exit codes 0 (success), 2 (degraded) and 1 (failed).
- CI/CD: lint, type check, test matrix (Python 3.10-3.13) with a 90% coverage gate,
  OpenSpec validation, package and container smoke tests, container image publishing
  to GHCR and GitHub Releases on version tags.

[Unreleased]: https://github.com/imranmali/react-planning-agent/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/imranmali/react-planning-agent/releases/tag/v0.1.0
