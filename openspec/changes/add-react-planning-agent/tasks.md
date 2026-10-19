# Tasks

## 1. Project scaffolding

- [x] 1.1 Add `pyproject.toml` (hatchling, src layout, `react-agent` script, `anthropic` extra, dev tools config) and verify `pip install -e .[dev]` succeeds
- [x] 1.2 Add shared contract module `src/react_agent/types.py` and verify `mypy --strict` passes on it

## 2. Tools and parsing (parallel track A)

- [x] 2.1 Implement `ToolRegistry`, tool timeout and observation truncation in `tools.py`; verify `tests/test_tools.py` covers duplicate names, unknown tools, exceptions, timeouts and truncation
- [x] 2.2 Implement safe `calculator` and `lookup` tools; verify tests cover arithmetic, injection rejection, huge exponents and case-insensitive lookup
- [x] 2.3 Implement tolerant ReAct/JSON output parser in `parser.py`; verify `tests/test_parser.py` covers text, JSON, fenced, final answer and unparseable inputs

## 3. Guards and critique (parallel track B)

- [x] 3.1 Implement `IterationGuard` (iterations, time budget, consecutive errors, config validation) in `guards.py`; verify `tests/test_guards.py` with an injectable clock
- [x] 3.2 Implement `LoopDetector` (NEW / REPEAT / LOOP, input normalisation, alternating cycles); verify tests
- [x] 3.3 Implement `Reflector`, `HeuristicCritic` and fail-open `LLMCritic` in `critic.py`; verify `tests/test_critic.py`

## 4. Model backends (parallel track C)

- [x] 4.1 Implement `ScriptedLLM`, `StubbornLLM` and deterministic `OfflinePlannerLLM` in `llm.py`; verify `tests/test_llm.py`
- [x] 4.2 Implement `ResilientLLM` retry/backoff/fallback with injectable sleep; verify tests for recovery, fallback and total failure
- [x] 4.3 Implement lazy optional `AnthropicLLM`; verify tests with a fake SDK module (no network)

## 5. Agent loop and CLI (integration)

- [x] 5.1 Implement prompt rendering and `ReActAgent` observe/think/act/reflect loop; verify `tests/test_agent.py` covers every termination reason and the best-effort answer
- [x] 5.2 Implement CLI (`react-agent`, `python -m react_agent`) with limits, `--trace`, `--json`, `--llm offline|stubborn|anthropic` and exit codes; verify `tests/test_cli.py`
- [x] 5.3 Write README (architecture, usage, guarantees, development) and verify the documented commands run as written

## 6. Delivery pipeline

- [x] 6.1 Add `ci.yml` (ruff, mypy, pytest matrix + coverage gate, openspec validate, build) and verify it is green on the pull request
- [x] 6.2 Add `Dockerfile`, `deploy.yml` (GHCR on main) and `release.yml` (GitHub Release on tags); verify the Docker image builds and runs the CLI in CI
- [x] 6.3 Add CONTRIBUTING (branching, commit convention, OpenSpec workflow) and verify links resolve

## 7. Integration verification

- [x] 7.1 Run the full suite locally (`ruff`, `mypy --strict`, `pytest --cov` ≥ 90%, `openspec validate --all --strict`) and verify all pass

## Workflow follow-up

- Archive the change with `openspec archive add-react-planning-agent` once implementation is verified.
- Open a pull request to `main` and merge once CI is green.
