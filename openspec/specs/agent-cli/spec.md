# agent-cli Specification

## Purpose
Provides a command-line entry point to run the planning agent on a question, tune its
limits and inspect its reasoning trace without writing code.

## Requirements

### Requirement: Run a question from the command line
The `react-agent` command (also `python -m react_agent`) SHALL run the agent on the
question given as an argument using the offline planner by default and print the answer.

#### Scenario: Offline run
- **WHEN** the user runs `react-agent "What is 12 * 7?"`
- **THEN** the command prints `84` and exits with code 0

### Requirement: Exit codes reflect status
The command SHALL exit with code 0 for `success`, 2 for `degraded` and 1 for `failed`.

#### Scenario: Degraded run
- **WHEN** the run ends with status `degraded`
- **THEN** the command exits with code 2

### Requirement: Limits and output options
The command SHALL accept `--max-iterations`, `--time-budget`, `--max-repeats`, `--trace`
to print each step, and `--json` to print the full result as JSON.

#### Scenario: JSON output
- **WHEN** the user passes `--json`
- **THEN** stdout contains a single JSON document with the run result

#### Scenario: Loop guard demo
- **WHEN** the user runs the command with `--llm stubborn`, a demo model that repeats the same action forever
- **THEN** the run stops with termination reason `loop_detected` instead of running forever

### Requirement: Optional hosted model
The command SHALL accept `--llm anthropic` to use a hosted model. When the provider SDK or API key is unavailable it SHALL print a warning to stderr and fall back to the offline planner.

#### Scenario: Missing API key
- **WHEN** the user passes `--llm anthropic` without an API key configured
- **THEN** a warning is printed to stderr and the offline planner answers the question
