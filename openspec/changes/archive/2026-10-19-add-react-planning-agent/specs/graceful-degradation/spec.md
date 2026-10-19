# Spec Delta

## Purpose

Ensures the agent degrades predictably instead of crashing or hanging when models, tools
or limits fail, and always tells the caller how trustworthy its answer is.

## ADDED Requirements

### Requirement: Runs never raise for runtime failures
A run SHALL NOT raise exceptions caused by model, tool or critic failures. Only invalid
configuration may raise, and only when the agent is constructed.

#### Scenario: Tool raises an exception
- **WHEN** a tool raises an unexpected exception
- **THEN** the step records an error observation containing the exception message and the run continues

### Requirement: LLM retry and fallback
Model calls SHALL be retried up to the configured retry count with backoff, and then
attempted on each fallback model in order. When every model fails the run SHALL stop
with termination reason `llm_unavailable`.

#### Scenario: Primary model recovers on retry
- **WHEN** the primary model fails once and then succeeds
- **THEN** the run continues using the successful response

#### Scenario: Fallback model used
- **WHEN** the primary model fails on every retry and a fallback model is configured
- **THEN** the fallback model's response is used and the result notes record the fallback

#### Scenario: All models unavailable
- **WHEN** every configured model fails
- **THEN** the run stops with termination reason `llm_unavailable` and a best-effort answer

### Requirement: Tool timeout
Each tool invocation SHALL be bounded by a configurable timeout; a tool that exceeds it SHALL yield an error observation instead of blocking the run.

#### Scenario: Slow tool
- **WHEN** a tool takes longer than the tool timeout
- **THEN** the step observation reports the timeout and the step is marked as an error

### Requirement: Best-effort answer
When a run ends without an accepted final answer, the result SHALL contain a best-effort
answer built from the most recent successful observation with status `degraded`, or, if no
step succeeded, an explanation of why the run stopped with status `failed`.

#### Scenario: Partial progress
- **WHEN** a run hits `max_iterations` after at least one successful tool observation
- **THEN** the answer contains the stop reason and the last successful observation, and the status is `degraded`

#### Scenario: No progress
- **WHEN** a run hits `error_limit` with no successful step
- **THEN** the status is `failed` and the answer explains the stop reason
