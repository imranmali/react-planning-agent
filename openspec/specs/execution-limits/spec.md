# execution-limits Specification

## Purpose
Guarantees that every agent run terminates in bounded time and steps, no matter how the
reasoning model or tools behave, by enforcing explicit, configurable limits.

## Requirements

### Requirement: Maximum iteration limit
The agent SHALL stop after at most `max_iterations` think/act cycles (default 10) and
report termination reason `max_iterations`. A `max_iterations` lower than 1 SHALL be
rejected when the agent is configured.

#### Scenario: Model never finishes
- **WHEN** the model keeps requesting new, distinct actions and `max_iterations` is 4
- **THEN** the run stops after exactly 4 iterations with termination reason `max_iterations`

#### Scenario: Invalid limit
- **WHEN** an agent is configured with `max_iterations` of 0
- **THEN** configuration fails with a validation error before any run starts

### Requirement: Wall-clock time budget
The agent SHALL check the elapsed time before each iteration and stop with termination
reason `time_budget` once the configured time budget is exhausted.

#### Scenario: Budget exhausted
- **WHEN** the time budget is 1 second and 1.5 seconds have elapsed after an iteration
- **THEN** no further iteration starts and the termination reason is `time_budget`

### Requirement: Repeated action loop detection
The agent SHALL detect when the same action (same tool and same normalised input) is
requested repeatedly. A repeated action below the threshold SHALL NOT be re-executed;
the cached observation is returned with a warning. Reaching `max_repeats` (default 3)
occurrences SHALL stop the run with termination reason `loop_detected`.

#### Scenario: Warning on first repeat
- **WHEN** the model requests `lookup[france]` a second time
- **THEN** the tool is not executed again and the observation repeats the earlier result prefixed with a warning that the action was already performed

#### Scenario: Loop stopped
- **WHEN** the model requests `lookup[france]` for the third time with `max_repeats` of 3
- **THEN** the run stops with termination reason `loop_detected`

#### Scenario: Alternating cycle stopped
- **WHEN** the model alternates `lookup[a]`, `lookup[b]`, `lookup[a]`, `lookup[b]`, `lookup[a]`
- **THEN** the run stops with termination reason `loop_detected`

### Requirement: Consecutive error limit
The agent SHALL stop with termination reason `error_limit` after `max_consecutive_errors`
(default 3) consecutive failed steps. A successful step SHALL reset the counter.

#### Scenario: Errors in a row
- **WHEN** three consecutive steps fail because of unknown tools or unparseable output
- **THEN** the run stops with termination reason `error_limit`

#### Scenario: Recovery resets the counter
- **WHEN** two steps fail, the third succeeds and the fourth fails
- **THEN** the run continues because the consecutive error count is 1
