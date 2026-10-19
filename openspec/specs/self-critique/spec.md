# self-critique Specification

## Purpose
Makes the agent evaluate its own progress after every action and scrutinise its final
answer before returning it, so weak answers are revised within a bounded budget.

## Requirements

### Requirement: Per-step reflection
After each action the agent SHALL produce a reflection that assesses the step (new
information gained, error encountered, or repeated action) and suggests how to proceed.
The reflection SHALL be stored on the step and included in the next prompt.

#### Scenario: Reflection on a failed tool
- **WHEN** a tool call fails with an error
- **THEN** the step's reflection states that the action failed and recommends a different tool or input

#### Scenario: Reflection on progress
- **WHEN** a tool call succeeds with a new observation
- **THEN** the step's reflection records that progress was made

### Requirement: Final answer critique
Before accepting a proposed final answer the agent SHALL submit it to a critic. A rejected answer SHALL NOT end the run while revisions remain; the critique is returned to the model as the observation and the loop continues.

#### Scenario: Empty answer rejected
- **WHEN** the model proposes an empty final answer and revisions remain
- **THEN** the critic rejects it, the observation contains the critique and the run continues

#### Scenario: Good answer accepted
- **WHEN** the model proposes a non-empty answer supported by the trajectory
- **THEN** the critic approves it and the run ends with status `success`

### Requirement: Bounded revisions
The number of rejected final answers SHALL be limited by `max_revisions` (default 2).
When the limit is exceeded the latest proposed answer SHALL be returned with status
`degraded` and termination reason `final_answer`, including the critic's notes.

#### Scenario: Revisions exhausted
- **WHEN** the critic rejects three consecutive proposed answers and `max_revisions` is 2
- **THEN** the third answer is returned with status `degraded` and the critique notes are attached to the result

### Requirement: Pluggable critic
The agent SHALL support a model-based critic that asks a model to approve or reject the
answer. If the critic itself fails, the answer SHALL be treated as approved and the
failure SHALL be recorded in the result notes.

#### Scenario: Critic failure
- **WHEN** the model-based critic raises an error while reviewing an answer
- **THEN** the answer is accepted and the result notes mention that critique was unavailable
