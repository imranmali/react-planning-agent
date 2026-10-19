# tool-execution Specification

## Purpose
Defines how tools are registered, described to the model, invoked and validated, and the
safe built-in tools that ship with the agent.

## Requirements

### Requirement: Tool registry
Tools SHALL be registered with a unique name, a description and a callable taking a
single string input. Registering a duplicate name SHALL be rejected. The registry SHALL
render a description of all tools for inclusion in prompts.

#### Scenario: Duplicate registration
- **WHEN** a tool named `calculator` is registered twice
- **THEN** the second registration fails with an error

#### Scenario: Unknown tool requested
- **WHEN** the model requests a tool that is not registered
- **THEN** the step fails with an observation listing the available tool names

### Requirement: Safe calculator
The built-in `calculator` tool SHALL evaluate arithmetic expressions (numbers, `+ - * / // % **`, parentheses and a fixed set of math functions and constants) without executing arbitrary code, and SHALL reject any other construct.

#### Scenario: Arithmetic
- **WHEN** the calculator receives `(2 + 3) * 4 ** 2`
- **THEN** it returns `80`

#### Scenario: Code injection rejected
- **WHEN** the calculator receives `__import__('os').system('ls')`
- **THEN** it fails with an error and nothing is executed

#### Scenario: Oversized exponent rejected
- **WHEN** the calculator receives `9 ** 9 ** 9`
- **THEN** it fails with an error instead of hanging

### Requirement: Knowledge lookup
The built-in `lookup` tool SHALL return the entry for a key from a configurable
knowledge base using case-insensitive matching and SHALL report when no entry exists.

#### Scenario: Known key
- **WHEN** `lookup` receives `France` and the knowledge base holds `france`
- **THEN** it returns that entry

#### Scenario: Unknown key
- **WHEN** `lookup` receives a key not in the knowledge base
- **THEN** it fails with an error naming the missing key

### Requirement: Observation size limit
Observations longer than the configured maximum length SHALL be truncated with a marker
before being stored or shown to the model.

#### Scenario: Long output
- **WHEN** a tool returns 10,000 characters and the limit is 2,000
- **THEN** the observation holds the first 2,000 characters followed by a truncation marker
