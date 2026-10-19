# react-loop Specification

## Purpose
Defines the observe → think → act → reflect cycle the planning agent runs to answer a
task, and the result contract every run returns to its caller.

## Requirements

### Requirement: Observe-think-act-reflect cycle
Each iteration of the agent SHALL observe the task and trajectory so far, ask the
reasoning model for a thought and exactly one action, execute that action, and record a
reflection on the outcome before the next iteration begins.

#### Scenario: Single tool call then finish
- **WHEN** the model first requests the `calculator` tool with input `2 + 3` and then requests `finish` with answer `5`
- **THEN** the run returns answer `5` with status `success` and termination reason `final_answer`
- **AND** the trace contains two steps, the first holding the thought, the action `calculator[2 + 3]`, the observation `5` and a reflection

#### Scenario: Observations feed the next thought
- **WHEN** a tool returns an observation in iteration N
- **THEN** the prompt sent to the model in iteration N+1 contains that observation and the reflection from iteration N

### Requirement: Structured run result
Every run SHALL return a result containing the final answer text, a status of `success`,
`degraded` or `failed`, a termination reason, the ordered list of steps, the number of
iterations used and the elapsed time.

#### Scenario: Result is serialisable
- **WHEN** a run completes for any reason
- **THEN** the result can be converted to a JSON document containing `answer`, `status`, `termination_reason`, `iterations`, `elapsed_seconds` and `steps`

### Requirement: Tolerant parsing of model output
The agent SHALL accept model output in the ReAct text format (`Thought:` / `Action: tool[input]` or `Final Answer:`) and in a JSON object format. Output it cannot parse SHALL be recorded as a failed step whose observation explains the expected format, without crashing the run.

#### Scenario: Text format action
- **WHEN** the model replies `Thought: need math\nAction: calculator[6*7]`
- **THEN** the agent executes tool `calculator` with input `6*7`

#### Scenario: JSON format final answer
- **WHEN** the model replies `{"thought": "done", "action": "finish", "input": "42"}`
- **THEN** the agent treats `42` as the proposed final answer

#### Scenario: Unparseable output
- **WHEN** the model replies with text that contains neither an action nor a final answer
- **THEN** the step is marked as an error and its observation tells the model the expected format
