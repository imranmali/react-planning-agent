"""Prompt construction: system instructions and the running ReAct transcript."""

from __future__ import annotations

from collections.abc import Sequence

from react_agent.types import FINISH_ACTION, Message, Step

QUESTION_PREFIX = "Question: "
OBSERVATION_PREFIX = "Observation: "
REFLECTION_PREFIX = "Reflection: "

SYSTEM_TEMPLATE = """You are a careful planning agent that solves tasks step by step.
On every turn reply with exactly one thought and exactly one action, in this format:

Thought: <your reasoning about what to do next>
Action: <tool name>[<tool input>]

When you know the answer, reply instead with:

Thought: <why you are confident>
Final Answer: <the answer>

Available tools:
{tools}
- {finish}: finish with the final answer as input.

Rules:
- Use only the tools listed above, one per turn.
- Never repeat an action you already performed; reuse its observation instead.
- If an action fails, change your approach instead of retrying the same thing.
- You have at most {max_iterations} steps.
"""


def system_prompt(tool_descriptions: str, max_iterations: int) -> str:
    return SYSTEM_TEMPLATE.format(
        tools=tool_descriptions, finish=FINISH_ACTION, max_iterations=max_iterations
    )


def observation_message(observation: str, reflection: str) -> str:
    text = f"{OBSERVATION_PREFIX}{observation}"
    if reflection:
        text += f"\n{REFLECTION_PREFIX}{reflection}"
    return text


def build_messages(system: str, question: str, steps: Sequence[Step]) -> list[Message]:
    """Render the full conversation for the next model call."""
    messages = [
        Message("system", system),
        Message("user", f"{QUESTION_PREFIX}{question}"),
    ]
    for step in steps:
        messages.append(Message("assistant", step.raw_output))
        messages.append(Message("user", observation_message(step.observation, step.reflection)))
    return messages


def extract_question(messages: Sequence[Message]) -> str:
    """Return the question from a conversation built by ``build_messages``."""
    for message in messages:
        if message.role == "user" and message.content.startswith(QUESTION_PREFIX):
            return message.content[len(QUESTION_PREFIX) :]
    return ""


def extract_observations(messages: Sequence[Message]) -> list[str]:
    """Return observation texts (without reflections) in order."""
    observations = []
    for message in messages:
        if message.role == "user" and message.content.startswith(OBSERVATION_PREFIX):
            body = message.content[len(OBSERVATION_PREFIX) :]
            observations.append(body.split(f"\n{REFLECTION_PREFIX}", 1)[0])
    return observations
