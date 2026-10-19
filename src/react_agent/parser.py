"""Tolerant parsing of model output into a thought plus exactly one action.

Two formats are accepted (design D2):

* ReAct text - ``Thought: ...`` followed by ``Action: tool[input]`` (or LangChain-style
  ``Action: tool`` + ``Action Input: input``) or ``Final Answer: ...``;
* a JSON object - ``{"thought": ..., "action": ..., "input": ...}``.

Labels are case-insensitive and may be wrapped in markdown bold; code fences are ignored.
A final answer is returned as the reserved ``finish`` action. ``parse_output`` never
raises: anything it cannot understand becomes a ``ParseFailure`` whose message explains
the expected format to the model.
"""

from __future__ import annotations

import json
import re
from typing import Any

from react_agent.types import FINISH_ACTION, Action, ParsedOutput, ParseFailure

FORMAT_HINT = (
    "Reply in exactly this format:\n"
    "Thought: <your reasoning about what to do next>\n"
    "Action: <tool name>[<tool input>]\n"
    "or, when you know the answer:\n"
    "Thought: <why you are confident>\n"
    "Final Answer: <the answer>\n"
    'You may instead reply with one JSON object: {"thought": "...", '
    '"action": "<tool name or finish>", "input": "..."}'
)
"""Shown to the model whenever its output cannot be parsed."""

_LABEL = re.compile(
    r"^[ \t]*(?:[*_#>]+[ \t]*)?"  # optional markdown: **Thought, ## Thought, > Thought
    r"(?P<label>thought|action[ \t_]*input|action|final[ \t_]*answer|observation)"
    r"(?:[ \t]*\d+)?[ \t]*(?:[*_]+[ \t]*)?:(?:\*\*|__)?[ \t]*",  # "Thought 1:", "**Thought**:"
    re.IGNORECASE | re.MULTILINE,
)
_FENCED_BLOCK = re.compile(r"```[\w+-]*[ \t]*\n?(.*?)```", re.DOTALL)
_STRAY_FENCE = re.compile(r"^[ \t]*```[\w+-]*[ \t]*$", re.MULTILINE)
_SEPARATORS = re.compile(r"[\s_-]+")
_LABEL_SEPARATORS = re.compile(r"[\s_]+")
_FINISH_ALIASES = frozenset({"finish", "final answer", "finalanswer"})
_JSON_INPUT_KEYS = ("input", "action_input", "tool_input", "answer")
_JSON_ANSWER_KEYS = ("final_answer", "final answer", "finalanswer")
_NAME_DECORATION = " \t\r\n*_`'\""


def parse_output(text: str) -> ParsedOutput | ParseFailure:
    """Parse one model reply. Never raises."""
    body = _strip_code_fences(text).strip()
    if not body:
        return _failure("the reply was empty.", text)

    json_object = _find_json_object(body)
    if json_object is not None and body.startswith("{"):
        return _finish_parse(_from_json(json_object), text)

    result = _from_text(body)
    if isinstance(result, str) and json_object is not None:
        embedded = _from_json(json_object)
        if isinstance(embedded, ParsedOutput):
            return embedded
    return _finish_parse(result, text)


def _failure(reason: str, raw: str) -> ParseFailure:
    return ParseFailure(message=f"Could not parse your reply: {reason}\n{FORMAT_HINT}", raw=raw)


def _finish_parse(result: ParsedOutput | str, raw: str) -> ParsedOutput | ParseFailure:
    return _failure(result, raw) if isinstance(result, str) else result


def _strip_code_fences(text: str) -> str:
    """Replace fenced blocks by their contents and drop unmatched fence lines."""
    unfenced = _FENCED_BLOCK.sub(lambda match: match.group(1), text)
    return _STRAY_FENCE.sub("", unfenced)


def _canonical_tool(name: str) -> str:
    """Strip decoration from a tool name and map final-answer aliases to ``finish``."""
    cleaned = name.strip(_NAME_DECORATION)
    if _SEPARATORS.sub(" ", cleaned).lower() in _FINISH_ALIASES:
        return FINISH_ACTION
    return cleaned


def _split_bracketed(spec: str) -> tuple[str, str] | None:
    """Split ``tool[input]`` into name and input; the input ends at the LAST ``]``."""
    start, end = spec.find("["), spec.rfind("]")
    if start == -1 or end < start:
        return None
    return spec[:start], spec[start + 1 : end].strip()


def _strip_quotes(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        return text[1:-1].strip()
    return text


# --------------------------------------------------------------------------- text format


def _from_text(body: str) -> ParsedOutput | str:
    """Parse the ReAct text format; return a failure reason string on error."""
    matches = list(_LABEL.finditer(body))
    sections = [
        (
            _LABEL_SEPARATORS.sub("", match["label"]).lower(),
            body[match.end() : matches[i + 1].start() if i + 1 < len(matches) else len(body)],
        )
        for i, match in enumerate(matches)
    ]
    preamble = body[: matches[0].start()] if matches else body

    for index, (label, content) in enumerate(sections):
        if label not in ("action", "finalanswer"):
            continue
        thoughts = [text for kind, text in sections[:index] if kind == "thought"]
        thought = (thoughts[0] if thoughts else preamble).strip()
        if label == "finalanswer":
            return ParsedOutput(thought, Action(FINISH_ACTION, content.strip()))
        following = sections[index + 1] if index + 1 < len(sections) else None
        action = _text_action(content.strip(), following)
        return action if isinstance(action, str) else ParsedOutput(thought, action)
    return "found neither an 'Action:' nor a 'Final Answer:' line."


def _text_action(spec: str, following: tuple[str, str] | None) -> Action | str:
    """Build an action from ``tool[input]`` or ``tool`` + a following ``Action Input:``."""
    split = _split_bracketed(spec)
    if split is not None:
        name, tool_input = split
    elif following is not None and following[0] == "actioninput":
        name, tool_input = spec, _strip_quotes(following[1].strip())
    else:
        first_line = spec.splitlines()[0] if spec else ""
        return f"the action {first_line!r} has no input; write it as 'Action: tool[input]'."
    tool = _canonical_tool(name)
    if not tool:
        return "the action has no tool name; write it as 'Action: tool[input]'."
    return Action(tool, tool_input)


# --------------------------------------------------------------------------- JSON format


def _find_json_object(body: str) -> dict[str, Any] | None:
    """Return the JSON object that is, or is embedded in, ``body`` (or ``None``)."""
    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end < start:
        return None
    try:
        value = json.loads(body[start : end + 1])
    except ValueError:
        return None
    return value if isinstance(value, dict) else None


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value)


def _first_present(data: dict[str, Any], keys: tuple[str, ...]) -> tuple[bool, object]:
    for key in keys:
        if key in data:
            return True, data[key]
    return False, None


def _from_json(raw: dict[str, Any]) -> ParsedOutput | str:
    """Interpret a JSON reply; return a failure reason string on error."""
    data = {str(key).strip().lower(): value for key, value in raw.items()}
    thought = _as_text(data.get("thought"))
    action = data.get("action")
    has_input, tool_input = _first_present(data, _JSON_INPUT_KEYS)

    if isinstance(action, dict):  # {"action": {"tool"|"name": ..., "input": ...}}
        nested = {str(key).strip().lower(): value for key, value in action.items()}
        _, action = _first_present(nested, ("tool", "name"))
        has_input, tool_input = _first_present(nested, _JSON_INPUT_KEYS)

    if action is None:
        has_answer, answer = _first_present(data, _JSON_ANSWER_KEYS)
        if has_answer:
            return ParsedOutput(thought, Action(FINISH_ACTION, _as_text(answer)))
        return "the JSON object has no 'action' field."
    if not isinstance(action, str):
        return "the JSON 'action' field must be a tool name string."

    if not has_input:
        split = _split_bracketed(action)
        if split is not None:  # {"action": "calculator[2 + 3]"}
            action, tool_input = split
    tool = _canonical_tool(action)
    if not tool:
        return "the JSON 'action' field must be a non-empty tool name."
    return ParsedOutput(thought, Action(tool, _as_text(tool_input)))
