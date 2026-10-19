"""Tests for parsing model output in the ReAct text and JSON formats."""

from __future__ import annotations

import pytest

from react_agent.parser import FORMAT_HINT, parse_output
from react_agent.types import FINISH_ACTION, Action, ParsedOutput, ParseFailure


def parsed(text: str) -> ParsedOutput:
    result = parse_output(text)
    assert isinstance(result, ParsedOutput), result
    return result


def failed(text: str) -> ParseFailure:
    result = parse_output(text)
    assert isinstance(result, ParseFailure), result
    return result


# --------------------------------------------------------------------------- spec scenarios


def test_text_format_action() -> None:
    assert parse_output("Thought: need math\nAction: calculator[6*7]") == ParsedOutput(
        "need math", Action("calculator", "6*7")
    )


def test_json_format_final_answer() -> None:
    result = parsed('{"thought": "done", "action": "finish", "input": "42"}')

    assert result == ParsedOutput("done", Action(FINISH_ACTION, "42"))
    assert result.action.is_finish


def test_unparseable_output_explains_the_format() -> None:
    text = "I am not sure what to do here."

    result = failed(text)

    assert FORMAT_HINT in result.message
    assert "Action:" in result.message
    assert result.raw == text


# --------------------------------------------------------------------------- text format


@pytest.mark.parametrize(
    ("text", "thought", "tool", "tool_input"),
    [
        ("thought: lower\naction: Calculator[1 + 1]", "lower", "Calculator", "1 + 1"),
        ("THOUGHT: upper\nACTION: lookup[France]", "upper", "lookup", "France"),
        ("  Thought: indented\n  Action: lookup[x]", "indented", "lookup", "x"),
        ("Thought: line one\nline two\nAction: lookup[x]", "line one\nline two", "lookup", "x"),
        ("**Thought:** bold\n**Action:** lookup[x]", "bold", "lookup", "x"),
        ("**Thought**: bold\n**Action**: lookup[x]", "bold", "lookup", "x"),
        ("## Thought: heading\nAction: lookup[x]", "heading", "lookup", "x"),
        ("Thought 1: numbered\nAction 1: lookup[x]", "numbered", "lookup", "x"),
        ("Action: lookup[x]", "", "lookup", "x"),
        ("Let me look it up.\nAction: lookup[x]", "Let me look it up.", "lookup", "x"),
        ("Thought: t\nAction: calculator[ max([1, 2]) ]", "t", "calculator", "max([1, 2])"),
        ("Thought: t\nAction: code[\nline 1\nline 2\n]", "t", "code", "line 1\nline 2"),
        ("Thought: t\nAction: `calculator`[2]", "t", "calculator", "2"),
        ("Thought: t\nAction: **lookup** [x]", "t", "lookup", "x"),
        ("Thought: t\nAction: calculator[]", "t", "calculator", ""),
        ("Thought: t\nAction: search\nAction Input: cats", "t", "search", "cats"),
        ('Thought: t\nAction: search\nAction Input: "cats"', "t", "search", "cats"),
        ("Thought: t\nAction: search\naction_input: 'cats'", "t", "search", "cats"),
        ("Thought: t\nAction: search\nAction Input: 5", "t", "search", "5"),
        ("Thought: t\nAction: search\nAction Input: \"cats'", "t", "search", "\"cats'"),
        ("Thought: t\nAction: lookup[a]\nAction Input: b", "t", "lookup", "a"),
    ],
)
def test_text_actions(text: str, thought: str, tool: str, tool_input: str) -> None:
    assert parsed(text) == ParsedOutput(thought, Action(tool, tool_input))


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("Thought: done\nAction: finish[42]", "42"),
        ("Thought: done\nAction: Finish[42]", "42"),
        ("Thought: done\nAction: Final Answer[42]", "42"),
        ("Thought: done\nAction: final_answer[42]", "42"),
        ("Thought: done\nFinal Answer: 42", "42"),
        ("Thought: done\nfinal answer:   42  ", "42"),
        ("Thought: done\nFinal_Answer: 42", "42"),
        ("Thought: done\n**Final Answer:** 42", "42"),
        ("Final Answer: Paris\nis the capital.", "Paris\nis the capital."),
        ("Thought: done\nFinal Answer:", ""),
    ],
)
def test_text_final_answers(text: str, answer: str) -> None:
    result = parsed(text)

    assert result.action == Action(FINISH_ACTION, answer)
    assert result.action.is_finish


def test_hallucinated_observation_after_action_is_ignored() -> None:
    text = (
        "Thought: compute first\n"
        "Action: calculator[2 + 3]\n"
        "Observation: 5\n"
        "Thought: now I know\n"
        "Final Answer: 5"
    )

    assert parsed(text) == ParsedOutput("compute first", Action("calculator", "2 + 3"))


def test_final_answer_before_action_wins() -> None:
    text = "Thought: easy\nFinal Answer: 5\nObservation: x\nAction: calculator[1]"

    assert parsed(text) == ParsedOutput("easy", Action(FINISH_ACTION, "5"))


def test_first_thought_before_the_action_is_used() -> None:
    text = "Thought: first\nThought: second\nAction: lookup[x]\nThought: after"

    assert parsed(text).thought == "first"


def test_inline_label_text_is_not_a_label() -> None:
    text = "Thought: the word Action: appears inline\nAction: lookup[x]"

    assert parsed(text).thought == "the word Action: appears inline"


def test_text_action_with_json_input_prefers_text_format() -> None:
    text = 'Thought: call api\nAction: api[{"action": "finish", "input": "1"}]'

    assert parsed(text).action == Action("api", '{"action": "finish", "input": "1"}')


# --------------------------------------------------------------------------- code fences


@pytest.mark.parametrize(
    "text",
    [
        "```\nThought: t\nAction: lookup[x]\n```",
        "```text\nThought: t\nAction: lookup[x]\n```",
        "Here you go:\n```\nThought: t\nAction: lookup[x]\n```\nHope that helps",
        '```json\n{"thought": "t", "action": "lookup", "input": "x"}\n```',
        '```{"thought": "t", "action": "lookup", "input": "x"}```',
        '```json\n{"thought": "t", "action": "lookup", "input": "x"}',
    ],
)
def test_code_fences_are_stripped(text: str) -> None:
    assert parsed(text) == ParsedOutput("t", Action("lookup", "x"))


# --------------------------------------------------------------------------- JSON format


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ('{"action": "calculator", "input": "2+3"}', ParsedOutput("", Action("calculator", "2+3"))),
        (
            '{"action": "calculator", "action_input": "1"}',
            ParsedOutput("", Action("calculator", "1")),
        ),
        (
            '{"action": "calculator", "tool_input": "1"}',
            ParsedOutput("", Action("calculator", "1")),
        ),
        ('{"action": "finish", "answer": "Paris"}', ParsedOutput("", Action("finish", "Paris"))),
        ('{"action": "Final Answer", "input": "7"}', ParsedOutput("", Action("finish", "7"))),
        ('{"action": "final_answer", "input": 7}', ParsedOutput("", Action("finish", "7"))),
        ('{"action": "FINISH", "input": null}', ParsedOutput("", Action("finish", ""))),
        ('{"action": "lookup"}', ParsedOutput("", Action("lookup", ""))),
        ('{"action": "api", "input": {"a": [1]}}', ParsedOutput("", Action("api", '{"a": [1]}'))),
        (
            '{"Thought": " t ", "ACTION": "lookup", "Input": " x "}',
            ParsedOutput("t", Action("lookup", "x")),
        ),
        (
            '{"thought": ["a"], "action": "lookup", "input": "x"}',
            ParsedOutput('["a"]', Action("lookup", "x")),
        ),
        ('{"action": "calculator[2 + 3]"}', ParsedOutput("", Action("calculator", "2 + 3"))),
        ('{"action": "finish[42]"}', ParsedOutput("", Action("finish", "42"))),
        ('{"thought": "t", "final_answer": "Paris"}', ParsedOutput("t", Action("finish", "Paris"))),
        ('{"final answer": 3.5}', ParsedOutput("", Action("finish", "3.5"))),
        (
            '{"thought": "t", "action": {"name": "lookup", "input": "x"}}',
            ParsedOutput("t", Action("lookup", "x")),
        ),
        (
            '{"action": {"Tool": "calculator", "action_input": "1"}}',
            ParsedOutput("", Action("calculator", "1")),
        ),
        (
            'Sure! Here it is: {"thought": "t", "action": "lookup", "input": "x"} Done.',
            ParsedOutput("t", Action("lookup", "x")),
        ),
    ],
)
def test_json_replies(text: str, expected: ParsedOutput) -> None:
    assert parse_output(text) == expected


# --------------------------------------------------------------------------- failures


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        ("", "the reply was empty"),
        ("   \n\t ", "the reply was empty"),
        ("```\n```", "the reply was empty"),
        ("Just some prose.", "neither an 'Action:' nor a 'Final Answer:'"),
        ("Thought: still thinking", "neither an 'Action:' nor a 'Final Answer:'"),
        ("Observation: 5", "neither an 'Action:' nor a 'Final Answer:'"),
        ("Thought: x\nActionInput: y", "neither an 'Action:' nor a 'Final Answer:'"),
        ("Thought: t\nAction: calculator", "the action 'calculator' has no input"),
        ("Thought: t\nAction: calculator\nmore text", "the action 'calculator' has no input"),
        ("Thought: t\nAction:", "the action '' has no input"),
        ("Thought: t\nAction: calc]2[", "has no input"),
        ("Thought: t\nAction: [2 + 3]", "the action has no tool name"),
        ('{"thought": "x"}', "no 'action' field"),
        ('{"action": 5, "input": "x"}', "must be a tool name string"),
        ('{"action": ["lookup"]}', "must be a tool name string"),
        ('{"action": "  "}', "must be a non-empty tool name"),
        ('{"action": "[x]"}', "must be a non-empty tool name"),
        ("{not json at all}", "neither an 'Action:' nor a 'Final Answer:'"),
        ("[1, 2, 3]", "neither an 'Action:' nor a 'Final Answer:'"),
        ('I think {"a": 1} is relevant', "neither an 'Action:' nor a 'Final Answer:'"),
        ("oops } then {", "neither an 'Action:' nor a 'Final Answer:'"),
    ],
)
def test_parse_failures_explain_what_is_wrong(text: str, reason: str) -> None:
    result = failed(text)

    assert reason in result.message
    assert result.message.endswith(FORMAT_HINT)
    assert result.raw == text


@pytest.mark.parametrize(
    "text",
    ["{", "}", "[", "]", "```", "Action", "Action:[", "Final", "\x00", "{{}}", '{"action": {}}'],
)
def test_parse_output_never_raises(text: str) -> None:
    assert isinstance(parse_output(text), (ParsedOutput, ParseFailure))


def test_format_hint_describes_both_formats() -> None:
    for fragment in ("Thought:", "Action: <tool name>[<tool input>]", "Final Answer:", '"action"'):
        assert fragment in FORMAT_HINT
