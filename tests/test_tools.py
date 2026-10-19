"""Tests for the tool registry, timeouts, truncation and the built-in tools."""

from __future__ import annotations

import os
import threading
import time

import pytest

from react_agent.tools import (
    DEFAULT_KNOWLEDGE_BASE,
    Tool,
    ToolRegistry,
    calculator,
    default_registry,
    make_calculator_tool,
    make_lookup_tool,
    truncate,
)
from react_agent.types import Action, ConfigError, ToolError, ToolResult


def echo_tool(name: str = "echo", description: str = "repeat the input") -> Tool:
    return Tool(name=name, description=description, func=lambda text: text)


def run(registry: ToolRegistry, tool: str, tool_input: str, max_chars: int = 2000) -> ToolResult:
    return registry.execute(Action(tool, tool_input), timeout=1.0, max_chars=max_chars)


# --------------------------------------------------------------------------- truncate


@pytest.mark.parametrize("text", ["", "short", "x" * 2000])
def test_truncate_keeps_text_within_limit(text: str) -> None:
    assert truncate(text, 2000) == text


def test_truncate_long_output_keeps_prefix_and_adds_marker() -> None:
    text = "a" * 5000 + "b" * 5000

    result = truncate(text, 2000)

    assert result.startswith("a" * 2000)
    assert result[2000:] == "... [truncated 8000 chars]"


# --------------------------------------------------------------------------- registry


def test_register_and_get_is_case_insensitive() -> None:
    tool = echo_tool("Echo")
    registry = ToolRegistry()

    registry.register(tool)

    assert registry.get("echo") is tool
    assert registry.get("  ECHO \n") is tool
    assert registry.get("missing") is None
    assert "eCHo" in registry
    assert "missing" not in registry
    assert 42 not in registry
    assert len(registry) == 1


def test_duplicate_registration_is_rejected() -> None:
    registry = ToolRegistry([make_calculator_tool()])

    with pytest.raises(ConfigError, match="already registered"):
        registry.register(make_calculator_tool())
    with pytest.raises(ConfigError, match="already registered"):
        registry.register(echo_tool("CALCULATOR"))
    assert len(registry) == 1


@pytest.mark.parametrize("name", ["finish", "Finish", "FINISH"])
def test_reserved_finish_name_is_rejected(name: str) -> None:
    with pytest.raises(ConfigError, match="reserved"):
        ToolRegistry().register(echo_tool(name))


@pytest.mark.parametrize("name", ["", "   ", "two words", " padded", "calc[1]", "a/b"])
def test_invalid_tool_names_are_rejected(name: str) -> None:
    with pytest.raises(ConfigError, match="Invalid tool name"):
        ToolRegistry().register(echo_tool(name))


def test_names_and_describe_follow_registration_order() -> None:
    registry = ToolRegistry([echo_tool("zeta", "last letter"), echo_tool("alpha", "first")])

    assert registry.names() == ["zeta", "alpha"]
    assert registry.describe() == "- zeta: last letter\n- alpha: first"


def test_empty_registry() -> None:
    registry = ToolRegistry()

    assert registry.names() == []
    assert registry.describe() == ""
    assert len(registry) == 0


# --------------------------------------------------------------------------- execute


def test_execute_returns_tool_output() -> None:
    registry = ToolRegistry([echo_tool()])

    assert run(registry, " ECHO ", "hello") == ToolResult("hello")


def test_execute_converts_non_string_output() -> None:
    registry = ToolRegistry([Tool("number", "returns an int", lambda _: 42)])  # type: ignore[arg-type,return-value]

    assert run(registry, "number", "") == ToolResult("42")


def test_unknown_tool_lists_available_tools() -> None:
    registry = ToolRegistry([echo_tool("alpha"), echo_tool("beta")])

    result = run(registry, " gamma ", "x")

    assert result.is_error
    assert result.output == "Unknown tool 'gamma'. Available tools: alpha, beta"


def test_unknown_tool_with_empty_registry() -> None:
    result = run(ToolRegistry(), "anything", "x")

    assert result == ToolResult("Unknown tool 'anything'. Available tools: (none)", is_error=True)


def test_tool_error_message_becomes_error_observation() -> None:
    def reject(_: str) -> str:
        raise ToolError("bad input, try again")

    result = run(ToolRegistry([Tool("strict", "rejects", reject)]), "strict", "x")

    assert result == ToolResult("bad input, try again", is_error=True)


def test_unexpected_exception_is_reported_not_raised() -> None:
    def crash(text: str) -> str:
        raise ValueError(f"cannot handle {text}")

    result = run(ToolRegistry([Tool("fragile", "crashes", crash)]), "fragile", "this")

    assert result == ToolResult("Tool 'fragile' raised ValueError: cannot handle this", True)


def test_base_exception_in_tool_is_contained() -> None:
    def leave(_: str) -> str:
        raise SystemExit(3)

    result = run(ToolRegistry([Tool("quitter", "exits", leave)]), "quitter", "")

    assert result == ToolResult("Tool 'quitter' raised SystemExit: 3", is_error=True)


def test_slow_tool_times_out_quickly() -> None:
    release = threading.Event()

    def slow(_: str) -> str:
        release.wait(1.0)
        return "too late"

    registry = ToolRegistry([Tool("slow", "sleeps", slow)])
    started = time.monotonic()
    try:
        result = registry.execute(Action("slow", ""), timeout=0.05, max_chars=2000)
    finally:
        release.set()
    elapsed = time.monotonic() - started

    assert result == ToolResult("Tool 'slow' timed out after 0.05 seconds", is_error=True)
    assert elapsed < 0.5


def test_registry_still_works_after_a_timeout() -> None:
    release = threading.Event()
    registry = ToolRegistry([Tool("hang", "blocks", lambda _: str(release.wait(1.0))), echo_tool()])
    try:
        assert registry.execute(Action("hang", ""), timeout=0.01, max_chars=100).is_error
        assert run(registry, "echo", "still alive") == ToolResult("still alive")
    finally:
        release.set()


def test_long_output_is_truncated() -> None:
    registry = ToolRegistry([Tool("big", "large output", lambda _: "x" * 10_000)])

    result = run(registry, "big", "", max_chars=2000)

    assert not result.is_error
    assert result.output == "x" * 2000 + "... [truncated 8000 chars]"


def test_error_messages_are_truncated() -> None:
    def shout(_: str) -> str:
        raise ToolError("e" * 500)

    result = run(ToolRegistry([Tool("loud", "long error", shout)]), "loud", "", max_chars=10)

    assert result == ToolResult("e" * 10 + "... [truncated 490 chars]", is_error=True)


# --------------------------------------------------------------------------- calculator


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("(2 + 3) * 4 ** 2", "80"),
        ("6*7", "42"),
        ("2 + 3", "5"),
        ("7 / 2", "3.5"),
        ("8 / 2", "4"),
        ("7 // 2", "3"),
        ("7 % 3", "1"),
        ("-3 + +5", "2"),
        ("2 ** -1", "0.5"),
        ("2 ^ 10", "1024"),
        ("  2 + 2 =  ", "4"),
        ("2 + 2?", "4"),
        ("0.1 + 0.2", "0.30000000000000004"),
        ("10 ** 20", "100000000000000000000"),
        ("1e20", "1e+20"),
        ("sqrt(16)", "4"),
        ("round(pi, 2)", "3.14"),
        ("min(3, 1, 2)", "1"),
        ("max(1.5, 2)", "2"),
        ("pow(2, 8)", "256"),
        ("log(e)", "1"),
        ("log10(1000)", "3"),
        ("floor(2.7) + ceil(2.1)", "5"),
        ("abs(-4)", "4"),
        ("cos(0) + sin(0)", "1"),
        ("tau / pi", "2"),
        ("84480000 - 68170000", "16310000"),
        ("299792458 / 1000", "299792.458"),
    ],
)
def test_calculator_evaluates_arithmetic(expression: str, expected: str) -> None:
    assert calculator(expression) == expected


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("__import__('os').system('ls')", "Only these functions"),
        ("x + 1", "Unknown name 'x'"),
        ("'a' * 3", "only numbers"),
        ("True + 1", "only numbers"),
        ("1j * 2", "only numbers"),
        ("[1, 2]", "List"),
        ("(lambda: 1)()", "Only these functions"),
        ("sqrt(x=4)", "Keyword arguments"),
        ("sqrt(*[4])", "Starred"),
        ("math.sqrt(4)", "Only these functions"),
        ("pi.real", "Attribute"),
        ("1 << 3", "Unsupported operator LShift"),
        ("~1", "Unsupported operator Invert"),
        ("1 < 2", "Compare"),
        ("", "Empty expression"),
        (" = ", "Empty expression"),
        ("1 +", "Invalid expression"),
        ("1 / 0", "Division by zero"),
        ("5 % 0", "Division by zero"),
        ("9 ** 9 ** 9", "too large"),
        ("2 ** 5000", "too large"),
        ("(10 ** 100) ** 200", "too large"),
        ("pow(7, 7 ** 7)", "too large"),
        ("(-8) ** (1 / 3)", "no real result"),
        ("1e308 * 10", "not a finite number"),
        ("exp(1000)", "too large"),
        ("sqrt(-1)", "Math error"),
        ("sqrt(1, 2)", "Invalid arguments for sqrt"),
        ("1+" * 600 + "1", "too long"),
        ("-" * 998 + "1", "nested too deeply"),
    ],
)
def test_calculator_rejects_invalid_input(expression: str, message: str) -> None:
    with pytest.raises(ToolError, match=message):
        calculator(expression)


def test_calculator_never_executes_injected_code(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_system(command: str) -> int:
        calls.append(command)
        return 0

    monkeypatch.setattr(os, "system", fake_system)

    with pytest.raises(ToolError):
        calculator("__import__('os').system('ls')")
    assert calls == []


def test_huge_exponent_fails_fast() -> None:
    started = time.monotonic()
    with pytest.raises(ToolError):
        calculator("9 ** 9 ** 9")
    assert time.monotonic() - started < 0.5


def test_calculator_tool_runs_through_registry() -> None:
    registry = ToolRegistry([make_calculator_tool()])

    assert run(registry, "calculator", "(2 + 3) * 4 ** 2") == ToolResult("80")
    assert run(registry, "calculator", "1/0") == ToolResult("Division by zero", is_error=True)


# --------------------------------------------------------------------------- lookup


def test_default_knowledge_base_contents() -> None:
    assert dict(DEFAULT_KNOWLEDGE_BASE) == {
        "france population": "68170000",
        "germany population": "84480000",
        "speed of light": "299792458",
        "earth radius": "6371",
        "boiling point of water": "100",
        "eiffel tower height": "330",
        "capital of france": "Paris",
        "capital of japan": "Tokyo",
    }


@pytest.mark.parametrize("key", sorted(DEFAULT_KNOWLEDGE_BASE))
def test_lookup_finds_every_default_entry(key: str) -> None:
    assert make_lookup_tool().func(key) == DEFAULT_KNOWLEDGE_BASE[key]


@pytest.mark.parametrize(
    "query",
    ["France Population", "  france   POPULATION ", "'france population'", '"France population?"'],
)
def test_lookup_normalises_the_query(query: str) -> None:
    assert make_lookup_tool().func(query) == "68170000"


def test_lookup_with_custom_knowledge_base_is_case_insensitive() -> None:
    lookup = make_lookup_tool({"france": "France: capital Paris", "Big  Ben": "96 m"})

    assert lookup.func("France") == "France: capital Paris"
    assert lookup.func("big ben") == "96 m"


def test_lookup_unknown_key_names_the_key() -> None:
    lookup = make_lookup_tool({"b": "2", "a": "1"})

    with pytest.raises(ToolError, match=r"No entry found for 'atlantis'\. Known keys: a, b"):
        lookup.func("Atlantis?")


def test_lookup_with_empty_knowledge_base() -> None:
    with pytest.raises(ToolError, match=r"Known keys: \(none\)"):
        make_lookup_tool({}).func("anything")


# --------------------------------------------------------------------------- defaults


def test_default_registry_has_calculator_and_lookup() -> None:
    registry = default_registry()

    assert registry.names() == ["calculator", "lookup"]
    assert registry.describe().startswith("- calculator: ")
    assert "\n- lookup: " in registry.describe()
    assert run(registry, "lookup", "Capital of Japan") == ToolResult("Tokyo")
    assert run(registry, "calculator", "6*7") == ToolResult("42")


def test_default_registry_accepts_custom_knowledge_base() -> None:
    registry = default_registry({"answer": "42"})

    assert run(registry, "lookup", "ANSWER") == ToolResult("42")
    assert run(registry, "lookup", "capital of france").is_error
