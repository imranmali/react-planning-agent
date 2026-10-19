"""Tools the agent can call: the registry, timeouts, truncation and built-in tools.

Every tool takes one string and returns one string. ``ToolRegistry.execute`` is the only
way the agent runs a tool, and it never raises: unknown tools, tool errors, unexpected
exceptions and timeouts all become ``ToolResult(is_error=True)`` observations.
"""

from __future__ import annotations

import ast
import math
import operator
import re
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TypeAlias

from react_agent.types import FINISH_ACTION, Action, ConfigError, ToolError, ToolResult

_TOOL_NAME = re.compile(r"[\w.-]+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True)
class Tool:
    """A named callable the model can invoke with a single string input."""

    name: str
    description: str
    func: Callable[[str], str]


def truncate(text: str, max_chars: int) -> str:
    """Return ``text`` cut to ``max_chars`` characters plus a marker if it was longer."""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


# --------------------------------------------------------------------------- registry


@dataclass
class _Outcome:
    """What a tool thread produced: a return value or the exception it raised."""

    value: object = None
    error: BaseException | None = None


class ToolRegistry:
    """Tools available to the agent, looked up by case-insensitive name."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    @staticmethod
    def _key(name: str) -> str:
        return name.strip().lower()

    def register(self, tool: Tool) -> None:
        """Add ``tool``. Raise ``ConfigError`` for invalid, reserved or duplicate names."""
        if not _TOOL_NAME.fullmatch(tool.name):
            raise ConfigError(
                f"Invalid tool name {tool.name!r}: use letters, digits, '_', '.' or '-'"
            )
        key = self._key(tool.name)
        if key == FINISH_ACTION:
            raise ConfigError(f"Tool name {FINISH_ACTION!r} is reserved for final answers")
        if key in self._tools:
            raise ConfigError(f"A tool named {self._tools[key].name!r} is already registered")
        self._tools[key] = tool

    def get(self, name: str) -> Tool | None:
        """Return the tool called ``name`` (case-insensitive), or ``None``."""
        return self._tools.get(self._key(name))

    def names(self) -> list[str]:
        """Registered tool names in registration order."""
        return [tool.name for tool in self._tools.values()]

    def describe(self) -> str:
        """One ``- name: description`` line per tool, for the system prompt."""
        return "\n".join(f"- {tool.name}: {tool.description}" for tool in self._tools.values())

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and self._key(name) in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def execute(self, action: Action, *, timeout: float, max_chars: int) -> ToolResult:
        """Run ``action`` with a timeout and return its (truncated) observation.

        Never raises. The tool runs in a daemon thread; if it is still running after
        ``timeout`` seconds it is abandoned (CPython cannot kill threads) and the result
        is an error. Daemon threads never block interpreter exit.
        """
        tool = self.get(action.tool)
        if tool is None:
            available = ", ".join(self.names()) or "(none)"
            message = f"Unknown tool {action.tool.strip()!r}. Available tools: {available}"
            return ToolResult(truncate(message, max_chars), is_error=True)

        outcome = _Outcome()
        func, tool_input = tool.func, action.input

        def run() -> None:
            try:
                outcome.value = func(tool_input)
            except BaseException as exc:  # reported to the model, never re-raised
                outcome.error = exc

        thread = threading.Thread(target=run, name=f"tool-{tool.name}", daemon=True)
        thread.start()
        thread.join(timeout)

        if thread.is_alive():
            message = f"Tool {tool.name!r} timed out after {timeout:g} seconds"
            return ToolResult(truncate(message, max_chars), is_error=True)
        if isinstance(outcome.error, ToolError):
            return ToolResult(truncate(str(outcome.error), max_chars), is_error=True)
        if outcome.error is not None:
            error = outcome.error
            message = f"Tool {tool.name!r} raised {type(error).__name__}: {error}"
            return ToolResult(truncate(message, max_chars), is_error=True)
        return ToolResult(truncate(str(outcome.value), max_chars))


# --------------------------------------------------------------------------- calculator

Number: TypeAlias = "int | float"

_MAX_EXPRESSION_CHARS = 1000
_MAX_EXPONENT = 1000
_MAX_RESULT_BITS = 10_000
_MAX_EXACT_FLOAT = 1e15

_BINARY_OPERATORS: dict[type[ast.operator], Callable[[Number, Number], Number]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_UNARY_OPERATORS: dict[type[ast.unaryop], Callable[[Number], Number]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_CONSTANTS: dict[str, float] = {"pi": math.pi, "e": math.e, "tau": math.tau}


def _power(base: Number, exponent: Number) -> Number:
    """``base ** exponent`` with bounds that keep evaluation fast."""
    if abs(exponent) > _MAX_EXPONENT:
        raise ToolError(f"Exponent {exponent} is too large (limit is {_MAX_EXPONENT})")
    if (
        isinstance(base, int)
        and isinstance(exponent, int)
        and abs(base) > 1
        and exponent * math.log2(abs(base)) > _MAX_RESULT_BITS
    ):
        raise ToolError(f"Result of {base} ** {exponent} would be too large")
    result = base**exponent
    if isinstance(result, complex):
        raise ToolError(f"pow({base}, {exponent}) has no real result")
    return result


_FUNCTIONS: dict[str, Callable[..., Number]] = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "pow": _power,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "floor": math.floor,
    "ceil": math.ceil,
}


def _evaluate(node: ast.AST) -> Number:
    """Evaluate an allow-listed arithmetic AST node; reject everything else."""
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ToolError(f"Unsupported literal {node.value!r}: only numbers are allowed")
    if isinstance(node, ast.BinOp):
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow):
            return _power(left, right)
        binary = _BINARY_OPERATORS.get(type(node.op))
        if binary is None:
            raise ToolError(f"Unsupported operator {type(node.op).__name__}")
        return binary(left, right)
    if isinstance(node, ast.UnaryOp):
        unary = _UNARY_OPERATORS.get(type(node.op))
        if unary is None:
            raise ToolError(f"Unsupported operator {type(node.op).__name__}")
        return unary(_evaluate(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ToolError(f"Unknown name {node.id!r}. Allowed constants: {', '.join(_CONSTANTS)}")
    if isinstance(node, ast.Call):
        return _call(node)
    raise ToolError(f"Unsupported expression element: {type(node).__name__}")


def _call(node: ast.Call) -> Number:
    """Evaluate a call to one of the allow-listed math functions."""
    if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCTIONS:
        allowed = ", ".join(sorted(_FUNCTIONS))
        raise ToolError(f"Only these functions can be called: {allowed}")
    if node.keywords:
        raise ToolError("Keyword arguments are not supported")
    args = [_evaluate(arg) for arg in node.args]  # Starred args fail in _evaluate
    try:
        return _FUNCTIONS[node.func.id](*args)
    except TypeError as exc:
        raise ToolError(f"Invalid arguments for {node.func.id}(): {exc}") from exc


def _format_number(value: Number) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ToolError("Result is not a finite number")
        if value.is_integer() and abs(value) < _MAX_EXACT_FLOAT:
            return str(int(value))
    return str(value)


def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression, e.g. ``(2 + 3) * 4 ** 2`` -> ``80``.

    Supports numbers, ``+ - * / // % **`` (``^`` is accepted as ``**``), parentheses,
    the constants ``pi``, ``e`` and ``tau`` and a fixed table of math functions. Anything
    else - names, attributes, strings, keyword arguments - raises ``ToolError``.
    """
    cleaned = expression.strip().rstrip("=?").strip().replace("^", "**")
    if not cleaned:
        raise ToolError("Empty expression")
    if len(cleaned) > _MAX_EXPRESSION_CHARS:
        raise ToolError(f"Expression is too long (limit is {_MAX_EXPRESSION_CHARS} chars)")
    try:
        tree = ast.parse(cleaned, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Invalid expression {cleaned!r}: {exc.msg}") from exc
    try:
        return _format_number(_evaluate(tree))
    except ZeroDivisionError as exc:
        raise ToolError("Division by zero") from exc
    except OverflowError as exc:
        raise ToolError("Result is too large") from exc
    except RecursionError as exc:
        raise ToolError("Expression is nested too deeply") from exc
    except ValueError as exc:  # math domain errors, int-to-str digit limit
        raise ToolError(f"Math error: {exc}") from exc


def make_calculator_tool() -> Tool:
    """The built-in ``calculator`` tool."""
    return Tool(
        name="calculator",
        description=(
            "evaluate an arithmetic expression, e.g. (2 + 3) * 4 ** 2. Supports + - * / // "
            "% **, parentheses, pi, e, tau and sqrt, log, exp, sin, cos, round, min, max ..."
        ),
        func=calculator,
    )


# --------------------------------------------------------------------------- lookup

DEFAULT_KNOWLEDGE_BASE: Mapping[str, str] = MappingProxyType(
    {
        "france population": "68170000",
        "germany population": "84480000",
        "speed of light": "299792458",
        "earth radius": "6371",
        "boiling point of water": "100",
        "eiffel tower height": "330",
        "capital of france": "Paris",
        "capital of japan": "Tokyo",
    }
)
"""Small offline facts used by demos. Units: people, m/s, km, degrees C, m."""

_KEY_PUNCTUATION = "\"'`?.!,;:"


def _normalize_key(text: str) -> str:
    """Lower-case, collapse whitespace and strip surrounding quotes/punctuation."""
    return _WHITESPACE.sub(" ", text).strip().strip(_KEY_PUNCTUATION).strip().lower()


def make_lookup_tool(knowledge_base: Mapping[str, str] | None = None) -> Tool:
    """The built-in ``lookup`` tool over ``knowledge_base`` (default: built-in facts)."""
    source = DEFAULT_KNOWLEDGE_BASE if knowledge_base is None else knowledge_base
    entries = {_normalize_key(key): value for key, value in source.items()}

    def lookup(query: str) -> str:
        key = _normalize_key(query)
        if key in entries:
            return entries[key]
        known = ", ".join(sorted(entries)) or "(none)"
        raise ToolError(f"No entry found for {key!r}. Known keys: {known}")

    return Tool(
        name="lookup",
        description="look up a fact by its exact key (case-insensitive), e.g. capital of france",
        func=lookup,
    )


def default_registry(knowledge_base: Mapping[str, str] | None = None) -> ToolRegistry:
    """A registry with the built-in ``calculator`` and ``lookup`` tools."""
    return ToolRegistry([make_calculator_tool(), make_lookup_tool(knowledge_base)])
