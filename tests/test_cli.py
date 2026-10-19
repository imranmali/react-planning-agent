"""Tests for the react-agent command-line interface."""

from __future__ import annotations

import io
import json
import subprocess
import sys
from types import ModuleType

import pytest

from react_agent import cli
from react_agent.llm import StubbornLLM


def run_cli(*argv: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    code = cli.main(list(argv), out=out, err=err)
    return code, out.getvalue(), err.getvalue()


def test_offline_run_prints_answer() -> None:
    code, out, err = run_cli("What is 12 * 7?")

    assert code == 0
    assert out.strip() == "84"
    assert err == ""


def test_json_output_is_a_single_document() -> None:
    code, out, _ = run_cli("--json", "What is 12 * 7?")

    document = json.loads(out)
    assert code == 0
    assert document["answer"] == "84"
    assert document["status"] == "success"


def test_trace_prints_steps_to_stderr() -> None:
    code, out, err = run_cli("--trace", "What is 12 * 7?")

    assert code == 0
    assert out.strip() == "84"
    assert "[1] Thought:" in err
    assert "Action: calculator[12 * 7]" in err
    assert "Reflection:" in err


def test_stubborn_model_is_stopped_by_loop_guard() -> None:
    code, out, _ = run_cli("--json", "--llm", "stubborn", "anything")

    document = json.loads(out)
    assert code == 2
    assert document["termination_reason"] == "loop_detected"
    assert document["status"] == "degraded"


def test_degraded_run_reports_status_on_stderr() -> None:
    code, _, err = run_cli("--llm", "stubborn", "--max-iterations", "2", "anything")

    assert code == 2
    assert "degraded: max_iterations" in err


def test_failed_run_exits_with_one(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli, "StubbornLLM", lambda: StubbornLLM("teleport[x]"))

    code, out, err = run_cli("--llm", "stubborn", "anything")

    assert code == 1
    assert out.startswith("No answer")
    assert "failed: loop_detected" in err


def test_negative_retries_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as excinfo:
        run_cli("--llm-retries", "-1", "question")

    assert excinfo.value.code == cli.EXIT_USAGE


def test_invalid_limit_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--max-iterations", "0", "question"])

    assert excinfo.value.code == cli.EXIT_USAGE
    assert "max_iterations" in capsys.readouterr().err


def test_anthropic_without_credentials_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    fake = ModuleType("anthropic")
    fake.Anthropic = object  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)

    code, out, err = run_cli("--llm", "anthropic", "What is 12 * 7?")

    assert code == 0
    assert out.strip() == "84"
    assert "warning" in err
    assert "offline planner" in err


def test_anthropic_with_client_uses_fallback_planner_on_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BrokenMessages:
        def create(self, **_: object) -> object:
            raise RuntimeError("network down")

    class BrokenClient:
        messages = BrokenMessages()

    fake = ModuleType("anthropic")
    fake.Anthropic = BrokenClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    code, out, err = run_cli("--llm", "anthropic", "--llm-retries", "0", "What is 12 * 7?")

    assert out.strip() == "84"
    assert code == 0
    assert "offline-planner" in err


def test_module_entry_point_runs() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "react_agent", "What is 12 * 7?"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert completed.stdout.strip() == "84"
