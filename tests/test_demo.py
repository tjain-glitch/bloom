"""The presentation demo must keep working (toy keys so it runs in about a second)."""

import importlib.util
from pathlib import Path

import pytest


def test_demo_runs_and_reports_success(capsys):
    path = Path(__file__).resolve().parents[1] / "demo.py"
    spec = importlib.util.spec_from_file_location("demo", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    assert demo.main(["--toy", "--seed", "7"]) == 0
    out = capsys.readouterr().out
    assert "ACCEPT" in out and "REJECT" in out and "Fig. 8" in out
    assert "False" not in out


def _load_demo():
    path = Path(__file__).resolve().parents[1] / "demo.py"
    spec = importlib.util.spec_from_file_location("demo", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    return demo


@pytest.mark.parametrize("argv", [["--k", "3"], ["--k", "0"], ["--signer", "4096"], ["--signer", "-1"]])
def test_demo_rejects_bad_arguments_with_usage_error(argv, capsys):
    """Bad command-line input exits with a usage message (status 2), not a traceback."""
    with pytest.raises(SystemExit) as exc:
        _load_demo().main(["--toy", *argv])
    assert exc.value.code == 2
    assert "error:" in capsys.readouterr().err
