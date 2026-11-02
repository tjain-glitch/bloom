"""The presentation demo must keep working (toy keys so it runs in about a second)."""

import importlib.util
from pathlib import Path


def test_demo_runs_and_reports_success(capsys):
    path = Path(__file__).resolve().parents[1] / "demo.py"
    spec = importlib.util.spec_from_file_location("demo", path)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    assert demo.main(["--toy", "--seed", "7"]) == 0
    out = capsys.readouterr().out
    assert "ACCEPT" in out and "REJECT" in out and "Fig. 8" in out
    assert "False" not in out
