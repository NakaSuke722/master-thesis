import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("args,expected_calls", [
    (["--run", "--aggregate-only"], 1),
    (["--run", "4", "--aggregate-only"], 1),
    (["--run", "4"], 2),
])
def test_baseline_aggregate_only_never_invokes_inference(tmp_path, args, expected_calls):
    python_stub = tmp_path / "python_stub"
    call_log = tmp_path / "calls.jsonl"
    python_stub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['BASELINE_TEST_LOG'], 'a') as f:\n"
        "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    python_stub.chmod(0o755)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["zsh", str(root / "scripts/run_baselines.sh"), *args],
        env={**os.environ, "AMBER_PYTHON": str(python_stub), "BASELINE_TEST_LOG": str(call_log)},
        capture_output=True, text=True, check=True,
    )
    calls = [json.loads(line) for line in call_log.read_text().splitlines()]
    assert len(calls) == expected_calls
    assert calls[-1][:3] == ["src/aggregate_results.py", "--config", "configs/baselines/run.yaml"]
    assert "--require-complete" in calls[-1]
    if expected_calls == 1:
        assert all("src/runner.py" not in call for call in calls)
        assert "no inference" in result.stdout
    else:
        assert calls[0][0] == "src/runner.py"
        assert calls[0][calls[0].index("--workers") + 1] == "4"
