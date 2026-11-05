"""Exercise the shipped browser handlers without loading any model."""
from pathlib import Path
import shutil
import subprocess

import pytest


def test_workbench_job_lifecycle():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is needed for the browser-handler regression harness")
    result = subprocess.run(
        [node, "--test", str(Path(__file__).with_name("web_jobs.test.cjs"))],
        capture_output=True, text=True, encoding="utf-8", timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
