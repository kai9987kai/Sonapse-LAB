"""Build metadata must describe the runtime without importing its heavy modules."""
import ast
from email.parser import Parser
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def built_metadata(tmp_path_factory):
    project = tmp_path_factory.mktemp("metadata-project")
    shutil.copyfile(ROOT / "pyproject.toml", project / "pyproject.toml")
    package = project / "poseidon"
    package.mkdir()
    source = (ROOT / "poseidon/__init__.py").read_text(encoding="utf-8")
    # An AST-readable version must not execute package imports during builds.
    (package / "__init__.py").write_text(
        source + '\nraise RuntimeError("metadata build imported poseidon")\n',
        encoding="utf-8",
    )
    metadata = project / "metadata"
    metadata.mkdir()
    command = (
        "from setuptools.build_meta import prepare_metadata_for_build_wheel; "
        "prepare_metadata_for_build_wheel('metadata')"
    )
    result = subprocess.run([sys.executable, "-c", command], cwd=project,
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    return Parser().parsestr(next(metadata.glob("*.dist-info/METADATA")).read_text(encoding="utf-8"))


def test_distribution_version_matches_runtime_without_importing_package(built_metadata):
    tree = ast.parse((ROOT / "poseidon/__init__.py").read_text(encoding="utf-8"))
    version = next(ast.literal_eval(node.value) for node in tree.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "__version__"
                           for target in node.targets))
    assert built_metadata["Version"] == version


def test_distribution_declares_eager_scientific_dependency(built_metadata):
    requirements = built_metadata.get_all("Requires-Dist", [])
    assert any(item.startswith("scipy") for item in requirements), requirements
