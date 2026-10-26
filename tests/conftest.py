"""Optional bounded-disk test execution; never touches experiment directories."""
from pathlib import Path
import shutil
import pytest


def pytest_addoption(parser):
    parser.addoption("--prune-test-artifacts", action="store_true",
                     help="Remove function-scoped pytest temp artifacts after each test")


@pytest.fixture(autouse=True)
def bounded_temp_artifacts(request):
    target = None
    if request.config.getoption("--prune-test-artifacts") and "tmp_path" in request.fixturenames:
        target = request.getfixturevalue("tmp_path").resolve()
        base = request.getfixturevalue("tmp_path_factory").getbasetemp().resolve()
        if target == base or base not in target.parents:
            raise RuntimeError("Refusing to clean a path outside pytest's temporary directory")
    yield
    if target is not None and target.exists():
        shutil.rmtree(target)
