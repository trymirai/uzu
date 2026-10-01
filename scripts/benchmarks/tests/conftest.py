"""Register engine CLI options and shortcuts before test collection."""

from pathlib import Path

import pytest

from tests.engines import llamacpp


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("benchmark engines")
    group.addoption("--engine", action="append", choices=["llamacpp"], default=[], help="Engine to test (repeatable).")
    group.addoption("--engine-timeout", type=float, default=600, help="Response timeout in seconds, including startup.")
    llamacpp.add_options(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Allow `pytest llamacpp` to select and enable the existing test class."""
    if "llamacpp" not in config.args:
        return

    target = f"{Path(__file__).with_name('test_engines.py')}::TestLlamaCpp"
    config.args[:] = [target if arg == "llamacpp" else arg for arg in config.args]
    engines = config.getoption("engine")
    if "llamacpp" not in engines:
        engines.append("llamacpp")
