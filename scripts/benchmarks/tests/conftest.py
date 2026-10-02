"""Register engine CLI options and shortcuts before test collection."""

from pathlib import Path

import pytest

from tests.engines import llamacpp, mlx, mtplx, omlx, splash


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("benchmark engines")
    group.addoption(
        "--engine",
        action="append",
        choices=["llamacpp", "mlx", "mtplx", "omlx", "splash"],
        default=[],
        help="Engine to test (repeatable).",
    )
    group.addoption("--engine-timeout", type=float, default=600, help="Response timeout in seconds, including startup.")
    llamacpp.add_options(parser)
    mlx.add_options(parser)
    mtplx.add_options(parser)
    omlx.add_options(parser)
    splash.add_options(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Allow engine names as shortcuts to select and enable engine tests."""
    engines = config.getoption("engine")
    for name, test_class in {
        "llamacpp": "TestLlamaCpp",
        "mlx": "TestMLX",
        "mtplx": "TestMTPLX",
        "omlx": "TestOMLX",
        "splash": "TestSplash",
    }.items():
        if name not in config.args:
            continue
        target = f"{Path(__file__).with_name('test_engines.py')}::{test_class}"
        config.args[:] = [target if arg == name else arg for arg in config.args]
        if name not in engines:
            engines.append(name)
