"""Register engine CLI options and shortcuts before test collection."""

from pathlib import Path

import pytest

from tests.engines import llamacpp, mldrift, mlx, mlxserve, mtplx, omlx, splash, uzu

ENGINE_TEST_CLASSES = {
    "llamacpp": "TestLlamaCpp",
    "mldrift": "TestMLDrift",
    "mlx": "TestMLX",
    "mlxserve": "TestMLXServe",
    "mtplx": "TestMTPLX",
    "omlx": "TestOMLX",
    "splash": "TestSplash",
    "uzu": "TestUzu",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("benchmark engines")
    group.addoption(
        "--engine",
        action="append",
        choices=list(ENGINE_TEST_CLASSES),
        default=[],
        help="Engine to test (repeatable).",
    )
    group.addoption("--engine-timeout", type=float, default=600, help="Response timeout in seconds, including startup.")
    llamacpp.add_options(parser)
    mldrift.add_options(parser)
    mlx.add_options(parser)
    mlxserve.add_options(parser)
    mtplx.add_options(parser)
    omlx.add_options(parser)
    splash.add_options(parser)
    uzu.add_options(parser)


def pytest_configure(config: pytest.Config) -> None:
    """Allow engine names or 'engines' to select and enable engine tests."""
    engines = config.getoption("engine")
    test_file = Path(__file__).with_name("test_engines.py")
    if "engines" in config.args:
        config.args[:] = [str(test_file) if arg == "engines" else arg for arg in config.args]
        engines.extend(name for name in ENGINE_TEST_CLASSES if name not in engines)

    for name, test_class in ENGINE_TEST_CLASSES.items():
        if name not in config.args:
            continue
        target = f"{test_file}::{test_class}"
        config.args[:] = [target if arg == name else arg for arg in config.args]
        if name not in engines:
            engines.append(name)
