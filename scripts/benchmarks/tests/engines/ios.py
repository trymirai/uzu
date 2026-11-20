"""Launch an engine inside the iOS app through the platform-ios bridge."""

from pathlib import Path

import pytest

from .engine_tests import EngineTests

ENGINE = "uzu"
MODEL = "trymirai/LFM2.5-230M-M"


def add_options(parser: pytest.Parser) -> None:
    group = parser.getgroup("iOS")
    group.addoption("--ios-engine", default=ENGINE, help="Engine inside the app, as platform-ios run -e takes it.")
    group.addoption("--ios-model", default=MODEL, help="Model identifier as the engine expects it.")
    group.addoption("--ios-device", default=None, help="UDID of the device. Default: the one platform-ios remembers.")
    group.addoption("--ios-host", default=None, help="Reach the app at this address instead of through usbmux.")


class TestIOS(EngineTests):
    engine_name = "ios"

    @staticmethod
    def command(config: pytest.Config, model: Path) -> list[str]:
        command = ["uv", "run", "--project", "platform-ios", "platform-ios", "run"]
        command += ["--engine", config.getoption("ios_engine"), str(model)]
        for option in ("device", "host"):
            if value := config.getoption(f"ios_{option}"):
                command += [f"--{option}", value]
        return command
