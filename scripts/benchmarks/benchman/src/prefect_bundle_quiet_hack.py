import subprocess

import prefect.bundles


class QuietSubprocess:
    def __getattr__(self, name: str) -> object:
        return getattr(subprocess, name)

    @staticmethod
    def check_output(command: list[str]) -> bytes:
        return subprocess.check_output([*command, "--quiet"])


def quieten_bundle_dependencies() -> None:
    # Prefect lists the packages for a bundle with `uv pip freeze`, which names the environment on our stderr.
    prefect.bundles.subprocess = QuietSubprocess()
