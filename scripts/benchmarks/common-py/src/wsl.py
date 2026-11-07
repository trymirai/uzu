import shlex
import sys


def linux_command(command: list[str], environment: str = "") -> list[str]:
    """A Linux-only engine's command. On Windows it runs inside WSL from the caller's working directory, with the
    `NAME=value` assignments in `environment`."""
    if sys.platform != "win32":
        return command
    # -e runs bash directly: through WSL's default shell the script would be parsed twice.
    return ["wsl.exe", "-e", "bash", "-lc", f"{environment} exec {shlex.join(command)}".lstrip()]


def linux_uv_command(project: str, script: str, args: list[str]) -> list[str]:
    """`uv run --project <project> <script> <args>` for a Linux-only engine project.

    Inside WSL the project environment lives on WSL's own disk: a virtualenv on the Windows drive is slow over its mount
    and would collide with Windows uv's. Model paths must be Hugging Face ids or paths WSL can open.
    """
    environment = f"UV_PROJECT_ENVIRONMENT=$HOME/.cache/uzu-benchmarks/{project}"
    return linux_command(["uv", "run", "--project", project, script, *args], environment)
