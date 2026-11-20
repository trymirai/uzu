import importlib
import sys
import traceback
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def standard_streams() -> Iterator[None]:
    streams = sys.stdin, sys.stdout, sys.stderr
    try:
        with (
            open(0, closefd=False) as sys.stdin,
            open(1, "w", buffering=1, closefd=False) as sys.stdout,
            open(2, "w", buffering=1, closefd=False) as sys.stderr,
        ):
            yield
    finally:
        sys.stdin, sys.stdout, sys.stderr = streams


def main(module: str, *arguments: str) -> int:
    sys.argv = [module, *arguments]
    with standard_streams():
        try:
            importlib.import_module(module).main()
        except SystemExit as exit:
            return exit.code if isinstance(exit.code, int) else int(exit.code is not None)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return 1
        return 0


def model_paths(*arguments: str) -> list[str]:
    with standard_streams():
        try:
            from common import get_model_arguments

            return get_model_arguments(arguments[0])
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            return []
