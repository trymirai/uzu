from __future__ import annotations

import sys
import threading
import time
from contextlib import contextmanager
from typing import Iterator

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
GREEN = "\033[32m"
CYAN = "\033[36m"
RESET = "\033[0m"
CLEAR_LINE = "\033[2K\r"


class Spinner:
    def __init__(self, message: str) -> None:
        self._message = message
        self._running = False
        self._thread: threading.Thread | None = None
        self._frame_idx = 0

    def _spin(self) -> None:
        while self._running:
            frame = SPINNER_FRAMES[self._frame_idx % len(SPINNER_FRAMES)]
            sys.stderr.write(f"{CLEAR_LINE}{CYAN}{frame}{RESET} {self._message}")
            sys.stderr.flush()
            self._frame_idx += 1
            time.sleep(0.08)

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def stop(self, success: bool = True) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.2)
        icon = f"{GREEN}✓{RESET}" if success else "✗"
        sys.stderr.write(f"{CLEAR_LINE}{icon} {self._message}\n")
        sys.stderr.flush()


@contextmanager
def status(message: str) -> Iterator[None]:
    spinner = Spinner(message)
    spinner.start()
    try:
        yield
    finally:
        spinner.stop(success=True)


def done(message: str) -> None:
    sys.stderr.write(f"{GREEN}✓{RESET} {message}\n")
    sys.stderr.flush()
