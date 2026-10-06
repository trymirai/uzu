import os
import signal
import subprocess
import sys
import time
from abc import ABC, abstractmethod
from contextlib import suppress
from enum import StrEnum
from typing import Any

import httpx2
from bench import BenchResponse


class ServerEngineType(StrEnum):
    UZU = "uzu"
    TENSORFOLD = "tensorfold"


class ServerEngine(ABC):
    host: str
    port: int
    model: str
    process: subprocess.Popen | None

    def __init__(self, host: str, port: int, model: str):
        self.host = host
        self.port = port
        self.model = model
        self.process = None

    @abstractmethod
    def create_process(self) -> subprocess.Popen:
        pass

    @abstractmethod
    def handle_request(self, request: dict[str, Any]) -> BenchResponse:
        pass

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("Stop the server before starting it again.")
        self.process = self.create_process()
        self.wait_ready()

    def stop(self) -> None:
        if self.process is None:
            return

        try:
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        finally:
            with suppress(ProcessLookupError):
                os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
            self.process = None

    def wait_ready(self, timeout: float | None = 600):
        deadline: float = time.monotonic() + timeout if timeout is not None else sys.float_info.max
        with httpx2.Client(trust_env=True) as client:
            while True:
                if self.process is None:
                    raise RuntimeError("Server is None")
                if self.process.poll() is not None:
                    raise RuntimeError(f"Server exited with code {self.process.returncode} before becoming ready.")

                timeout_remaining = deadline - time.monotonic()
                if timeout_remaining <= 0:
                    raise TimeoutError(f"Server did not become ready within {timeout} seconds.")

                try:
                    response = client.get(
                        f"http://{self.host}:{self.port}/v1/models",
                        timeout=min(1.0, timeout_remaining),
                    )
                    if response.is_success and self.process.poll() is None:
                        return
                except httpx2.TransportError:
                    pass

                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
