"""Shared JSON Lines transport for all benchmark engine subprocesses."""

import os
import signal
import subprocess
from collections.abc import Generator, Sequence
from contextlib import contextmanager, suppress
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
from typing import NoReturn

import pytest
from bench import BenchRequest, BenchResponse
from pydantic import TypeAdapter, ValidationError

RESPONSES = TypeAdapter(list[BenchResponse])


class EngineProcess:
    def __init__(self, process: subprocess.Popen[str], stderr_log: Path, timeout: float):
        self.process = process
        self.stderr_log = stderr_log
        self.timeout = timeout
        self.failed = False
        self.lines: Queue[str | Exception | None] = Queue()

    def fail(self, message: str) -> NoReturn:
        self.failed = True
        with self.stderr_log.open("rb") as log:
            log.seek(max(0, log.seek(0, os.SEEK_END) - 8192))
            tail = log.read().decode("utf-8", errors="replace")
        pytest.fail(f"{message}\nEngine stderr: {self.stderr_log}\n{tail}", pytrace=False)

    def read_stdout(self) -> None:
        assert self.process.stdout is not None
        try:
            for line in self.process.stdout:
                self.lines.put(line)
        except (OSError, UnicodeError) as error:
            self.lines.put(error)
        finally:
            self.lines.put(None)

    def request(self, request: BenchRequest) -> list[BenchResponse]:
        if self.failed:
            pytest.fail("Engine is unavailable after an earlier protocol failure", pytrace=False)
        if self.process.poll() is not None:
            self.fail(f"Engine exited before request (exit code {self.process.returncode})")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(request.model_dump_json(exclude_none=True) + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.fail(f"Could not send request: {error}")
        try:
            line = self.lines.get(timeout=self.timeout)
        except Empty:
            self.fail(f"No complete response within {self.timeout:g}s (includes build and model loading)")
        if line is None:
            self.fail(f"Engine closed stdout before responding (exit code {self.process.poll()})")
        if isinstance(line, Exception):
            self.fail(f"Could not read response: {line}")
        if not line.endswith("\n"):
            self.fail(f"Response is missing its JSON Lines newline: {line!r}")
        try:
            return RESPONSES.validate_json(line, strict=True, extra="forbid")
        except ValidationError as error:
            self.fail(f"Invalid BenchResponse array: {error}\nReceived: {line!r}")


def kill_process_group(process: subprocess.Popen[str]) -> None:
    # run.sh can have CMake/compiler children before it execs the engine.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Some sandboxes deny signals to a group whose leader has already exited.
        if process.poll() is None:
            raise


@contextmanager
def run_engine(command: Sequence[str], cwd: Path, stderr_log: Path, timeout: float) -> Generator[EngineProcess]:
    with stderr_log.open("wb") as stderr:
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=stderr,
                text=True,
                encoding="utf-8",
                bufsize=1,
                start_new_session=True,
            )
        except OSError as error:
            pytest.fail(f"Could not start engine {command[0]}: {error}", pytrace=False)
        engine = EngineProcess(process, stderr_log, timeout)
        reader = Thread(target=engine.read_stdout, daemon=True)
        reader.start()
        try:
            yield engine
        finally:
            assert process.stdin is not None and process.stdout is not None
            with suppress(BrokenPipeError):
                process.stdin.close()
            if engine.failed:
                kill_process_group(process)
            timed_out = False
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                timed_out = True
                kill_process_group(process)
                process.wait(timeout=10)
            finally:
                kill_process_group(process)
                reader.join(timeout=5)
                process.stdout.close()
            if not engine.failed:
                if timed_out:
                    engine.fail("Engine did not exit within 10s after stdin closed")
                if process.returncode != 0:
                    engine.fail(f"Engine exited with code {process.returncode}")
                while not engine.lines.empty():
                    if (extra := engine.lines.get_nowait()) is not None:
                        engine.fail(f"Unexpected extra output on stdout: {extra!r}")
