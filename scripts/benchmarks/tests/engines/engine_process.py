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
from bench import BenchRequest, BenchResponse, CooldownRequest, CooldownResponse, ReadyResponse
from pydantic import BaseModel, ValidationError


class EngineProcess:
    def __init__(self, process: subprocess.Popen[str], stderr_log: Path, timeout: float):
        self.process = process
        self.stderr_log = stderr_log
        self.timeout = timeout
        self.failed = False
        self.rejected = False
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

    def read_line(self, waiting_for: str) -> str:
        try:
            line = self.lines.get(timeout=self.timeout)
        except Empty:
            self.fail(f"No complete {waiting_for} within {self.timeout:g}s")
        if line is None:
            self.fail(f"Engine closed stdout before its {waiting_for} (exit code {self.process.poll()})")
        if isinstance(line, Exception):
            self.fail(f"Could not read {waiting_for}: {line}")
        if not line.endswith("\n"):
            self.fail(f"{waiting_for.capitalize()} is missing its JSON Lines newline: {line!r}")
        return line

    def read_response(self, response: type[BaseModel], waiting_for: str) -> None:
        line = self.read_line(waiting_for)
        try:
            response.model_validate_json(line, strict=True, extra="forbid")
        except ValidationError as error:
            self.fail(f"Invalid {response.__name__}: {error}\nReceived: {line!r}")

    def wait_ready(self) -> None:
        self.read_response(ReadyResponse, "ReadyResponse (includes build and model loading)")

    def write(self, line: str) -> None:
        if self.failed:
            pytest.fail("Engine is unavailable after an earlier protocol failure", pytrace=False)
        if self.process.poll() is not None:
            self.fail(f"Engine exited before request (exit code {self.process.returncode})")
        assert self.process.stdin is not None
        try:
            self.process.stdin.write(line + "\n")
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.fail(f"Could not send request: {error}")

    def send(self, request: BenchRequest | CooldownRequest) -> None:
        self.write(request.model_dump_json(exclude_none=True))

    def expect_failure(self, cause: str) -> None:
        """After `cause`, the engine must write nothing more and exit with a nonzero code."""
        try:
            line = self.lines.get(timeout=self.timeout)
        except Empty:
            self.fail(f"Engine kept stdout open for {self.timeout:g}s after {cause}")
        if line is not None:
            self.fail(f"Engine wrote a line after {cause}: {line!r}")
        try:
            exit_code = self.process.wait(timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self.fail(f"Engine did not exit within {self.timeout:g}s after {cause}")
        if exit_code == 0:
            self.fail(f"Engine exited with code 0 after {cause}")
        self.rejected = True

    def reject(self, line: str) -> None:
        """Sends a line the engine cannot process."""
        self.write(line)
        self.expect_failure("an invalid request")

    def cooldown(self, request: CooldownRequest) -> None:
        self.send(request)
        self.read_response(CooldownResponse, "CooldownResponse")

    def request(self, request: BenchRequest) -> BenchResponse:
        self.send(request)
        line = self.read_line("response")
        try:
            return BenchResponse.model_validate_json(line, strict=True, extra="forbid")
        except ValidationError as error:
            self.fail(f"Invalid BenchResponse: {error}\nReceived: {line!r}")


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
def run_engine(
    command: Sequence[str], cwd: Path, stderr_log: Path, timeout: float, *, ready: bool = True
) -> Generator[EngineProcess]:
    """`ready` is whether the engine is expected to accept its model.json."""
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
            if ready:
                engine.wait_ready()
            else:
                engine.expect_failure("reading a model.json it cannot serve")
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
            if not engine.failed and not engine.rejected:
                if timed_out:
                    engine.fail("Engine did not exit within 10s after stdin closed")
                if process.returncode != 0:
                    engine.fail(f"Engine exited with code {process.returncode}")
                while not engine.lines.empty():
                    if (extra := engine.lines.get_nowait()) is not None:
                        engine.fail(f"Unexpected extra output on stdout: {extra!r}")
