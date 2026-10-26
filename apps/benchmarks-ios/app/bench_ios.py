import importlib
import io
import os
import sys
import traceback
from contextlib import redirect_stderr, redirect_stdout


class Stream(io.TextIOBase):
    def __init__(self, fd: int, transcript: io.StringIO):
        self.fd = fd
        self.transcript = transcript

    def write(self, text: str) -> int:
        os.write(self.fd, text.encode())
        return self.transcript.write(text)


def run(engine: str, model: str, requests: str) -> str:
    transcript = io.StringIO()
    stdin, sys.stdin = sys.stdin, io.StringIO(requests)
    try:
        with redirect_stdout(Stream(1, transcript)), redirect_stderr(Stream(2, transcript)):
            importlib.import_module(f"bench_{engine}").run(model)
    except Exception:  # noqa: BLE001
        traceback.print_exc(file=Stream(2, transcript))
    finally:
        sys.stdin = stdin
    return transcript.getvalue()
