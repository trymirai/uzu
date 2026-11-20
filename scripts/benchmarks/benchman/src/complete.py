import os
import re
import shlex
import signal
import sys
from collections.abc import Callable, Iterable
from contextlib import redirect_stdout, suppress

import typer
from typer._completion_classes import FishComplete, ZshComplete

TIMEOUT = 5


# Not an Exception, which network retries would catch.
class Expired(BaseException):
    pass


# Click's CompletionItem(type="file"), in the strings Typer's callbacks are limited to.
class Files(str):
    __slots__ = ()


class OrderedZshComplete(ZshComplete):
    def complete(self) -> str:
        items = self.get_completions(*self.get_completion_args())
        replies = ["_files"] if any(isinstance(item.value, Files) for item in items) else []
        items = [item for item in items if not isinstance(item.value, Files)]
        if items:
            values = [item.value for item in items]
            width = max(map(len, values))
            shown = [f"{item.value:<{width}}  -- {item.help}" if item.help else item.value for item in items]
            rows = "-l " if shown != values else ""
            # -V keeps the order; zsh sorts the group that Typer's own reply makes.
            arrays = f"values=({shlex.join(values)}) shown=({shlex.join(shown)})"
            replies.append(f"local -a {arrays}; compadd -V benchman {rows}-d shown -a values")
        return "; ".join(replies)


class SingleCallFishComplete(FishComplete):
    def complete(self) -> str:
        items = self.get_completions(*self.get_completion_args())
        candidates = [item for item in items if not isinstance(item.value, Files)]
        reply = "\n".join(f"{item.value}\t{item.help}" if item.help else item.value for item in candidates)
        if len(candidates) < len(items):
            # The script adds the files itself, as after Typer's own is-args.
            typer.echo(reply)
            sys.exit(1)
        return reply


def completes(
    candidates: Callable[[typer.Context, str], Iterable[str]], prefix: str = "!?"
) -> Callable[[typer.Context, str], list[str]]:
    def expire(*_: object) -> None:
        raise Expired

    def complete(ctx: typer.Context, incomplete: str) -> list[str]:
        head = re.match(rf"(?:.*,)?(?:{prefix})", incomplete)[0]
        text = incomplete[len(head) :]
        found = []
        os.dup2(os.open(os.devnull, os.O_WRONLY), 2)
        signal.signal(signal.SIGALRM, expire)
        signal.setitimer(signal.ITIMER_REAL, TIMEOUT)
        with open(os.devnull, "w") as silence, redirect_stdout(silence), suppress(Exception, Expired):
            for candidate in candidates(ctx, text):
                if candidate.startswith(text):
                    found.append(type(candidate)(head + candidate))
        signal.setitimer(signal.ITIMER_REAL, 0)
        return found

    return complete
