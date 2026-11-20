import asyncio
import json
import os
import re
import shutil
import signal
import string
import subprocess
import sys
import termios
import tty
from collections.abc import Callable, Coroutine, Iterable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from fnmatch import fnmatchcase
from functools import cache, partial
from itertools import product
from pathlib import Path
from typing import Annotated, Any, Literal, NamedTuple

import typer
from benchman.complete import Files, OrderedZshComplete, SingleCallFishComplete, completes
from benchman.datasets import LOADERS, load_dataset
from benchman.device import Hardware, read_hardware
from benchman.local import run_local
from benchman.report import RUNS, Progress, Run, console, progress, report
from benchman.source import pack
from benchman.task import TaskRequest
from benchman.version import VERSION
from typer._click.core import ParameterSource
from typer._click.exceptions import BadOptionUsage, UsageError
from typer._click.shell_completion import CompletionItem, add_completion_class
from typer.core import TyperCommand

app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
add_completion_class(OrderedZshComplete, "zsh")
add_completion_class(SingleCallFishComplete, "fish")


class RepeatableOptionsCommand(TyperCommand):
    """A repeatable option takes every value up to the next option: `-m a b` is `-m a -m b`."""

    def parse_args(self, ctx: typer.Context, args: list[str]) -> list[str]:
        selections: list[list[str]] = [[]]
        for arg in args:
            if arg == "+":
                selections.append([])
            else:
                selections[-1].append(arg)
        if ctx.resilient_parsing:
            # A completion is of the selection that is being typed.
            selections = selections[-1:]
        elif len(selections) > 1 and not all(selections):
            raise UsageError("A + is between two selections", ctx)
        args, *others = selections
        ctx.meta["selections"] = [self.selection(ctx, other) for other in others]
        repeatable = {flag for parameter in self.params if parameter.multiple for flag in parameter.opts}
        expanded = []
        flag = None
        valued = True
        for arg in [*args, None]:
            if arg is None or arg.startswith("-"):
                # In a completion, the values are yet to be typed.
                if not valued and not ctx.resilient_parsing:
                    raise BadOptionUsage(flag, f"Option '{flag}' requires an argument.", ctx)
                if arg is None:
                    break
                flag = arg if arg in repeatable else None
                valued = flag is None
                if flag is None:
                    expanded.append(arg)
            elif flag is not None:
                expanded += [flag, arg]
                valued = True
            else:
                expanded.append(arg)
        ctx.meta["repeating"] = flag
        return super().parse_args(ctx, expanded)

    def selection(self, ctx: typer.Context, args: list[str]) -> "Selection":
        """A selection after a +, which has only what a selection is of."""
        parsed = self.make_context(ctx.info_name, args, parent=ctx.parent)
        for parameter in self.params:
            given = parsed.get_parameter_source(parameter.name) is ParameterSource.COMMANDLINE
            if given and parameter.name not in Selection._fields:
                raise UsageError(f"{parameter.opts[-1]} is of the whole command, and goes before the first +", ctx)
        return Selection(**{field: parsed.params[field] for field in Selection._fields})

    def shell_complete(self, ctx: typer.Context, incomplete: str) -> list[CompletionItem]:
        flag = ctx.meta["repeating"]
        if flag is None or incomplete.startswith("-"):
            # Click lists options only after a dash.
            options = super().shell_complete(ctx, incomplete or "--")
            names = [[name for name in parameter.opts if name.startswith("--")] for parameter in self.params]
            singulars = {name for long in names for name in long[:-1]}
            return [option for option in options if option.value not in singulars]
        return next(parameter for parameter in self.params if flag in parameter.opts).shell_complete(ctx, incomplete)


def expand(value: str) -> list[str]:
    """Shell-style braces: `a{b,c}` is `ab` and `ac`. Not nested."""
    braces = re.search(r"\{([^{}]*,[^{}]*)\}", value)
    if braces is None:
        return [value]
    return [
        item
        for alternative in braces[1].split(",")
        for item in expand(value[: braces.start()] + alternative + value[braces.end() :])
    ]


def split(values: list[str]) -> list[str]:
    """Splits values at commas outside braces and expands the braces."""
    items = [item for value in values for item in re.split(r",(?![^{]*\})", value)]
    return [expanded for item in items for expanded in expand(item)]


def select(values: list[str], match: Callable[[str], list[str]]) -> tuple[list[str], list[str]]:
    """What the values match without what the values after `!` match, and the `!` values that took nothing out."""
    matched = list(dict.fromkeys(item for value in values if not value.startswith("!") for item in match(value)))
    excluded = {value: set(match(value[1:])) & set(matched) for value in values if value.startswith("!")}
    kept = [item for item in matched if not any(item in items for items in excluded.values())]
    return kept, [value for value, items in excluded.items() if not items]


def named(values: list[str], match: Callable[[str], list[str]]) -> list[str]:
    """What the values without a pattern or `!` are: these have to work, where a pattern's matches may not."""
    return [item for value in values if not value.startswith("!") and not GLOB.search(value) for item in match(value)]


LOCAL = ("local-nix", "local-direct")
GLOB = re.compile(r"[*?\[]")
GIB = 2**30
# A stupid proxy for the state an engine holds besides the weights: a share of the model's size.
STATE_SHARE = 0.2
# Magnitude's startup peak was a third over the size of Qwen3.5-4B.
MAGNITUDE_STATE_SHARE = 1 / 3
# TensorFold holds the caches of a model's whole context window, 6.5 GiB with Qwen3.5-2B.
TENSORFOLD_STATE = 6.5 * GIB
# What the system keeps, for engines with no limit of their own.
SYSTEM_MEMORY = 4 * GIB
# Chips an engine refuses to start on, as patterns of the fleet's chip tags.
UNSUPPORTED_CHIPS = {
    # Splash needs Apple GPU family 9, an M3 or newer.
    "splash": ("m1", "m1-*", "m2", "m2-*"),
}


@cache
def queues(work_pool: str) -> dict[str, Hardware]:
    """The work pool's queues that online workers serve."""
    from benchman import fleet

    return fleet.queues(work_pool)


@cache
def hardware(host: str) -> Hardware:
    if host in LOCAL:
        return asyncio.run(read_hardware())
    work_pool, _, queue = host.partition("/")
    if queue not in queues(work_pool):
        raise typer.BadParameter(f"No online worker serves {host}", param_hint="--host")
    return queues(work_pool)[queue]


def match_hosts(host: str) -> list[str]:
    """The host, or the queues with an online worker that a `POOL/PATTERN` matches."""
    if host == "local":
        return ["local-nix" if shutil.which("nix") else "local-direct"]
    work_pool, _, queue = host.partition("/")
    if host not in LOCAL and not (work_pool and queue):
        raise typer.BadParameter("Expected local, local-nix, local-direct or POOL/QUEUE", param_hint="--host")
    if not GLOB.search(queue):
        return [host]
    return [f"{work_pool}/{name}" for name in sorted(queues(work_pool)) if fnmatchcase(name, queue)]


def match_engines(engine: str) -> list[str]:
    """The engine, or the engines of the checkout that a pattern matches."""
    if not GLOB.search(engine):
        return [engine]
    catalogs = sorted((root() / "scripts/benchmarks").glob("engine-*/models.json"))
    return [name for catalog in catalogs if fnmatchcase(name := catalog.parent.name.removeprefix("engine-"), engine)]


def match_models(engine: str, model: str) -> list[str]:
    """The model, or the models of the engine's catalog that a pattern matches."""
    model = model if "/" in model else f"trymirai/{model}"
    if not GLOB.search(model):
        return [model]
    return [name for name in catalog(engine) if fnmatchcase(name, model)]


def unsupported_chips(host: str, engine: str) -> list[str]:
    """The chips of the host that the engine doesn't run on."""
    patterns = UNSUPPORTED_CHIPS.get(engine, ())
    chips = hardware(host).chips if patterns else ()
    return sorted(chip for chip in chips if any(fnmatchcase(chip, pattern) for pattern in patterns))


def working_set(memory: int) -> float:
    """Assumed: Metal's recommended working set on a Mac with this much memory."""
    return memory * 2 / 3 if memory <= 24 * GIB else memory - min(memory / 4, 16 * GIB)


def fits(host: str, engine: str, config: dict[str, Any]) -> bool:
    """Whether the engine takes the model with the host's memory, by the engine's own limit where it has one."""
    if "model_size" not in config:
        message = f"engine-{engine}/models.json has no model_size of {config['model']}"
        raise typer.BadParameter(message, param_hint="--filter-ram")
    size = config["model_size"]
    memory = hardware(host).memory
    if engine == "magnitude":
        # As its MemoryReserves::standard: the working set less a planning reserve.
        limit = working_set(memory)
        return size * (1 + MAGNITUDE_STATE_SHARE) <= limit - max(limit / 10, 2 * GIB)
    if engine == "tensorfold":
        # As its metal_admit::room: 70% of the memory, at most the working set.
        return size + TENSORFOLD_STATE <= min(0.7 * memory, working_set(memory))
    return size * (1 + STATE_SHARE) <= memory - SYSTEM_MEMORY


@cache
def root() -> Path:
    git = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True, text=True)
    return Path(git.stdout.strip())


@cache
def checkout() -> tuple[str, bytes]:
    """The archive of the working directory's checkout, after its hash."""
    with console.status("Archiving the checkout"):
        return pack(root())


BASE62 = string.digits + string.ascii_uppercase + string.ascii_lowercase
# A source is named by 8 bytes of its hash, which take 11 characters in base 62.
NAME_BYTES, NAME_LENGTH = 8, 11


def source_name(source: str) -> str:
    """A short name of the source: the start of its hash, in base 62 so that it is not taken for a commit's."""
    number = int(source[: 2 * NAME_BYTES], 16)
    name = ""
    for _ in range(NAME_LENGTH):
        number, digit = divmod(number, 62)
        name = BASE62[digit] + name
    return name


def stored_source(name: str, hosts: list[str]) -> str:
    """The source with this name, in the storage of the hosts' work pools."""
    if any(host in LOCAL for host in hosts):
        raise typer.BadParameter("Local hosts run this checkout", param_hint="--source")
    if len(name) != NAME_LENGTH or not set(name) <= set(BASE62):
        raise typer.BadParameter("Invalid source hash", param_hint="--source")
    from benchman import flow

    number = 0
    for character in name:
        number = number * 62 + BASE62.index(character)
    start = f"{number:0{2 * NAME_BYTES}x}"
    work_pools = {host.partition("/")[0] for host in hosts}
    found = {source for work_pool in work_pools for source in flow.pool_storage(work_pool).sources(start)}
    if len(found) != 1:
        raise typer.BadParameter(f"{len(found)} stored sources are named {name}", param_hint="--source")
    return found.pop()


def catalog(engine: str) -> dict[str, dict[str, Any]]:
    """The engine's catalog: each model's model.json."""
    try:
        return json.loads((root() / "scripts/benchmarks" / f"engine-{engine}" / "models.json").read_text())
    except FileNotFoundError:
        raise typer.BadParameter(f"There is no engine-{engine}", param_hint="--engine") from None


# For tasks without their own limit; the value of the previous release benchmarks.
MAX_OUTPUT_TOKENS = 4096


@cache
def dataset(name: str) -> dict[str, TaskRequest]:
    """The dataset's tasks by prompt."""
    with console.status(f"Loading {name}"):
        requests = asyncio.run(load_dataset(name))
    return {
        key: TaskRequest(**{"max_output_tokens": MAX_OUTPUT_TOKENS} | request.model_dump(exclude_none=True))
        for key, request in requests.items()
    }


def match_tasks(pattern: str) -> dict[str, TaskRequest]:
    """The tasks that `NAME[/KEY]` names or matches, keyed `NAME/KEY`."""
    name, _, key = pattern.partition("/")
    named = not GLOB.search(name)
    if named and name not in LOADERS:
        raise typer.BadParameter(f"No dataset {name}; there are {', '.join(LOADERS)}", param_hint="--dataset")
    tasks = {}
    for loaded in [name] if named else [loader for loader in LOADERS if fnmatchcase(loader, name)]:
        prompts = dataset(loaded)
        if named and key and not GLOB.search(key) and key not in prompts:
            raise typer.BadParameter(f"{name} has no prompt {key}", param_hint="--dataset")
        tasks |= {f"{loaded}/{prompt}": task for prompt, task in prompts.items() if not key or fnmatchcase(prompt, key)}
    return tasks


DEFAULT_DATASETS = ("spec:mirai-mixture-fast", "!spec:oneoff/london-summary")
SELECTION = re.compile(r"(?P<only>!?spec:)?(?P<exclude>!)?(?P<tasks>.+?)(?:\^(?P<repeats>[1-9]\d*))?")


def check_datasets(datasets: Sequence[str]) -> None:
    """Fails for a selection after `!` that leaves nothing out."""
    selections = [SELECTION.fullmatch(selection).groupdict() for selection in datasets]
    included = {
        name for selection in selections if not selection["exclude"] for name in match_tasks(selection["tasks"])
    }
    for text, selection in zip(datasets, selections, strict=True):
        if selection["exclude"] and (selection["repeats"] or included.isdisjoint(match_tasks(selection["tasks"]))):
            raise typer.BadParameter(f"{text} leaves no prompt out", param_hint="--dataset")


def select_tasks(datasets: Sequence[str], specdec: bool) -> dict[str, TaskRequest]:
    """The tasks for a model that decodes speculatively, or for one that doesn't."""
    included: dict[str, TaskRequest] = {}
    excluded: set[str] = set()
    for selection in datasets:
        only, exclude, pattern, repeats = SELECTION.fullmatch(selection).groups()
        if only is not None and (only == "spec:") != specdec:
            continue
        if exclude:
            excluded |= match_tasks(pattern).keys()
        else:
            tasks = match_tasks(pattern).items()
            included |= {name: task.model_copy(update={"repeats": int(repeats or 1)}) for name, task in tasks}
    return {name: task for name, task in included.items() if name not in excluded}


class Selection(NamedTuple):
    """What one selection of a command is of; a command has several, separated by `+`."""

    hosts: list[str]
    engines: list[str]
    models: list[str] | None
    datasets: list[str]
    seed: int | None
    filter_ram: bool


def match_selection(selection: Selection) -> list[tuple[str, str, str]]:
    """The hosts, engines and models of the selection that go together."""
    if not selection.models:
        raise typer.BadParameter("Every selection needs models", param_hint="--model")
    hosts, idle = select(split(selection.hosts), match_hosts)
    if idle:
        raise typer.BadParameter(f"{', '.join(idle)} leaves no host out", param_hint="--host")
    engines, idle = select(split(selection.engines), match_engines)
    if idle:
        raise typer.BadParameter(f"{', '.join(idle)} leaves no engine out", param_hint="--engine")
    models = {engine: select(split(selection.models), partial(match_models, engine)) for engine in engines}
    idle = [value for value in split(selection.models) if all(value in unused for _, unused in models.values())]
    if idle:
        raise typer.BadParameter(f"{', '.join(idle)} leaves no model out", param_hint="--model")
    check_datasets(split(selection.datasets))
    named_engines = named(split(selection.engines), match_engines)
    for host, engine in product(named(split(selection.hosts), match_hosts), named_engines):
        if chips := unsupported_chips(host, engine):
            raise typer.BadParameter(f"{engine} does not run on the {', '.join(chips)} of {host}", param_hint="--host")
    return [
        (host, engine, model)
        for host, (engine, (engine_models, _)) in product(hosts, models.items())
        if not unsupported_chips(host, engine)
        for model in engine_models
        if engine in named_engines or model in catalog(engine)
    ]


def plan(source: str, host: str, engine: str, model: str, datasets: Sequence[str], seed: int | None, label: str) -> Run:
    """A run to measure: the model's tasks of the datasets, each `[spec:|!spec:][!]NAME[/KEY][^N]`."""
    config = {"model": model} | catalog(engine).get(model, {})
    if "specdec" in config:
        tasks = select_tasks(datasets, config["specdec"])
    else:
        speculative, plain = select_tasks(datasets, True), select_tasks(datasets, False)
        either = speculative.keys() & plain.keys()
        if differing := [name for name in either if speculative[name] != plain[name]]:
            message = (
                f"{differing[0]} repeats differently with and without speculative decoding, which {model} leaves open"
            )
            raise typer.BadParameter(message, param_hint="--dataset")
        tasks = {
            name: task.model_copy(update={"specdec": None if name in either else name in speculative})
            for name, task in (speculative | plain).items()
        }
    return Run(
        source=source,
        host=host,
        engine=engine,
        model=model,
        config=config,
        tasks={name: task.model_copy(update={"seed": seed}) for name, task in tasks.items()},
        label=label,
    )


def complete_pools(text: str, names: Callable[[str], Iterable[str]]) -> Iterator[str]:
    from benchman import fleet

    work_pool, slash, _ = text.partition("/")
    work_pools = [work_pool] if slash else [name for name in fleet.work_pools() if name.startswith(text)]
    yield from (f"{name}/" for name in work_pools)
    if len(work_pools) == 1:
        yield from (f"{work_pools[0]}/{name}" for name in names(work_pools[0]))


def complete_hosts(_: typer.Context, text: str) -> Iterator[str]:
    yield from ("local", *LOCAL)
    yield from complete_pools(text, lambda work_pool: sorted(queues(work_pool)))


def complete_models(ctx: typer.Context, text: str) -> Iterator[str]:
    engines, _ = select(split(ctx.params.get("engines") or ["uzu"]), match_engines)
    for name in dict.fromkeys(name for engine in engines for name in catalog(engine)):
        yield name if text.startswith("trymirai/") else name.removeprefix("trymirai/")


def complete_datasets(_: typer.Context, text: str) -> Iterator[str]:
    yield from LOADERS
    if text.startswith("oneoff/"):
        yield from (f"oneoff/{prompt}" for prompt in asyncio.run(load_dataset("oneoff")))


def complete_tags(_: typer.Context, text: str) -> Iterator[str]:
    if not text.startswith("tag:"):
        yield Files(text)
        if "tag:".startswith(text):
            from benchman import fleet

            if fleet.configured():
                yield "tag:"
        return
    from benchman import flow

    tags = complete_pools(text.removeprefix("tag:"), lambda work_pool: flow.pool_storage(work_pool).tags())
    yield from (f"tag:{name}" for name in tags)


def show_completion(shell: Literal["zsh", "fish"] | None) -> None:
    if shell is not None:
        typer.echo(Path(__file__).with_name(f"completion.{shell}").read_text(), nl=False)
        raise typer.Exit


def tag(target: str) -> tuple[str, str] | None:
    """The work pool and name of a `tag:POOL/NAME`, or None for a file."""
    if not target.startswith("tag:"):
        return None
    work_pool, _, name = target.removeprefix("tag:").partition("/")
    if not work_pool or not name or "/" in name:
        raise typer.BadParameter(f"Expected tag:POOL/NAME, not {target}")
    return work_pool, name


def name_of(target: str) -> str:
    """The name that a file or `tag:POOL/NAME` gives its runs."""
    stored = tag(target)
    return stored[1] if stored else Path(target).stem


def load(source: str) -> list[Run]:
    """The runs saved in a file or `tag:POOL/NAME`, labelled with its name or with `@LABEL`."""
    target, _, label = source.rpartition("@") if "@" in source else (source, "", "")
    if stored := tag(target):
        # Prefect is imported only for the fleet.
        from benchman import flow

        try:
            content = flow.pool_storage(stored[0]).read_tag(stored[1])
        except LookupError:
            raise typer.BadParameter(f"There is no {target}", param_hint="--input") from None
    else:
        content = Path(target).read_bytes()
    saved = json.loads(content)
    if saved["version"] != VERSION:
        message = f"{target} is of benchman version {saved['version']}, and this is version {VERSION}"
        raise typer.BadParameter(message, param_hint="--input")
    runs = RUNS.validate_python(saved["runs"])
    if stored:
        flow.read_tagged(stored[1], runs)
    for run in runs:
        run.label, run.state, run.completed = label or name_of(target), "Saved", True
        if run.failures:
            run.state, run.completed, run.message = "Failed", False, run.failure()
        elif len(run.responses) < len(run.tasks):
            run.state, run.completed = "Missing", None
    return runs


def save(runs: list[Run]) -> bytes:
    return json.dumps({"version": VERSION, "runs": RUNS.dump_python(runs, mode="json")}, indent=2).encode()


@contextmanager
def typed_keys(stdin: int) -> Iterator[None]:
    """Has the terminal give keys as they are typed, unechoed, and keep its output when Ctrl-C is typed."""
    saved = tty.setcbreak(stdin)
    mode = termios.tcgetattr(stdin)
    mode[3] |= termios.NOFLSH
    termios.tcsetattr(stdin, termios.TCSANOW, mode)
    try:
        yield
    finally:
        # Also drops what was typed meanwhile.
        termios.tcsetattr(stdin, termios.TCSAFLUSH, saved)


async def until_detached(watching: Coroutine[None, None, None]) -> bool:
    """Awaits it until done or Ctrl-] is typed, and returns whether it was. The terminal has to give typed keys."""
    stdin = sys.stdin.fileno()
    task = asyncio.ensure_future(watching)
    detached = False

    def key() -> None:
        nonlocal detached
        if os.read(stdin, 1) == b"\x1d":
            detached = True
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            task.cancel()

    asyncio.get_running_loop().add_reader(stdin, key)
    try:
        await task
    except asyncio.CancelledError:
        if not detached:
            raise
    finally:
        asyncio.get_running_loop().remove_reader(stdin)
    return detached


def measure(
    runs: list[Run],
    how: Literal["run", "rerun", "wait", "no-run", "cancel"],
    background: bool,
    tags: Sequence[tuple[str, str]],
    archive: bytes | None,
) -> bool:
    """Measures the runs, showing progress. The archive is of their source, which only stored runs go without.

    On the fleet, `run` starts what is neither stored nor being measured and waits, `rerun` measures everything
    again, `wait` and `no-run` start nothing and `no-run` doesn't wait, `cancel` gives up what is being measured.
    Local runs are always measured and stored nowhere.

    A tag, a work pool and a name, becomes one of these runs: it gets a copy of each outcome, also of those
    that come later.

    Returns whether it left the runs to be measured: with `background` once the flow runs are submitted, or on
    Ctrl-] (detach). Exits the process on Ctrl-C.
    """
    local = [run for run in runs if run.host in LOCAL]
    if local and (background or how in ("wait", "no-run", "cancel")):
        raise typer.BadParameter("Local hosts run in this process and store no results", param_hint="--host")
    for work_pool, _ in tags:
        if any(run.work_pool != work_pool for run in runs):
            raise typer.BadParameter(f"A tag of {work_pool} is of runs on its hosts", param_hint="--output")

    dispatch = None
    detached = background
    watching = False

    def started() -> tuple[set[Run], set[Run]]:
        """The runs with tasks that flow runs of this invocation are measuring, and those with tasks of others'."""
        flow_runs = list(dispatch.measuring.items())
        return (
            {run for flow_run_id, run in flow_runs if flow_run_id in dispatch.started},
            {run for flow_run_id, run in flow_runs if flow_run_id not in dispatch.started},
        )

    def keys() -> str:
        here, elsewhere = started()
        if not here:
            return "Ctrl-] / Ctrl-C detaches"
        if not elsewhere:
            return "Ctrl-] detaches, Ctrl-C cancels"
        cancels = f"Ctrl-C cancels {len(here)} started here, detaches from {len(elsewhere)} started elsewhere"
        return f"Ctrl-] detaches, {cancels}"

    def leave(how: str, left: list[Run]) -> None:
        """Leaves the progress of these runs on the screen, and says how to reach them."""
        console.print(Progress(left, None))
        # Without an archive, the source was given by its name.
        option = f" To reach these runs from a changed or another checkout, add --source {source_name(runs[0].source)}."
        console.print(f"{how}.{option if archive else ''}", highlight=False)

    # Only fleet runs can be detached, and only from a foreground terminal.
    stdin = sys.stdin.fileno()
    detachable = not local and sys.stdin.isatty() and os.tcgetpgrp(stdin) == os.getpgrp()
    with typed_keys(stdin) if detachable else nullcontext():
        try:
            if len(local) < len(runs):
                from benchman import flow

                dispatch = flow.Dispatch([run for run in runs if run.host not in LOCAL], [name for _, name in tags])
                for work_pool, name in tags:
                    flow.pool_storage(work_pool).write_tag(name, save(runs), dispatch.records.values())
                with console.status("Reading results"):
                    unclaimed = dispatch.inspect(rerun=how == "rerun")
                if how == "run":
                    # What this would have measured, had it not been stored.
                    for run in runs:
                        if run.completed:
                            run.state = "Cached"
                while unclaimed and how in ("run", "rerun"):
                    with console.status("Creating flow runs"):
                        contested = dispatch.submit(archive, unclaimed)
                    with console.status("Reading results"):
                        unclaimed = dispatch.inspect(contested)
                if how == "cancel":
                    dispatch.cancel()
                    dispatch.inspect()
                if tags:
                    dispatch.tag()
            watching = dispatch is not None and how != "no-run" and bool(dispatch.measuring)

            async def execute() -> None:
                # Ctrl-C works once, as asyncio has it: typed again, it would interrupt the leaving at any point.
                interrupt = signal.getsignal(signal.SIGINT)

                def once(*arguments: object) -> None:
                    signal.signal(signal.SIGINT, signal.SIG_IGN)
                    interrupt(*arguments)

                signal.signal(signal.SIGINT, once)
                async with asyncio.TaskGroup() as group:
                    if watching:
                        group.create_task(dispatch.watch())
                    for run in local:
                        await run_local(run, root(), archive)

            if not detached and (watching or local):
                # The missing ones first, as nothing tells when they went missing, then the others as they ended.
                for run in runs:
                    if run.missing or run.state == "Missing":
                        run.announce(missing=True)
                for run in sorted((run for run in runs if run.completed is not None), key=lambda run: run.ended):
                    run.announce()
                if detachable:
                    with progress(runs, keys):
                        detached = asyncio.run(until_detached(execute()))
                else:
                    with progress(runs):
                        asyncio.run(execute())
        except KeyboardInterrupt:
            # As in the waiting, for a Ctrl-C before it.
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            # TODO: decide whether Ctrl-C should detach instead, as in tools that watch remote jobs.
            # Only what this invocation started; others may be waiting for the rest.
            if dispatch is not None:
                dispatch.cancel(started=True)
                _, elsewhere = started()
                if watching and elsewhere:
                    leave("Detached", [run for run in runs if run in elsewhere])
            sys.exit(130)
        if detached and watching:
            leave("Running in the background" if background else "Detached", runs)
    return detached


@app.command(cls=RepeatableOptionsCommand, no_args_is_help=True)
def benchman(
    ctx: typer.Context,
    hosts: Annotated[
        list[str],
        typer.Option(
            "--host",
            "--hosts",
            "-h",
            metavar="HOST...",
            help="This Mac: local (local-nix if Nix is installed, else local-direct), local-nix (built with Nix, as on"
            " the fleet) or local-direct (built in the checkout). Or POOL/QUEUE on the fleet. !HOST leaves hosts out.",
            autocompletion=completes(complete_hosts),
        ),
    ] = ["local"],  # noqa: B006
    engines: Annotated[
        list[str],
        typer.Option(
            "--engine",
            "--engines",
            "-e",
            metavar="ENGINE...",
            help="Engines to run each model with. !ENGINE leaves engines out.",
            autocompletion=completes(lambda _, _text: match_engines("*")),
        ),
    ] = ["uzu"],  # noqa: B006
    models: Annotated[
        list[str] | None,
        typer.Option(
            "--model",
            "--models",
            "-m",
            metavar="MODEL...",
            help="Models the engines can load; trymirai/ is implied. !MODEL leaves models out.",
            autocompletion=completes(complete_models),
        ),
    ] = None,
    datasets: Annotated[
        list[str],
        typer.Option(
            "--dataset",
            "--datasets",
            "-d",
            metavar="[spec:|!spec:][!]NAME[/KEY][^REPEATS]...",
            help="Datasets or their prompts, each run REPEATS times; after spec: only by models that decode"
            " speculatively, after !spec: only by the others; after ! left out.",
            autocompletion=completes(complete_datasets, prefix="(?:!?spec:)?!?"),
        ),
    ] = list(DEFAULT_DATASETS),  # noqa: B006
    seed: Annotated[int | None, typer.Option("--seed", "-s", help="Sampling seed. Unseeded by default.")] = None,
    filter_ram: Annotated[
        bool, typer.Option("--filter-ram", help="Leave out the models that don't fit a host's memory.")
    ] = False,
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            metavar="NAME",
            help="Instead of this checkout, the source that benchman printed on detaching or in the background.",
        ),
    ] = None,
    inputs: Annotated[
        list[str] | None,
        typer.Option(
            "--input",
            "--inputs",
            "-i",
            metavar="FILE|tag:POOL/NAME[@LABEL]...",
            help="Also show saved runs, labelled by name or LABEL; the first is the baseline. Without models, show"
            " only these.",
            autocompletion=completes(complete_tags, prefix=""),
        ),
    ] = None,
    outputs: Annotated[
        list[str] | None,
        typer.Option(
            "--output",
            "--outputs",
            "-o",
            metavar="FILE|tag:POOL/NAME...",
            help="Save the runs as JSON to a file, or tag them in the work pool's storage, where the tag gets a copy"
            " of each result, also of those that come later.",
            autocompletion=completes(complete_tags, prefix=""),
        ),
    ] = None,
    rerun: Annotated[
        bool, typer.Option("--rerun", "-rr", help="Measure everything again, also what is stored or being measured.")
    ] = False,
    wait: Annotated[
        bool, typer.Option("--wait", "-w", help="Start nothing; wait for flow runs that are measuring.")
    ] = False,
    no_run: Annotated[
        bool, typer.Option("--no-run", "-n", help="Start nothing, wait for nothing; show stored results.")
    ] = False,
    cancel: Annotated[
        bool,
        typer.Option("--cancel", "-c", help="Give up what is being measured; show stored results."),
    ] = False,
    background: Annotated[
        bool, typer.Option("--background", "-b", help="Exit once the flow runs are submitted.")
    ] = False,
    _show_completion: Annotated[
        Literal["zsh", "fish"] | None,
        typer.Option(
            "--show-completion",
            metavar="zsh|fish",
            callback=show_completion,
            is_eager=True,
            help="Print the shell's completion script.",
        ),
    ] = None,
) -> None:
    """Benchmark this checkout, with one run per host, engine and model.

    Fleet results are stored and reused. Local runs are always measured and not stored.

    Multi-value options take values separated by spaces or commas, or repeated; `m{4,5}-max` expands as in a shell.

    Several selections, separated by `+`, are measured and shown together. Each has its own hosts, engines, models,
    datasets, seed and --filter-ram; the other options go before the first `+`.

    What is named has to work; what a pattern matches is left out where it can't, like a host an engine can't use.

    Exit codes: 0 everything measured, 1 a run failed, 75 results missing.
    """

    saved = [run for source in split(inputs or []) for run in load(source)]
    baseline = saved[0].label if saved else None
    if not models and not ctx.meta["selections"]:
        if not saved:
            raise typer.BadParameter("Give models to benchmark or saved runs to show", param_hint="--model")
        raise typer.Exit(report(saved, baseline))
    if rerun + wait + no_run + cancel > 1:
        raise typer.BadParameter("--rerun, --wait, --no-run and --cancel are alternatives")
    targets = split(outputs or [])
    tags = [stored for target in targets if (stored := tag(target))]
    files = [Path(target) for target in targets if not tag(target)]
    if background and (wait or no_run or cancel or files):
        raise typer.BadParameter("There is nothing to submit, or to save yet", param_hint="--background")

    label = name_of(targets[0]) if targets else "now"
    selections = [Selection(hosts, engines, models, datasets, seed, filter_ram), *ctx.meta["selections"]]
    matched = [match_selection(selection) for selection in selections]
    selected_hosts = list(dict.fromkeys(host for matches in matched for host, _, _ in matches))
    source, archive = checkout() if source is None else (stored_source(source, selected_hosts), None)
    planned: dict[tuple[str, str, str], Run] = {}
    for selection, matches in zip(selections, matched, strict=True):
        for host, engine, model in matches:
            run = plan(source, host, engine, model, split(selection.datasets), selection.seed, label)
            if not run.tasks or (selection.filter_ram and not fits(host, engine, run.config)):
                continue
            # A run that several selections are of has the tasks of each.
            tasks = planned.setdefault((host, engine, model), run).tasks
            if differing := [name for name, task in run.tasks.items() if tasks.get(name, task) != task]:
                message = f"{differing[0]} is selected twice, differently, for {model} on {host}"
                raise typer.BadParameter(message, param_hint="--dataset")
            tasks |= run.tasks
    runs = list(planned.values())
    if not runs:
        raise typer.BadParameter("Nothing to run: the hosts, engines, models and datasets don't go together")
    how = "rerun" if rerun else "wait" if wait else "no-run" if no_run else "cancel" if cancel else "run"
    if measure(runs, how, background, tags, archive):
        return
    for file in files:
        file.write_bytes(save(runs))
    raise typer.Exit(report([*saved, *runs], baseline))
