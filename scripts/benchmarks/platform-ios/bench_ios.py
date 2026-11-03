import importlib.util
import io
import json
import os
import plistlib
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from contextlib import suppress
from functools import cache
from pathlib import Path
from typing import Annotated, Any

import typer
from huggingface_hub import HfApi
from huggingface_hub.constants import HF_HUB_CACHE

PORT = 7171
BUNDLE_ID = "com.mirai.tech.benchmarks"
PHONE_CACHE = "Library/Caches/.cache"
MAC_CACHE = Path("~/.cache").expanduser()
DEVICE_CACHE = MAC_CACHE / "bench-ios/device"
AVAILABLE_MEMORY_PERCENT = 0.75
SPLASH_FAMILIES = Path(__file__).resolve().parents[1] / "engine-splash/deps/splash/install/families.py"

app = typer.Typer(help="Run benchmark engines inside the iOS app and stream their output back.")
Device = Annotated[
    str | None,
    typer.Option(
        "--device",
        help="UDID of the iPhone or iPad to use, remembered for later runs. Without it, the remembered one, or ask.",
    ),
]
Host = Annotated[str | None, typer.Option("--host", help="Reach the app at this address instead of through usbmux.")]
Engine = Annotated[
    str,
    typer.Option(
        "-e", "--engine", help="llamacpp, mldrift, mlx, mlxserve, mtplx, omlx, splash or uzu, as in scripts/benchmarks."
    ),
]
Model = Annotated[str, typer.Option("-m", "--model", help="Model identifier as the engine expects it.")]
Arguments = Annotated[list[str] | None, typer.Argument(help="More engine arguments, after --, as on macOS.")]


def receive(connection: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = connection.recv(size - len(data))
        if not chunk:
            raise ConnectionError("The app closed the connection")
        data += chunk
    return data


def usbmux(message: dict[str, Any]) -> tuple[socket.socket, dict[str, Any]]:
    connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    connection.connect("/var/run/usbmuxd")
    payload = plistlib.dumps({"ClientVersionString": "bench-ios", "ProgName": "bench-ios", **message})
    connection.sendall(struct.pack("<4I", 16 + len(payload), 1, 8, 1) + payload)
    length = struct.unpack("<4I", receive(connection, 16))[0]
    return connection, plistlib.loads(receive(connection, length - 16))


def coredevices() -> list[dict[str, Any]]:
    with tempfile.NamedTemporaryFile() as output:
        subprocess.run(
            ["xcrun", "devicectl", "list", "devices", "--json-output", output.name], check=True, capture_output=True
        )
        devices = json.loads(Path(output.name).read_text())["result"]["devices"]
    return [
        device
        for device in devices
        if device["hardwareProperties"].get("platform") == "iOS"
        and device["hardwareProperties"].get("reality") == "physical"
    ]


def choose_device() -> str:
    devices = coredevices()
    if not devices:
        raise SystemExit("No iPhone or iPad is paired with this Mac; connect one and trust it")
    try:
        terminal = open("/dev/tty"), open("/dev/tty", "w")  # noqa: SIM115
    except OSError:
        raise SystemExit("Pass --device UDID, or choose a device once with `bench-ios device`") from None
    with terminal[0] as keyboard, terminal[1] as screen:
        for index, device in enumerate(devices, 1):
            name, hardware = device["deviceProperties"]["name"], device["hardwareProperties"]
            state = device["connectionProperties"].get("tunnelState", "unknown")
            print(f"{index}. {name}, {hardware['marketingName']}, {hardware['udid']} ({state})", file=screen)
        print(f"Device to benchmark on [1-{len(devices)}]: ", end="", file=screen, flush=True)
        answer = keyboard.readline().strip()
    if not answer.isdigit() or not 1 <= int(answer) <= len(devices):
        raise SystemExit(f"Not a device number: {answer!r}")
    return devices[int(answer) - 1]["hardwareProperties"]["udid"]


def device_udid(udid: str | None) -> str:
    if udid is None and DEVICE_CACHE.is_file():
        return DEVICE_CACHE.read_text().strip()
    return udid or choose_device()


def remember(udid: str) -> None:
    DEVICE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    DEVICE_CACHE.write_text(udid + "\n")


@cache
def paired_device(udid: str | None) -> dict[str, Any]:
    udid = device_udid(udid)
    connection, reply = usbmux({"MessageType": "ListDevices"})
    connection.close()
    devices = [device["Properties"] for device in reply["DeviceList"] if device["Properties"]["SerialNumber"] == udid]
    if devices:
        device = min(devices, key=lambda device: device["ConnectionType"] != "USB")
    else:
        address = tunnel_address(udid)
        if address is None:
            subprocess.run(
                ["xcrun", "devicectl", "device", "info", "details", "--device", udid], capture_output=True, check=False
            )
            address = tunnel_address(udid)
        if address is None:
            raise SystemExit(
                f"Device {udid} is not connected; connect and unlock it, or choose another with `bench-ios device`"
            )
        device = {"SerialNumber": udid, "Address": address}
    remember(udid)
    return device


def tunnel_address(udid: str) -> str | None:
    return next(
        (
            device["connectionProperties"]["tunnelIPAddress"]
            for device in coredevices()
            if device["hardwareProperties"]["udid"] == udid and device["connectionProperties"].get("tunnelIPAddress")
        ),
        None,
    )


def connect(device: dict[str, Any] | None, host: str | None) -> socket.socket:
    if device is None or "Address" in device:
        return socket.create_connection((host or device["Address"], PORT))
    connection, reply = usbmux(
        {"MessageType": "Connect", "DeviceID": device["DeviceID"], "PortNumber": socket.htons(PORT)}
    )
    if reply.get("Number") != 0:
        connection.close()
        raise ConnectionRefusedError(f"usbmux refused port {PORT}: {reply}")
    return connection


def launch(device: dict[str, Any], *, fresh: bool) -> None:
    options = ["--terminate-existing"] if fresh else []
    command = ["xcrun", "devicectl", "device", "process", "launch", "--device", device["SerialNumber"], *options]
    result = subprocess.run([*command, BUNDLE_ID], capture_output=True, text=True, check=False)
    if result.returncode:
        errors = [line.strip() for line in result.stderr.splitlines() if line.strip()]
        raise SystemExit(
            f"Could not launch the app on {device['SerialNumber']}: {errors[0] if errors else result.returncode}. "
            "Unlock the device, keep it connected and install the app from Xcode first"
        )


def session(command: dict[str, Any], udid: str | None, host: str | None, *, fresh: bool = False) -> socket.socket:
    device = None if host else paired_device(udid)
    if device is not None:
        launch(device, fresh=fresh)
    for _ in range(60):
        try:
            connection = connect(device, host)
        except OSError:
            time.sleep(1)
            continue
        connection.sendall((json.dumps(command) + "\n").encode())
        return connection
    raise SystemExit("The app did not accept a connection; keep the device unlocked")


def relay(connection: socket.socket, output: io.BufferedIOBase | None = None) -> int:
    while True:
        try:
            kind, length = struct.unpack(">BI", receive(connection, 5))
            payload = receive(connection, length)
        except ConnectionError:
            raise SystemExit(
                "The app stopped before the engine finished; iOS ends apps that run out of memory or leave the "
                "foreground, so keep the device unlocked and check its memory with smaller models"
            ) from None
        if kind == 3:
            return struct.unpack(">i", payload)[0]
        stream = (output or sys.stdout.buffer) if kind == 1 else sys.stderr.buffer
        stream.write(payload)
        stream.flush()


def forward_stdin(connection: socket.socket) -> None:
    with suppress(OSError):
        while chunk := os.read(0, 1 << 16):
            connection.sendall(struct.pack(">I", len(chunk)) + chunk)
        connection.sendall(struct.pack(">I", 0))


def finish(command: dict[str, Any], udid: str | None, host: str | None) -> None:
    raise typer.Exit(relay(session(command, udid, host)))


def physical_memory(udid: str | None, host: str | None) -> int:
    output = io.BytesIO()
    if relay(session({"command": "device"}, udid, host), output) != 0:
        raise SystemExit("The app did not report its memory")
    return json.loads(output.getvalue())["physical_memory_bytes"]


def repository_files(repo: str) -> list[Any]:
    return HfApi().model_info(repo, files_metadata=True).siblings or []


def repository_size(repo: str) -> int:
    return sum(file.lfs.size for file in repository_files(repo) if file.lfs is not None)


def splash_draft_repositories() -> list[str]:
    spec = importlib.util.spec_from_file_location("splash_families", SPLASH_FAMILIES)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Bootstrap engine-splash first: {SPLASH_FAMILIES} is missing")
    families = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = families
    spec.loader.exec_module(families)
    return [family.draft_repo for family in families.FAMILIES]


def fits_in_memory(engine: str, model: str, memory: int) -> bool:
    if model.startswith(("/", "models/")):
        return True
    budget = memory * AVAILABLE_MEMORY_PERCENT
    if engine == "llamacpp":
        files = repository_files(model.partition(":")[0])
        return any(file.size is not None and file.size <= budget for file in files if file.rfilename.endswith(".gguf"))
    if engine == "splash":
        repo, _, variant = model.partition(":")
        files = repository_files(repo)
        target = sum(file.size or 0 for file in files if file.rfilename.endswith(".gguf") and variant in file.rfilename)
        return target + max(map(repository_size, splash_draft_repositories())) <= budget
    return repository_size(model) <= budget


def start(engine: str, model: str, arguments: list[str] | None, udid: str | None, host: str | None) -> socket.socket:
    if not fits_in_memory(engine, model, physical_memory(udid, host)):
        print("Skip, model is too large", file=sys.stderr)
        raise typer.Exit(1)
    command = {"command": "run", "engine": engine, "arguments": ["-m", model, *(arguments or [])]}
    return session(command, udid, host, fresh=True)


def resolve(model: str, staging: Path) -> tuple[Path, str]:
    path = Path(model).expanduser()
    if path.is_dir():
        return shutil.copytree(path, staging / path.name), f"Documents/models/{path.name}"

    mirai = MAC_CACHE / "mirai"
    registry = mirai / "registry.json"
    if registry.is_file():
        for entry in json.loads(registry.read_text()):
            repository = entry["accessibility"]["reference"]["repository"]
            if repository["identifier"] != model:
                continue
            for directory in (mirai / "models").glob(f"*/*/{repository['commit_hash']}"):
                return directory, f"{PHONE_CACHE}/mirai/{directory.relative_to(mirai)}"

    hub = Path(HF_HUB_CACHE) / f"models--{model.replace('/', '--')}"
    if hub.is_dir():
        staged = staging / hub.name
        shutil.copytree(hub, staged, ignore=shutil.ignore_patterns("blobs", ".locks"))
        return staged, f"{PHONE_CACHE}/huggingface/hub/{hub.name}"

    raise SystemExit(f"{model} is neither a directory nor cached in {MAC_CACHE}")


@app.command()
def run(engine: Engine, model: Model, arguments: Arguments = None, device: Device = None, host: Host = None) -> None:
    """Run an engine session in a fresh app process: requests on stdin, responses on stdout, logs on stderr."""
    connection = start(engine, model, arguments, device, host)
    threading.Thread(target=forward_stdin, args=(connection,), daemon=True).start()
    raise typer.Exit(relay(connection))


@app.command()
def download(
    engine: Engine, model: Model, arguments: Arguments = None, device: Device = None, host: Host = None
) -> None:
    """Load the model as the engine does, downloading it into the app's cache, without running requests."""
    connection = start(engine, model, arguments, device, host)
    connection.sendall(struct.pack(">I", 0))
    raise typer.Exit(relay(connection))


@app.command()
def device(device: Device = None) -> None:
    """Choose the iPhone or iPad to benchmark on, or remember the one given with --device."""
    udid = device or choose_device()
    if device is not None and udid not in (paired["hardwareProperties"]["udid"] for paired in coredevices()):
        raise SystemExit(f"No iPhone or iPad {udid} is paired with this Mac")
    remember(udid)
    print(udid)


@app.command()
def models(device: Device = None, host: Host = None) -> None:
    """List cached models on the device as JSON."""
    finish({"command": "models"}, device, host)


@app.command()
def delete(path: str, device: Device = None, host: Host = None) -> None:
    """Delete a cached model by the path that `models` reports."""
    finish({"command": "delete", "path": path}, device, host)


@app.command()
def push(model: str, device: Device = None) -> None:
    """Copy a model from this Mac: a uzu registry identifier, a Hugging Face repository or a directory."""
    udid = paired_device(device)["SerialNumber"]
    with tempfile.TemporaryDirectory() as staging:
        source, destination = resolve(model, Path(staging))
        subprocess.run(
            [
                *("xcrun", "devicectl", "device", "copy", "to", "--device", udid),
                *("--domain-type", "appDataContainer", "--domain-identifier", BUNDLE_ID),
                *("--source", str(source), "--destination", destination),
            ],
            check=True,
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
