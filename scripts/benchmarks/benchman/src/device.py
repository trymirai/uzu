from typing import NamedTuple

from benchman.subprocess import subprocess_output
from pydantic import BaseModel


class Hardware(NamedTuple):
    """What a host's Macs have: the bytes of memory of the smallest, and their chips as the fleet tags them."""

    memory: int
    chips: frozenset[str]


class Device(BaseModel):
    model: str
    chip: str
    memory: int
    os: str


async def read_device() -> Device:
    names = ("hw.model", "machdep.cpu.brand_string", "hw.memsize", "kern.osproductversion")
    model, chip, memory, os_version = (await subprocess_output("sysctl", "-n", *names)).splitlines()
    return Device(model=model, chip=chip, memory=int(memory), os=os_version)


async def read_hardware() -> Hardware:
    device = await read_device()
    return Hardware(device.memory, frozenset({device.chip.removeprefix("Apple ").lower().replace(" ", "-")}))
