import ctypes

from . import _bindings as _library


class MemoryCounters(_library.memory_counters_t):
    """Python-owned memory_counters_t; memory values are in bytes."""

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name)}" for name in self._field_names_())
        return f"{type(self).__name__}({fields})"

    def __str__(self) -> str:
        fields = []
        for name in self._field_names_():
            value = getattr(self, name)
            field = f"{name}={value}"
            if name != "pid":
                field += f" ({value / 1024**3:.3f} GiB)"
            fields.append(field)
        formatted_fields = ",\n  ".join(fields)
        return f"{type(self).__name__}(\n  {formatted_fields}\n)"


def await_cooldown(temperature: float, only_if_capped: bool) -> None:
    result = _library.await_cooldown(temperature, only_if_capped)
    if result != 0:
        raise RuntimeError(f"Cooldown failed ({result})")


def get_gpu_energy() -> float:
    """Return the cumulative GPU energy in joules, raising RuntimeError when IOReport is unavailable."""
    joules = ctypes.c_double()
    result = _library.get_gpu_energy(ctypes.byref(joules))
    if result != 0:
        raise RuntimeError(f"get_gpu_energy failed ({result})")
    return joules.value


def get_memory_counters(with_malloc_zone_stats: bool = False, *, pid: int | None = None) -> MemoryCounters:
    """Return a native counter snapshot, raising RuntimeError on collection failure.

    By default, collect the current process. Set pid to inspect another process
    owned by the same user, including a native engine subprocess.

    Set with_malloc_zone_stats=True to collect the more expensive allocator
    statistics for the current process only. Otherwise, the malloc_* fields
    remain zero.
    """
    counters = MemoryCounters()
    if pid is None:
        result = _library.get_memory_counters(ctypes.byref(counters), with_malloc_zone_stats)
    else:
        result = _library.get_memory_counters_for_pid(ctypes.byref(counters), pid, with_malloc_zone_stats)
    if result != 0:
        detail = _library.string_cast(_library.memory_counters_error_string(result))
        raise RuntimeError(f"get_memory_counters failed ({result}): {detail}")
    return counters
