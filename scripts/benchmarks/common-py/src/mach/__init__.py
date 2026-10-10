import ctypes

from . import _mach


class MemoryCounters(ctypes.Structure):
    """Python-owned memory_counters_t; memory values are in bytes."""

    pid: int
    phys_footprint: int
    resident_size: int
    resident_size_peak: int
    device: int
    device_peak: int
    internal: int
    compressed: int
    graphics_footprint: int
    graphics_footprint_compressed: int
    graphics_nofootprint: int
    graphics_nofootprint_compressed: int
    graphics_total: int
    malloc_allocated: int
    malloc_in_use: int
    malloc_max_in_use: int

    # Keep field types and order identical to memory_counters_t in memory_counters.h.
    _fields_ = [
        ("pid", ctypes.c_int32),
        ("phys_footprint", ctypes.c_uint64),
        ("resident_size", ctypes.c_uint64),
        ("resident_size_peak", ctypes.c_uint64),
        ("device", ctypes.c_uint64),
        ("device_peak", ctypes.c_uint64),
        ("internal", ctypes.c_uint64),
        ("compressed", ctypes.c_uint64),
        ("graphics_footprint", ctypes.c_uint64),
        ("graphics_footprint_compressed", ctypes.c_uint64),
        ("graphics_nofootprint", ctypes.c_uint64),
        ("graphics_nofootprint_compressed", ctypes.c_uint64),
        ("graphics_total", ctypes.c_uint64),
        ("malloc_allocated", ctypes.c_uint64),
        ("malloc_in_use", ctypes.c_uint64),
        ("malloc_max_in_use", ctypes.c_uint64),
    ]

    def __repr__(self) -> str:
        fields = ", ".join(f"{name}={getattr(self, name)}" for name, _ in self._fields_)
        return f"{type(self).__name__}({fields})"

    def __str__(self) -> str:
        fields = []
        for name, _ in self._fields_:
            value = getattr(self, name)
            field = f"{name}={value}"
            if name != "pid":
                field += f" ({value / 1024**3:.3f} GiB)"
            fields.append(field)
        formatted_fields = ",\n  ".join(fields)
        return f"{type(self).__name__}(\n  {formatted_fields}\n)"


class AppleTempSensors(ctypes.Structure):
    """Python-owned apple_temp_sensors_t; sensor averages are in degrees Celsius."""

    cpu_avg: float
    gpu_avg: float

    # Keep field types and order identical to apple_temp_sensors_t in apple_temp_sensors.h.
    _fields_ = [
        ("cpu_avg", ctypes.c_double),
        ("gpu_avg", ctypes.c_double),
    ]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(cpu_avg={self.cpu_avg}, gpu_avg={self.gpu_avg})"


_library = ctypes.CDLL(_mach.__file__)
_library.get_memory_counters.argtypes = [ctypes.POINTER(MemoryCounters), ctypes.c_bool]
_library.get_memory_counters.restype = ctypes.c_int
_library.get_memory_counters_for_pid.argtypes = [ctypes.POINTER(MemoryCounters), ctypes.c_int32, ctypes.c_bool]
_library.get_memory_counters_for_pid.restype = ctypes.c_int
_library.memory_counters_error_string.argtypes = [ctypes.c_int]
_library.memory_counters_error_string.restype = ctypes.c_char_p
_library.get_temp_sensors.argtypes = [ctypes.POINTER(AppleTempSensors)]
_library.get_temp_sensors.restype = ctypes.c_int
_library.mach_error_string.argtypes = [ctypes.c_int]
_library.mach_error_string.restype = ctypes.c_char_p


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
        if not isinstance(pid, int) or isinstance(pid, bool) or not 0 < pid < 2**31:
            raise ValueError("pid must be a positive 32-bit integer")
        result = _library.get_memory_counters_for_pid(ctypes.byref(counters), pid, with_malloc_zone_stats)
    if result != 0:
        message = _library.memory_counters_error_string(result)
        detail = message.decode("utf-8", errors="replace") if message else "Unknown error"
        raise RuntimeError(f"get_memory_counters failed ({result}): {detail}")
    return counters


def get_temp_sensors() -> AppleTempSensors:
    """Return CPU/GPU sensor averages in Celsius on supported Apple Silicon Macs.

    Raise RuntimeError if sensor access fails or either average is unavailable.
    """
    sensors = AppleTempSensors()
    result = _library.get_temp_sensors(ctypes.byref(sensors))
    if result != 0:
        message = _library.mach_error_string(result)
        detail = message.decode("utf-8", errors="replace") if message else "Unknown error"
        raise RuntimeError(f"get_temp_sensors failed ({result}): {detail}")
    return sensors
