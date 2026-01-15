from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class GpuState(Enum):
    ACTIVE = "Active"
    IDLE = "Idle"
    UNKNOWN = "Unknown"


class PerformanceState(Enum):
    MINIMUM = "Minimum"
    NOMINAL = "Nominal"
    MAXIMUM = "Maximum"
    UNKNOWN = "Unknown"


class ShaderType(Enum):
    COMPUTE = "Compute"
    VERTEX = "Vertex"
    FRAGMENT = "Fragment"
    UNKNOWN = "Unknown"


class CounterType(Enum):
    PERCENTAGE = "Percentage"
    VALUE = "Value"


@dataclass(frozen=True)
class TraceMetadata:
    device_name: str
    device_model: str
    os_version: str
    target_process: str
    target_pid: int
    duration_ns: int
    start_date: str
    end_date: str
    template_name: str


@dataclass(frozen=True)
class GpuUtilization:
    run_duration_ns: int
    global_active_ns: int
    global_idle_ns: int
    global_coverage_ns: int
    target_active_ns: int
    window_start_ns: int
    window_end_ns: int
    command_buffer_count: int
    encoder_count: int

    @property
    def global_utilization_percent(self) -> float:
        if self.run_duration_ns == 0:
            return 0.0
        return self.global_active_ns / self.run_duration_ns * 100.0

    @property
    def target_utilization_percent(self) -> float:
        if self.run_duration_ns == 0:
            return 0.0
        return self.target_active_ns / self.run_duration_ns * 100.0

    @property
    def window_duration_ns(self) -> int:
        return max(0, self.window_end_ns - self.window_start_ns)

    @property
    def window_utilization_percent(self) -> float:
        window_ns = self.window_duration_ns
        if window_ns == 0:
            return 0.0
        return self.target_active_ns / window_ns * 100.0


@dataclass(frozen=True)
class GpuCounterDefinition:
    counter_id: int
    name: str
    description: str
    max_value: int
    counter_type: CounterType
    sample_interval_us: int


@dataclass(frozen=True)
class GpuPerformanceStateInterval:
    start_ns: int
    duration_ns: int
    state: PerformanceState
    is_induced: bool
    gpu_name: str


@dataclass(frozen=True)
class KernelDispatch:
    name: str
    start_ns: int
    duration_ns: int
    command_buffer_id: int
    channel: str


@dataclass(frozen=True)
class ShaderInfo:
    shader_id: int
    name: str
    shader_type: ShaderType
    pc_start: int
    pc_end: int


@dataclass(frozen=True)
class KernelStats:
    name: str
    total_ns: int
    count: int
    min_ns: int
    max_ns: int

    @property
    def avg_ns(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_ns / self.count


@dataclass(frozen=True)
class CounterSample:
    timestamp_ns: int
    counter_id: int
    value: float


@dataclass(frozen=True)
class KernelCounterStats:
    name: str
    total_ns: int
    count: int
    counters: tuple[tuple[str, float], ...]  # (counter_name, avg_value)


@dataclass(frozen=True)
class TraceExport:
    metadata: TraceMetadata
    utilization: GpuUtilization
    performance_states: tuple[GpuPerformanceStateInterval, ...] = ()
    counter_definitions: tuple[GpuCounterDefinition, ...] = ()
    dispatches: tuple[KernelDispatch, ...] = ()
    shaders: tuple[ShaderInfo, ...] = ()
    kernel_summary: tuple[KernelStats, ...] = ()
    counter_samples: tuple[CounterSample, ...] = ()
    kernel_counter_stats: tuple[KernelCounterStats, ...] = ()
