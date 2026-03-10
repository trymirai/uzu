from dataclasses import dataclass


@dataclass(frozen=True)
class PerfResult:
    combo: str
    shape: str
    dispatch_path: str
    duration_ms: float
    gflops: float
    status: str
    error: str | None = None

    @property
    def m(self) -> int:
        return int(self.shape.split("x")[0])

    @property
    def k(self) -> int:
        return int(self.shape.split("x")[1])

    @property
    def n(self) -> int:
        return int(self.shape.split("x")[2])


@dataclass(frozen=True)
class BenchmarkRun:
    device: str
    results: tuple[PerfResult, ...]
