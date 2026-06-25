# keisoku

System telemetry collector for Apple platforms — CPU/GPU/ANE power, memory, bandwidth,
frequencies, temperatures and sensors. Power is read from the SoC's IOReport/SMC counters,
so sampling runs off your work thread and adds no measurable load.

## Two ways to use it

**One-shot** — take a single reading averaged over a window:

```rust
let mut collector = keisoku::Collector::new();
let snapshot = collector.sample(std::time::Duration::from_millis(100));
if let Some(power) = snapshot.power {
    println!("gpu {} / total {}", power.gpu, power.total); // Watts
}
```

**Background recorder** — sample at an interval while something runs, then collect:

```rust
let recorder = keisoku::start(keisoku::Config {
    interval: std::time::Duration::from_millis(100),
});
// ... do work ...
let session = recorder.stop();          // stops the sampler, returns what it collected
let samples = session.snapshots.len();
```

## What you get back

- `Session { interval, snapshots }` — the snapshots sampled between `start` and `stop`.
- `Snapshot { elapsed, power, memory, cpu, gpu, neural_engine, bandwidth, temperatures, .. }`.
- `PowerMetrics { cpu, gpu, gpu_sram, ane, ram, total, package }` — all `Watts`.
- `Device { chip, gpu_cores, performance_cores, efficiency_cores, ram_total, os }`.

## Platform

Real numbers on Apple Silicon (macOS, and iOS for the subset exposed there). On other
platforms the power/SoC fields are `None` and the calls are safe no-ops, so code that
depends on keisoku still builds and runs everywhere.

Used by [`kanshi`](../kanshi) (live monitor); reaches Apple's private counters via
[`kanka`](../kanka).
