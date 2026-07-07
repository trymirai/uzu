# keisoku

System telemetry for Apple platforms — CPU/GPU/ANE power and energy, memory, bandwidth,
frequencies, temperatures and sensors. Power is read from the SoC's IOReport/SMC counters.

## Three providers

Each provider is parameterized by a compile-time tuple of metric marker types. You request
exactly the metrics you need; a single marker returns its value, a tuple returns a tuple of
values. The IOReport subscription an `Interval` builds is derived from exactly the metrics in
its template, so unused hardware groups are never subscribed.

**`Static<M>`** — device facts read once at construction (never touches IOReport):

```rust
use keisoku::{Chip, GpuCores, RamTotal, Static};

let (chip, ram_total, gpu_cores) = Static::<(Chip, RamTotal, GpuCores)>::new().into_inner();
println!("{chip}  {gpu_cores} GPU cores  {ram_total}");
```

**`Instant<M>`** — instantaneous gauges, each meaningful from a single read (RAM, temperatures,
voltage/current, fans, battery, thermal pressure, SMC package watts, rail power). No IOReport
subscription, so this is cheap:

```rust
use keisoku::{Instant, Memory, PackageWatts};

let mut gauges = Instant::<(Memory, PackageWatts)>::new();
let (memory, package) = gauges.read();
if let Some(memory) = memory {
    println!("ram {} / {}", memory.ram_usage, memory.ram_total);
}
```

**`Interval<M>`** — values measured over a window: `begin()` a session, run (or wait out) the
work, then `end()` it. The caller owns the window, so nothing in the library blocks. Build it
once and reuse it: `new` pays the IOReport subscription cost, while `begin`/`end` are cheap
counter reads. Metrics: `CpuUsage`, `GpuUsage`, `NeuralEngine`, `Power`, `Energy`, `Bandwidth`.

```rust
use keisoku::{Energy, Interval, Power};

let mut meter = Interval::<(Energy, Power)>::new();
let session = meter.begin();
// ... run the work you want to measure ...
let (energy, average_power) = meter.end(session);
println!("{} total", energy.total());                       // Joules
println!("gpu {} / total {}", average_power.gpu, average_power.total()); // Watts
```

## What you get back

- Marker values reuse the metric structs: `MemoryMetrics`, `PowerMetrics { cpu, gpu, ane, ram, package }`
  (`total()` sums the disjoint rails), `EnergyMetrics`, `CpuMetrics`, `GpuMetrics`,
  `NeuralEngineMetrics`, `BandwidthMetrics`, `FanMetrics`, `BatteryMetrics`, `Temperatures`,
  `ThermalPressure`, plus unit newtypes (`Watts`, `Joules`, `Bytes`, `Percent`, …) and `Sensor`.
- Which IOReport groups a metric decodes is declared once via `Measured::GROUPS`; a tuple's
  `GROUPS` is the compile-time union of its members, and that is exactly what the subscription
  covers.

## Platform

Apple platforms only — the crate does not build elsewhere. Full SoC power and energy come from
IOReport on macOS; on iOS keisoku reports the subset exposed there, including HID charger "wall"
power.

Used by [`kanshi`](../kanshi) (live monitor); reaches Apple's private counters via
[`kanka`](../kanka).
