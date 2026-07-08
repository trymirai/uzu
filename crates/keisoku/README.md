# keisoku

System telemetry for Apple platforms — CPU/GPU/ANE power and energy, memory, bandwidth,
frequencies, temperatures and sensors. Power is read from the SoC's IOReport/SMC counters.

## Architecture

`keisoku` is split into three layers:

- **System layer** — private/system API calls only (`IOReport`, `SMC`, HID sensor services,
  sysctl and IOKit registry traversal). This layer owns unsafe and platform-specific details.
- **Source layer** — safe adapters over those calls. It caches expensive handles, normalizes raw
  snapshots into interval frames, prewarms metadata before measurement windows, and exposes source
  availability. IOReport channel filtering stays private here.
- **Provider layer** — the user-facing generic API (`Constant<M>`, `Instant<M>`, `Interval<M>`).
  Callers specify marker types with `Select![...]` for exactly the facts, gauges or interval
  metrics they want.

## Three providers

Each provider is parameterized by a recursive compile-time type list built with `Select!`.
The returned `Sample` supports typed access via `sample.get::<Metric>()`. `Interval`
subscribes to the compile-time union of normalized interval inputs required by the selected
metrics. Repeating a marker in a selector, such as `Select![Memory, Memory]`, is rejected at
compile time when the selector is used by a provider.

**`Constant<M>` / `Static<M>`** — device facts read once at construction (never touches IOReport):

```rust
use keisoku::{Chip, GpuCores, RamTotal, Select, Static};

let facts = Static::<Select![Chip, RamTotal, GpuCores]>::new().into_sample();
let chip = facts.get::<Chip>();
let ram_total = facts.get::<RamTotal>();
let gpu_cores = facts.get::<GpuCores>();
println!("{chip}  {gpu_cores} GPU cores  {ram_total}");
```

**`Instant<M>`** — instantaneous gauges from a single read (RAM, temperatures, voltage/current,
fans, battery, thermal pressure, rail power). No IOReport subscription:

```rust
use keisoku::{Instant, Memory, Select};

let sample = Instant::<Select![Memory]>::new().read();
if let Some(memory) = sample.get::<Memory>() {
    println!("ram {} / {}", memory.ram_usage, memory.ram_total);
}
```

**`Interval<M>`** — values measured over a window. Call `start()`, run the work, then `stop()`.
The caller owns timing; the library never sleeps. Build once and reuse: `new` pays the IOReport
subscription cost, while `start`/`stop` are cheap counter reads. Interval metrics: `CpuUsage`,
`GpuUsage`, `NeuralEngine`, `Power`, `Energy`, `Bandwidth`.

```rust
use keisoku::{Energy, Interval, Power, Select};

let mut meter = Interval::<Select![Energy, Power]>::try_new().expect("power counters unavailable");

let session = meter.start();
// ... run work ...
let sample = meter.stop(session);
let energy = sample.get::<Energy>();
let power = sample.get::<Power>();
```

## What you get back

- Marker values reuse the metric structs: `MemoryMetrics`, `PowerMetrics { cpu, gpu, ane, ram, package }`
  (`total()` sums the disjoint rails), `EnergyMetrics`, `CpuMetrics`, `GpuMetrics`,
  `NeuralEngineMetrics`, `BandwidthMetrics`, `FanMetrics`, `BatteryMetrics`, `Temperatures`,
  `ThermalPressure`, plus unit newtypes (`Watts`, `Joules`, `Bytes`, `Percent`, …) and `Sensor`.
- Interval metrics declare normalized inputs (`ENERGY_RAILS`, `CPU_RESIDENCY`, …). `sources/interval`
  maps those inputs to the minimal IOReport subscription and builds one shared frame per window.
- SMC package watts remain an internal support input for `Power`/`Energy`; it is not a public marker.

## Platform

Apple platforms only — the crate does not build elsewhere. Full SoC power and energy come from
IOReport on macOS; on iOS keisoku reports the subset exposed there, including HID charger "wall"
power.

Used by [`kanshi`](../kanshi) (live monitor); reaches Apple's private counters via
[`kanka`](../kanka).
