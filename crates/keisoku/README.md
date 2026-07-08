# keisoku

System telemetry for Apple platforms — power, energy, memory, bandwidth, frequencies,
temperatures and sensors — read from the SoC's IOReport/SMC counters via [`kanka`](../kanka).

Two providers, each parameterized by a compile-time marker list built with `Select!`.
Read values from the returned `Sample` with `sample.get::<Marker>()`.

## `Instant<M>` — device facts and instantaneous gauges

```rust
use keisoku::{Chip, GpuCores, Instant, Memory, Select};

let sample = Instant::<Select![Chip, GpuCores, Memory]>::new().read();
println!("{}  {} GPU cores", sample.get::<Chip>(), sample.get::<GpuCores>());
```

## `Interval<M>` — values measured over a window

Build once and reuse; the caller owns timing. `new` pays the IOReport subscription cost,
`start`/`stop` are cheap counter reads.

```rust
use keisoku::{Energy, Interval, Power, Select};

let mut meter = Interval::<Select![Energy, Power]>::try_new().expect("power counters unavailable");
let session = meter.start();
// ... run work ...
let sample = meter.stop(session);
```

## Platform

Apple only. `Interval` (IOReport power/energy) is macOS-only; iOS exposes the `Instant` subset.
