# keisoku

System telemetry for Apple platforms — power, energy, memory, temperatures and sensors — read from the SoC's IOReport/SMC counters via [`kanka`](../kanka).

## `Device` — device facts and instantaneous gauges

```rust
use keisoku::Device;

let mut device = Device::new();
println!("{}  {} GPU cores", device.chip(), device.gpu_cores());
println!("battery {:?}", device.battery());
```

## `interval_measurement` — IOReport channel deltas over a window

Build once per measurement set. The caller owns timing; `start`/`stop` are cheap counter reads.

```rust
use keisoku::{AneBandwidth, Cpu, Device, DramBytes, DramRead, EnergyRail, Gpu, Select};

let mut handle = Device::interval_measurement::<Select![EnergyRail<Cpu>, EnergyRail<Gpu>, AneBandwidth, DramBytes<DramRead>]>();
handle.start();
// ... run work ...
let sample = handle.stop().expect("started");
println!("CPU energy: {}", sample.get::<EnergyRail<Cpu>>());
```

## Platform

Apple only. `interval_measurement` (IOReport) is macOS-only; iOS exposes the `Device` instant subset.
