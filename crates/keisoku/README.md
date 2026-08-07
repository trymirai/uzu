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

### Per-generation IOReport differences

The bandwidth group is named `PMP` through M4 and `PMP0` from M5 on, with `PMP1` for a second die. All
three are subscribed, so `DramHistogram` and `AneBandwidth` work across generations.

`DramBytes` reads from `AMC Stats`, and on some parts `IOReportCreateSubscription` refuses that group —
its channels remain enumerable but cannot be subscribed to. Measured: `DramBytes` works on M1, M2,
M2 Pro and base M4, and is refused on M3 Max, M4 Pro, M4 Max and M5 Max. `DramHistogram` covers
everything except M1, which reports no bandwidth. Every generation has one of the two, but neither
alone covers the whole range.

M1 also publishes no memory energy channels at all, so `EnergyRail<Ram>` is zero there.

`EnergyRail<Ram>` is the memory subsystem rather than the DRAM dies: from M5 the `DRAM`, `DCS` and
`AMCC` channels are published separately and all three are summed. They respond differently to
workload shape, so they are distinct consumers rather than duplicates.
