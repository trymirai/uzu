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

### Per-part IOReport differences

The bandwidth group is named `PMP` on some parts and `PMP0` on others, with `PMP1` for a second die.
This does not follow the generations: an M4 Max resolves it under `PMP` while an M5 Max needs `PMP0`.
All three names are subscribed and classified, so `DramHistogram` and `AneBandwidth` work regardless of
which one a given SoC uses — do not assume a rule from the chip's generation.

Memory traffic comes from two sources and neither covers every part:

| | `DramBytes` (`AMC Stats`) | `DramHistogram` (`PMP*`) |
| --- | --- | --- |
| M1 | yes | no |
| M2, M2 Pro, M4 | yes | yes |
| M3 Max, M4 Pro, M4 Max, M5 Max | no | yes |

`DramBytes` reads from `AMC Stats`, and where it is unavailable `IOReportCreateSubscription` refuses
that group outright — the channels remain enumerable but cannot be subscribed to. Every part measured
has one of the two, but code that needs memory traffic should handle either being zero.

M1 also publishes no memory energy channels at all, so `EnergyRail<Ram>` is zero there.

`EnergyRail<Ram>` is the memory subsystem rather than the DRAM dies: where the `DRAM`, `DCS` and `AMCC`
channels are published separately, all three are summed. They respond differently to workload shape, so
they are distinct consumers rather than duplicates.
