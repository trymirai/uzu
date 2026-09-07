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

## Performance control

The opt-in `hardware-control` feature exposes macOS-only `GpuPowerControl`,
`PowerModeControl` and `FanControl`. They reuse keisoku's IOKit/SMC access; private
functions resolve through `kanka`, and private service/property/key names are
obfuscated. Construction is read-only. The controllers are not exported without
the feature or on iOS.

`GpuPowerControl::set_power_limit_milliwatts` sets a GPU power **ceiling**, checks
the driver's readback, and preserves the original ceiling until `restore` succeeds.
Raising it permits a larger budget; it does not force power consumption, clocks,
voltage, or performance. There is no universal "uncapped" value: restoration uses
the exact original value. The reported ceiling changes under system policy and is
not a maximum supported wattage. Package-wide CLPC control is not used.

`GpuPowerControl::maximum_power_milliwatts` reads the driver's separately calibrated
budget for the highest performance state. On the tested M5 Max, this is 203,110 mW;
the current ceiling changes independently. This budget is neither actual power draw
nor an enforced upper bound or a promise of sustainable power or frequency. The
current policy ceiling can exceed it. macOS can overwrite a ceiling set through
`set_power_limit_milliwatts`, so a successful write does not lock the GPU.

`PowerModeControl::set_high` requests macOS High Power Mode on supported power
sources, preserving each source's original mode. It checks capability and verifies
the saved preference after each write. macOS controls the resulting GPU budget;
there is no fabricated wattage maximum. Unsupported devices return an error.

`FanControl::set_maximum` validates each fan's hardware-reported maximum and saves
its original mode, target and unlock state before changing anything. `restore`
attempts all saved writes and reports failures. The controllers roll back failed
changes and retry pending restoration on Drop. Call `restore` explicitly to handle
its result. A driver can reject restoration; SIGKILL, process abort or power loss
cannot run Drop. The library does not provide a watchdog daemon or coordinate with
other fan-control applications.

The CLI enables this feature by default. Open `/settings` (a model is not required):

- **Auto:** leave system settings alone, or restore the settings captured before Fast.
- **Fast:** request High Power Mode and set every fan to its hardware-reported maximum RPM.

This is one **Mode** setting combining power and fans. Space or left/right changes
the draft; Enter applies and verifies both controls, and Escape cancels the draft.
Fast uses the system's performance policy instead of setting an arbitrary AGX cap.

Overrides are session-only and never saved in the preferences file. Run the CLI as
your normal user: selecting Fast requests administrator authorization through macOS.
A session helper owns the hardware controls while the CLI keeps its normal user
identity, model access and preferences. Further mode changes reuse that helper until
the CLI exits. Cancelling authorization leaves Auto active and permits retrying.
macOS still requires this initial authorization; the CLI cannot bypass it.

The helper accepts only Auto/Fast requests through a private Unix socket, verifies
kernel peer credentials, and restores on connection loss, including a parent crash.
Normal exit waits for its restoration result. Ctrl+C, SIGINT, SIGTERM and SIGHUP
also attempt checked restoration; killing the helper itself with SIGKILL cannot.
Only one CLI session can own overrides at a time. While a hardware update is pending,
Escape waits for the operation to finish. Unsupported hardware and authorization
failures are reported. This does not install a daemon or modify sudoers.

Read capabilities without changing settings:

```sh
cargo run -p keisoku --example hardware_status --features hardware-control
```

To build a CLI without these controls:

```sh
cargo build -p cli --no-default-features --features backend-cpu,backend-metal,capability-grammar
```

Protocol references: [apple-gpu-dvfs](https://github.com/maderix/apple-gpu-dvfs/tree/4f903a721c00dbf24e96c75d5f1e4e37f7ff8053)
and [ThermalForge](https://github.com/ProducerGuy/ThermalForge/tree/3fbaa527aee05a5a0ed2606f00b50254df9d614f).
This is a Rust implementation using the existing keisoku transport, with no bundled
upstream executable or runtime dependency. Hardware support must be checked at runtime.
High Power Mode follows Apple's [power-management API](https://github.com/apple-oss-distributions/IOKitUser/blob/main/pwr_mgt.subproj/IOPMLibPrivate.h)
and [supported Mac modes](https://support.apple.com/en-us/101613).

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
