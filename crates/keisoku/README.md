# keisoku

System telemetry for Apple platforms — CPU/GPU/ANE power and energy, memory, bandwidth,
frequencies, temperatures and sensors. Power is read from the SoC's IOReport/SMC counters.

## Three ways to read it

**Gauges** — instantaneous values, each meaningful from a single read (RAM, temperatures,
voltage/current, fans, battery, thermal pressure, SMC package watts). No IOReport subscription,
so this is cheap:

```rust
let mut collector = keisoku::Collector::new();
let gauges = collector.gauges();
if let Some(memory) = gauges.memory {
    println!("ram {} / {}", memory.ram_usage, memory.ram_total);
}
```

**Energy meter** — energy and average power over a window, measured by differencing the SoC's
cumulative counters between `start` and `stop`:

```rust
let meter = keisoku::EnergyMeter::start();
// ... run the work you want to measure ...
if let Some(reading) = meter.stop() {
    println!("{} over {}", reading.energy.total(), reading.elapsed);        // Joules / ms
    println!("gpu {} / total {}", reading.average_power.gpu, reading.average_power.total()); // Watts
}
```

**Live snapshot** — everything at once, averaged over a window (gauges plus the windowed rates):

```rust
let mut collector = keisoku::Collector::new();
let snapshot = collector.sample(std::time::Duration::from_millis(100));
if let Some(power) = snapshot.power {
    println!("gpu {} / total {}", power.gpu, power.total()); // Watts
}
```

## What you get back

- `Gauges { memory, fans, battery, temperatures, thermal_pressure, package_watts, sensors, voltage, current }`.
- `EnergyReading { energy, average_power, elapsed, package_from_smc }`.
- `Snapshot { elapsed, cpu, gpu, neural_engine, power, memory, bandwidth, temperatures, .. }`.
- `PowerMetrics { cpu, gpu, gpu_sram, ane, ram, package }` — all `Watts`; `total()` sums the disjoint rails.
- `Device { chip, gpu_cores, performance_cores, efficiency_cores, ram_total, os }`.

## Platform

Apple platforms only — the crate does not build elsewhere. Full SoC power and energy come from
IOReport on macOS; on iOS keisoku reports the subset exposed there, including HID charger "wall"
power.

Used by [`kanshi`](../kanshi) (live monitor); reaches Apple's private counters via
[`kanka`](../kanka).
