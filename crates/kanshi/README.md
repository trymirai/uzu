# kanshi

A live terminal system monitor (mactop/asitop-style) built on [`keisoku`](../keisoku).
It samples the SoC on a background thread and renders it in a `ratatui` TUI.

## Run

```bash
cargo run -p kanshi --release
```

Press `q` (or `Ctrl-C`) to quit.

## Shows

- CPU/GPU/ANE power (Watts) and usage, with history charts
- Frequencies, memory and DRAM bandwidth, temperatures
- Top processes, network and disk activity (via `sysinfo`)
- Device summary (chip, GPU cores, RAM)

## Platform

Full power/SoC telemetry is Apple Silicon (macOS) only — that's where keisoku's counters
exist. On other platforms it still runs but power/GPU panels are empty.
